import gc
import os
import random
from pathlib import Path
from typing import Literal

import fire
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from tqdm import tqdm

from my_utils.ar_dataset import ARDataModule
from my_utils.consts import EOS_TOKEN, PREPROCESSED_MUQ_ENCODER, PROJECT_NAME, SOS_TOKEN
from my_utils.metrics import compute_metrics
from my_utils.metrics_original import compute_metrics_legacy
from networks.transformer.model import A2STransformer


def greedy_decode(
    model,
    x,
    start_token,
    end_token,
):
    device = x.device
    y_in = torch.zeros(1, model.max_seq_len + 1, dtype=torch.long, device=device)
    y_in[0, 0] = start_token
    with torch.inference_mode():
        x = model.encoder(x=x)
        yhat = []
        for step in range(model.max_seq_len):
            y_out_hat = model.decoder(
                tgt=y_in[:, : step + 1], memory=x, memory_len=None
            )
            y_out_hat = y_out_hat[0, :, -1]  # Last token
            y_out_hat_token = y_out_hat.argmax(dim=-1).item()
            if y_out_hat_token == end_token:
                break
            yhat.append(y_out_hat_token)
            y_in[0, step + 1] = y_out_hat_token

    return yhat


@torch.inference_mode()
def run_metrics(
    checkpoint_path: str = "",
    ds_location: str = "",
    encoder: str = PREPROCESSED_MUQ_ENCODER,
    max_samples: int = -1,
    print_random_sample: bool = True,
    tokeniser="word",
    dataset_type: Literal["lead_sheet", "quartets"] = "lead_sheet",
):
    gc.collect()
    torch.cuda.empty_cache()

    if checkpoint_path == "" or not os.path.exists(checkpoint_path):
        print(f"Invalid checkpoint path: {checkpoint_path}")
        return

    dataset_name = Path(ds_location).stem
    wandb_logger = WandbLogger(
        project=PROJECT_NAME + "_metrics",
        group=f"{Path(ds_location).stem} {encoder} metrics",
        name=f"Metrics-{encoder}-{dataset_name}",
        log_model=False,
    )
    wandb_logger.experiment.config.update(
        {
            "checkpoint_path": checkpoint_path,
            "ds_location": ds_location,
            "encoder": encoder,
            "split": "test",
            "max_samples": max_samples,
            "print_random_sample": print_random_sample,
        }
    )

    datamodule = ARDataModule(
        ds_name=dataset_name,
        ds_location=ds_location,
        batch_size=1,
        encoder_name=encoder,
        tokeniser=tokeniser,
        num_workers=1,
    )

    datamodule.setup(stage="test")

    token_ds = datamodule.test_ds

    model = A2STransformer.load_from_checkpoint(
        checkpoint_path, ytest_i2w=token_ds.i2w, strict=True
    )
    model.requires_grad_(False)
    model.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    y_true = []
    y_pred = []

    total = len(datamodule.test_dataloader())
    if max_samples > 0:
        total = min(total, max_samples)

    iterator = tqdm(datamodule.test_dataloader(), total=total, desc="test metrics")
    for idx, batch in enumerate(iterator):
        if max_samples > 0 and idx >= max_samples:
            break
        x, y, fn = batch
        x = x.to(device)
        y = y.to(device)
        start_token = token_ds.w2i[SOS_TOKEN]
        end_token = token_ds.w2i[EOS_TOKEN]
        pred_token_ids = greedy_decode(
            model=model,
            x=x,
            start_token=start_token,
            end_token=end_token,
        )

        y = [token_ds.i2w[i.item()] for i in y[0]]
        y_true.append([x for x in y if x not in {SOS_TOKEN, EOS_TOKEN}])
        yhat = [token_ds.i2w[t] for t in pred_token_ids]
        y_pred.append(yhat)
    if dataset_type == "lead_sheet":
        metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    else:
        metrics = compute_metrics_legacy(y_true=y_true, y_pred=y_pred)

    print("\nMETRICS (test)")
    for key in sorted(metrics.keys()):
        print(f"{key}: {metrics[key]}")

    wandb_logger.experiment.log(metrics)
    if print_random_sample:
        sample_idx = random.randint(0, len(y_true) - 1)
        print("\nRANDOM SAMPLE (split)")
        print(f"index: {sample_idx}")
        print(f"ground_truth: {y_true[sample_idx]}")
        print(f"prediction: {y_pred[sample_idx]}")

    wandb_logger.experiment.finish()


if __name__ == "__main__":
    fire.Fire(run_metrics)
