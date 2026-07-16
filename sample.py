import base64
import os
from pathlib import Path

import fire
import torch

from my_utils.ar_dataset import ARDataModule
from my_utils.consts import PREPROCESSED_MUQ_ENCODER
from my_utils.seed import seed_everything
from my_utils.tokeniser import untokenize
from my_utils.word_level_tokeniser import STEP_CHANGE_TOKEN, VOICE_CHANGE_TOKEN
from networks.transformer.model import A2STransformer

seed_everything(42, benchmark=False)


def add_more_kern_information(body, comments: list[str], quartets=False):
    comments = ["!! " + c for c in comments]
    if quartets:
        return "\n".join(comments + [body])

    body_lines = [line for line in body.split("\n") if line.strip()]
    num_spines = max((line.count("\t") + 1 for line in body_lines), default=2)
    assert 2 <= num_spines <= 3

    spine_types = ["**kern"] + ["**cdata"] + ["**text"]
    spine_types = spine_types[:num_spines]
    clefs = ["*clefG2"] + ["*"] * (num_spines - 1)

    headers = ["\t".join(spine_types), "\t".join(clefs)]
    footer = ["\t".join(["=="] * num_spines), "\t".join(["*-"] * num_spines)]
    return "\n".join(comments + headers + [body] + footer)


def view_kern_in_browser(kern_data, comments: list[str]):
    kern_data = add_more_kern_information(kern_data, comments)
    kern_bytes = kern_data.encode("utf-8")

    base64_bytes = base64.b64encode(kern_bytes)

    mime_encoded_kern = base64_bytes.decode("utf-8")

    base_url = "https://verovio.humdrum.org/"
    final_url = f"{base_url}?t={mime_encoded_kern}"
    print(final_url)


def untokenise_quartets(kern):
    out = "\t".join(["**kern"] * 4) + "\n"
    line = []
    for token in kern:
        if token == STEP_CHANGE_TOKEN:
            if len(line) > 0:
                if len(line) < 4:
                    line.extend(["."] * (4 - len(line)))
                out += "\t".join(line) + "\n"
            line = []
        else:
            if token != "DOT" and token != VOICE_CHANGE_TOKEN:
                line.append(token)
            else:
                line.append(".")
    return out


def test(
    checkpoint_path: str = "", ds_location: str = "", number: int = 0, tokeniser="word"
):
    if checkpoint_path == "" or not os.path.exists(checkpoint_path):
        print("Invalid checkpoint path.")
        return
    if tokeniser == "original":
        from my_utils.metrics_original import compute_metrics

        _untokenise = untokenise_quartets
    else:
        from my_utils.metrics import compute_metrics

        _untokenise = untokenize
    print("loading datamodule")
    dataset_name = Path(ds_location).stem
    datamodule = ARDataModule(
        encoder_name=PREPROCESSED_MUQ_ENCODER,
        num_workers=0,
        batch_size=1,
        ds_name=dataset_name,
        ds_location=ds_location,
        tokeniser=tokeniser,
    )
    datamodule.setup(stage="test")
    ytest_i2w = datamodule.test_ds.i2w

    print("loading model")
    model = A2STransformer.load_from_checkpoint(checkpoint_path, ytest_i2w=ytest_i2w)
    model.freeze()
    model.eval()

    print("running inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    dataloader = datamodule.test_dataloader()
    i = -1
    for batch in dataloader:
        i += 1
        print(i)
        if i >= number:
            break
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        x, y_out, fn = batch
        start_token = model.w2i["<SOS>"]
        end_token = model.w2i["<EOS>"]

        y_in = torch.tensor([start_token]).unsqueeze(0).long().to(x.device)

        print(f"\n--- Sample ---")
        pred_tokens = []

        for _ in range(datamodule.get_max_seq_len()):
            y_out_hat = model(x, None, y_in)
            y_out_hat = y_out_hat[0, :, -1]
            y_out_hat_token = y_out_hat.argmax(dim=-1).item()
            y_out_hat_word = model.i2w[y_out_hat_token]
            if y_out_hat_token == end_token:
                break
            pred_tokens.append(y_out_hat_word)

            y_in = torch.cat(
                [y_in, torch.tensor([[y_out_hat_token]]).long().to(x.device)], dim=1
            )
        gt_tokens = [
            model.i2w[t.item()]
            for t in y_out[0]
            if t.item() not in [start_token, end_token]
        ]

        print("file name: ", fn[0])
        print(f"https://hookpad.hooktheory.com/?idOfSong={fn[0]}")
        print("GT: ")
        print(gt_tokens)
        view_kern_in_browser(_untokenise(gt_tokens), ["Ground truth"])
        print()
        print()
        print()
        print("PRED: ")
        view_kern_in_browser(_untokenise(pred_tokens), ["Model prediction"])
        print()
        print()
        print()
        print()
        metrics = compute_metrics(y_true=[gt_tokens], y_pred=[pred_tokens])
        print(metrics)


if __name__ == "__main__":
    fire.Fire(test)
