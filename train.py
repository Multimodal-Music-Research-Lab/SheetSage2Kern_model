import gc

import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from my_utils.ar_dataset import ARDataModule
from my_utils.consts import CNN_ENCODER, DATASET_NAME, VALID_ENCODERS
from my_utils.seed import seed_everything
from networks.transformer.model import A2STransformer

seed_everything(42, benchmark=False)
PROJECT_NAME = "HookKernModel"


def train(
    model_type: str = "transformer",
    attn_window: int = -1,
    use_voice_change_token: bool = False,
    epochs: int = 1000,
    patience: int = 20,
    batch_size: int = 16,
    check_val_every_n_epoch: int = 5,
    encoder=CNN_ENCODER,
):
    gc.collect()
    torch.cuda.empty_cache()

    # Experiment info
    print("TRAIN EXPERIMENT")
    print(f"\tAttention window: {attn_window} (Used if model type is transformer)")
    print(f"\tUse voice change token: {use_voice_change_token}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tBatch size: {batch_size}")
    print(f"\tCheck Val Every N epoch: {check_val_every_n_epoch}")

    if encoder not in VALID_ENCODERS:
        raise ValueError(encoder)

    # Data module
    datamodule = ARDataModule(
        use_voice_change_token=use_voice_change_token,
        batch_size=batch_size,
        encoder_name=encoder,
    )
    datamodule.setup(stage="fit")
    w2i, i2w = datamodule.get_w2i_and_i2w()

    # Model
    model = A2STransformer(
        max_seq_len=datamodule.get_max_seq_len(),
        max_audio_len=datamodule.get_max_audio_len(),
        max_encoder_output_length=datamodule.get_max_encoder_output_len(),
        w2i=w2i,
        i2w=i2w,
        attn_window=attn_window,
        teacher_forcing_prob=0.2,
        encoder=encoder,
    )

    # Train, validate and test
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{model_type}/{encoder}"
            if not use_voice_change_token
            else f"weights/{model_type}-VCT",
            filename=DATASET_NAME,
            monitor="val_loss",
            # monitor="val_sym-er",
            verbose=True,
            save_last=False,
            save_top_k=1,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=5,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            # monitor="val_sym-er",
            monitor="val_loss",
            min_delta=0.01,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=100.00,
            check_on_train_epoch_end=False,
        ),
    ]
    trainer = Trainer(
        logger=WandbLogger(
            project=PROJECT_NAME + "_test2",
            group=f"{model_type}"
            if not use_voice_change_token
            else f"{model_type}-VCT",
            # name=f"Train-{ds_name}_Test-{ds_name}",
            name=f"Hooktheory dataset test run with {encoder}",
            log_model=False,
        ),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        deterministic=False,  # If True, raises error saying that CTC loss does not have this behaviour
        benchmark=False,
        precision="16-mixed",  # Mixed precision training
    )
    trainer.fit(model, datamodule=datamodule)
    model = A2STransformer.load_from_checkpoint(callbacks[0].best_model_path)
    model.freeze()
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(train)
