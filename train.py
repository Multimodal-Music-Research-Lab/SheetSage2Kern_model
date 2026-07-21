import gc
from pathlib import Path

import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from my_utils.ar_dataset import ARDataModule
from my_utils.consts import (
    CNN_ENCODER,
    MID_LEVEL_TOKENISER,
    PROJECT_NAME,
    VALID_ENCODERS,
)
from my_utils.seed import seed_everything
from networks.transformer.model import A2STransformer

seed_everything(42, benchmark=False)


def train(
    ds_location: str,
    attn_window: int = -1,
    epochs: int = 1000,
    patience: int = 20,
    batch_size: int = 8,
    train_subset_size=None,
    check_val_every_n_epoch: int = 5,
    encoder=CNN_ENCODER,
    weight_decay=0.0,
    learning_rate=1e-4,
    ff_dim_multiplier=1,
    resume_from=None,
    label_smoothing=0.0,
    tokeniser=MID_LEVEL_TOKENISER,
    use_pre_norm=True,
):
    gc.collect()
    torch.cuda.empty_cache()

    dataset_name = Path(ds_location).stem
    # Experiment info
    print("TRAIN EXPERIMENT")
    print(f"\tAttention window: {attn_window} (Used if model type is transformer)")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tBatch size: {batch_size}")
    print(f"\tTrain subset size: {train_subset_size}")
    print(f"\tCheck Val Every N epoch: {check_val_every_n_epoch}")
    print(f"\tlearning rate: {learning_rate}")
    print(f"\tweight decay : {weight_decay}")
    print(f"\tencoder : {encoder}")
    print(f"\tff_dim_multiplier : {ff_dim_multiplier}")
    print(f"\tResuming from  : {resume_from}")
    print(f"\tLabel smoothing  : {label_smoothing}")
    print(f"\ttokeniser  : {tokeniser}")
    print(f"\tUse pre norm  : {use_pre_norm}")

    if encoder not in VALID_ENCODERS:
        raise ValueError(encoder)

    # Data module
    datamodule = ARDataModule(
        ds_name=dataset_name,
        ds_location=ds_location,
        batch_size=batch_size,
        train_subset_size=train_subset_size,
        encoder_name=encoder,
        tokeniser=tokeniser,
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
        lr=learning_rate,
        weight_decay=weight_decay,
        ff_dim_multiplier=ff_dim_multiplier,
        label_smoothing=label_smoothing,
        use_pre_norm=use_pre_norm,
    )

    # Train, validate and test

    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{encoder}",
            filename=dataset_name,
            monitor="val_loss",
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
            monitor="val_loss",
            min_delta=0.001,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=100.00,
            check_on_train_epoch_end=False,
        ),
    ]
    assert 8 % batch_size == 0 and 8 >= batch_size
    accumulate_batches = 8 // batch_size
    trainer = Trainer(
        logger=WandbLogger(
            project=PROJECT_NAME,
            group=f"{encoder} {dataset_name} {tokeniser}",
            name=f"Train-{encoder}-{dataset_name}",
            log_model=False,
        ),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        deterministic=False,
        benchmark=False,
        precision="16-mixed",  # Mixed precision training
        accumulate_grad_batches=accumulate_batches,
    )
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=resume_from,
    )


if __name__ == "__main__":
    fire.Fire(train)
