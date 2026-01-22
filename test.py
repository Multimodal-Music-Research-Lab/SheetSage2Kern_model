import gc
import os

import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from my_utils.ar_dataset import ARDataModule
from my_utils.seed import seed_everything
from networks.transformer.model import A2STransformer
from train import PROJECT_NAME

seed_everything(42, benchmark=False)


def test(
    model_type: str = "transformer",
    use_voice_change_token: bool = False,
    checkpoint_path: str = "",
):
    gc.collect()
    torch.cuda.empty_cache()

    # Check if checkpoint path is empty or does not exist
    if checkpoint_path == "":
        print("Checkpoint path not provided")
        return
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint path {checkpoint_path} does not exist")
        return

    # Get source dataset name
    src_ds_name = os.path.basename(checkpoint_path).split(".")[0]

    # Experiment info
    print("TEST EXPERIMENT")
    print(f"\tSource dataset: {src_ds_name}")
    print(f"\tModel type: {model_type}")
    print(f"\tUse voice change token: {use_voice_change_token}")
    print(f"\tCheckpoint path: {checkpoint_path}")

    # Data module
    datamodule = ARDataModule(
        use_voice_change_token=use_voice_change_token,
    )
    datamodule.setup(stage="test")
    ytest_i2w = datamodule.test_ds.i2w

    # Model
    model = A2STransformer.load_from_checkpoint(checkpoint_path, ytest_i2w=ytest_i2w)

    # Test
    trainer = Trainer(
        logger=WandbLogger(
            project=PROJECT_NAME,
            group=f"{model_type}"
            if not use_voice_change_token
            else f"{model_type}-VCT",
            name=f"Test run",
            log_model=False,
        ),
        precision="16-mixed",  # Mixed precision training
    )
    model.freeze()
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(test)
