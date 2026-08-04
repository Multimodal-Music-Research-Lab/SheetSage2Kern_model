import os

from huggingface_hub import hf_hub_download


def resolve_checkpoint(checkpoint_path: str) -> str:
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    parts = checkpoint_path.split("/", 2)
    if len(parts) < 3:
        raise FileNotFoundError(
            f"{checkpoint_path!r} is not a local file or an org/repo/file.ckpt id."
        )

    repo_id = f"{parts[0]}/{parts[1]}"
    print(f"Downloading {parts[2]} from {repo_id}...")
    return hf_hub_download(repo_id=repo_id, filename=parts[2])
