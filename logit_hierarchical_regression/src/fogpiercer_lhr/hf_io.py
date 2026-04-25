from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def push_model_dir(model_dir: Path, repo_id: str, token: str | None = None) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=".",
    )


def download_model_dir(repo_id: str, token: str | None = None) -> Path:
    return Path(snapshot_download(repo_id=repo_id, repo_type="model", token=token))

