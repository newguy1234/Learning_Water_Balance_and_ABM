"""Upload this repository to a Hugging Face Space (Docker SDK).

    pip install huggingface_hub
    hf auth login                                   # paste a token with write access
    python deploy/push_to_space.py <username>/<space-name>

Creates the Space if it does not exist, uploads the code, data and Dockerfile, and
uses deploy/space_README.md as the Space's README (it carries the Space config).
Large data files go up through Git LFS automatically. Re-run it to redeploy.
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IGNORE = [
    ".git/*", ".venv/*", ".venv-*/*", "outputs/*", ".presentation-build/*",
    ".pytest_cache/*", ".claude/*", "**/__pycache__/*", "*.pyc", "data/copernicus/*",
    "README.md",  # replaced by deploy/space_README.md below
]


def main(repo_id):
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=ROOT, repo_id=repo_id, repo_type="space",
                      ignore_patterns=IGNORE, commit_message="Deploy from repository")
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "deploy", "space_README.md"),
                    path_in_repo="README.md", repo_id=repo_id, repo_type="space",
                    commit_message="Space README")
    print(f"Deployed. Build log and app: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit(__doc__)
    main(sys.argv[1])
