"""
Deploy the Streamlit demo to a HuggingFace Space (Docker SDK).

Uploads only the files the demo needs --- the Dockerfile, requirements, the
Streamlit app, the src package, the configs, and the two model checkpoints ---
plus a Space README with the required YAML frontmatter. It does NOT push the
thesis, data, results, notebooks, or tests.

Auth: uses an existing `huggingface-cli login` token (or HF_TOKEN env var).

Usage:
    python scripts/deploy_hf.py
    python scripts/deploy_hf.py --repo-id omaralsafarti/adversarial-robustness-demo
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from huggingface_hub import HfApi, get_token

SPACE_README = """\
---
title: Adversarial Robustness Demo
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Adversarial Robustness Benchmark

Interactive demo for my MSc thesis *Adversarial Attacks on Deep Neural
Networks: Generation, Evaluation, and Robust Training*. Pick a CIFAR-10 image
(or upload your own), choose an attack (FGSM, PGD, or DeepFool) and a
perturbation budget, and compare how a **standard** ResNet-18 and a
**PGD adversarially trained** ResNet-18 respond.

The standard model reaches 94.1% clean accuracy but drops to 1.2% under a
20-step PGD attack at epsilon = 8/255; the adversarially trained model keeps
45.1% under the same attack. Source and full thesis:
https://github.com/Brlr2003/adversarial-robustness
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="omaralsafarti/adversarial-robustness-demo")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        sys.exit("No HF token. Run `huggingface-cli login` or set HF_TOKEN.")

    api = HfApi(token=token)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Sanity: checkpoints must exist or the Space will start with no models.
    for ck in ["checkpoints/standard_best.pt", "checkpoints/robust_best.pt"]:
        if not os.path.exists(os.path.join(root, ck)):
            sys.exit(f"Missing {ck}; train or place the checkpoint first.")

    print(f"Creating Space {args.repo_id} (docker) ...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    # Space README with frontmatter (uploaded as README.md).
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(SPACE_README)
        readme_path = f.name
    print("Uploading Space README ...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="space",
    )
    os.unlink(readme_path)

    print("Uploading code and checkpoints (checkpoints go through LFS) ...")
    api.upload_folder(
        folder_path=root,
        repo_id=args.repo_id,
        repo_type="space",
        allow_patterns=[
            "Dockerfile",
            "requirements.txt",
            "frontend/**",
            "src/**",
            "configs/**",
            "checkpoints/*.pt",
        ],
        ignore_patterns=["**/__pycache__/**", "*.pyc"],
        commit_message="Deploy adversarial robustness demo",
    )

    url = f"https://huggingface.co/spaces/{args.repo_id}"
    print(f"\nDone. Space: {url}")
    print("First build takes a few minutes (installing torch + building the image).")


if __name__ == "__main__":
    main()
