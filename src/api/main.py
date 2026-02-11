"""
FastAPI backend for serving adversarial robustness demo.

Endpoints:
- POST /predict: Get predictions from both standard and robust models
- POST /attack: Apply an adversarial attack and return predictions + visualization
- GET /health: Health check
"""

import io
import base64
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torchvision.transforms as transforms

from src.models.resnet import resnet18_cifar10
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.deepfool import deepfool_attack
from src.utils.data import CIFAR10_CLASSES


app = FastAPI(
    title="Adversarial Robustness Demo",
    description="Compare standard vs. robust neural networks under adversarial attack",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
standard_model = None
robust_model = None

CHECKPOINT_DIR = Path("checkpoints")


def load_models():
    """Load both standard and robust models."""
    global standard_model, robust_model

    standard_path = CHECKPOINT_DIR / "standard_best.pt"
    robust_path = CHECKPOINT_DIR / "robust_best.pt"

    if standard_path.exists():
        standard_model = resnet18_cifar10()
        checkpoint = torch.load(standard_path, map_location=device, weights_only=True)
        standard_model.load_state_dict(checkpoint["model_state_dict"])
        standard_model.to(device).eval()
        print(f"✓ Standard model loaded (acc: {checkpoint.get('test_accuracy', 'N/A')}%)")

    if robust_path.exists():
        robust_model = resnet18_cifar10()
        checkpoint = torch.load(robust_path, map_location=device, weights_only=True)
        robust_model.load_state_dict(checkpoint["model_state_dict"])
        robust_model.to(device).eval()
        print(f"✓ Robust model loaded (PGD acc: {checkpoint.get('test_pgd_accuracy', 'N/A')}%)")


@app.on_event("startup")
async def startup():
    load_models()


# --- Schemas ---
class PredictionResult(BaseModel):
    class_name: str
    confidence: float
    all_probabilities: dict[str, float]


class AttackResult(BaseModel):
    attack_name: str
    epsilon: float
    standard_clean: PredictionResult
    standard_attacked: PredictionResult
    robust_clean: PredictionResult | None
    robust_attacked: PredictionResult | None
    adversarial_image_b64: str  # Base64 encoded PNG
    perturbation_image_b64: str  # Amplified perturbation visualization
    l2_norm: float
    linf_norm: float


# --- Helpers ---
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess uploaded image to CIFAR-10 format."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # (1, 3, 32, 32)


def get_prediction(model: torch.nn.Module, image: torch.Tensor) -> PredictionResult:
    """Get model prediction with confidence scores."""
    with torch.no_grad():
        output = model(image.to(device))
        probs = F.softmax(output, dim=1)[0]
        conf, pred = probs.max(0)

    all_probs = {CIFAR10_CLASSES[i]: round(probs[i].item(), 4) for i in range(10)}
    return PredictionResult(
        class_name=CIFAR10_CLASSES[pred.item()],
        confidence=round(conf.item(), 4),
        all_probabilities=all_probs,
    )


def tensor_to_b64(tensor: torch.Tensor) -> str:
    """Convert image tensor to base64 PNG string."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((256, 256), Image.NEAREST)  # Upscale for visibility
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --- Endpoints ---
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "standard_model_loaded": standard_model is not None,
        "robust_model_loaded": robust_model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Get predictions from both models on a clean image."""
    if standard_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    image_bytes = await file.read()
    image = preprocess_image(image_bytes)

    result = {"standard": get_prediction(standard_model, image).model_dump()}
    if robust_model is not None:
        result["robust"] = get_prediction(robust_model, image).model_dump()

    return result


@app.post("/attack", response_model=AttackResult)
async def attack(
    file: UploadFile = File(...),
    attack_name: str = "fgsm",
    epsilon: float = 0.031373,
):
    """Apply an adversarial attack and compare model predictions."""
    if standard_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    image_bytes = await file.read()
    clean_image = preprocess_image(image_bytes).to(device)

    # We need a label for targeted attacks - use the standard model's prediction
    with torch.no_grad():
        pred_label = standard_model(clean_image).argmax(dim=1)

    # Generate adversarial example
    if attack_name == "fgsm":
        adv_image = fgsm_attack(standard_model, clean_image, pred_label, epsilon, device)
    elif attack_name == "pgd":
        adv_image = pgd_attack(
            standard_model, clean_image, pred_label,
            epsilon=epsilon, alpha=epsilon / 4, steps=20,
            random_start=True, device=device,
        )
    elif attack_name == "deepfool":
        adv_image, _ = deepfool_attack(standard_model, clean_image, device=device)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown attack: {attack_name}")

    # Compute perturbation metrics
    perturbation = adv_image - clean_image
    l2_norm = perturbation.flatten().norm(2).item()
    linf_norm = perturbation.abs().max().item()

    # Amplified perturbation for visualization (scale to [0, 1])
    pert_vis = perturbation.abs()
    if pert_vis.max() > 0:
        pert_vis = pert_vis / pert_vis.max()

    # Get predictions
    result = AttackResult(
        attack_name=attack_name,
        epsilon=epsilon,
        standard_clean=get_prediction(standard_model, clean_image),
        standard_attacked=get_prediction(standard_model, adv_image),
        robust_clean=get_prediction(robust_model, clean_image) if robust_model else None,
        robust_attacked=get_prediction(robust_model, adv_image) if robust_model else None,
        adversarial_image_b64=tensor_to_b64(adv_image),
        perturbation_image_b64=tensor_to_b64(pert_vis),
        l2_norm=round(l2_norm, 6),
        linf_norm=round(linf_norm, 6),
    )

    return result
