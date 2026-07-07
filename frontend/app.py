"""
Streamlit frontend for Adversarial Robustness Demo.

This is the main demo app that gets deployed to HuggingFace Spaces.
It loads models directly (no API needed) for simplicity in deployment.

Run locally: streamlit run frontend/app.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets

from src.attacks.cw import cw_l2_attack
from src.attacks.deepfool import deepfool_attack
from src.attacks.fgsm import fgsm_attack
from src.attacks.hopskipjump import hopskipjump_attack
from src.attacks.one_pixel import one_pixel_attack
from src.attacks.pgd import pgd_attack
from src.models.resnet import resnet18_cifar10
from src.utils.data import CIFAR10_CLASSES

# --- Page Config ---
st.set_page_config(
    page_title="🛡️ Adversarial Robustness Demo",
    page_icon="🛡️",
    layout="wide",
)

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


# --- Model Loading (cached) ---
@st.cache_resource
def load_model(path: str):
    """Load a model checkpoint."""
    model = resnet18_cifar10()
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE).eval()
        return model, checkpoint
    return None, None


@st.cache_resource
def load_sample_images():
    """Load a few CIFAR-10 sample images for the demo.

    The images are bundled with the app (sample_images.pt, one per class), so
    there is no 170 MB CIFAR-10 download at runtime. Downloading on the Space
    was slow/unreliable and left the app stuck on load_sample_images().
    """
    import os
    path = os.path.join(os.path.dirname(__file__), "sample_images.pt")
    bundle = torch.load(path, map_location="cpu")
    imgs, labels = bundle["images"], bundle["labels"]
    samples = {}
    for i in range(len(labels)):
        label = int(labels[i])
        samples[CIFAR10_CLASSES[label]] = (imgs[i], label)
    return samples


# --- Helper Functions ---
def get_prediction(model, image_tensor):
    """Get model prediction."""
    with torch.no_grad():
        output = model(image_tensor.to(DEVICE))
        probs = F.softmax(output, dim=1)[0]
        conf, pred = probs.max(0)
    return CIFAR10_CLASSES[pred.item()], conf.item(), probs.cpu().numpy()


def tensor_to_pil(tensor, upscale=8):
    """Convert tensor to displayable PIL image."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    if upscale > 1:
        w, h = pil_img.size
        pil_img = pil_img.resize((w * upscale, h * upscale), Image.NEAREST)
    return pil_img


def create_confidence_chart(probs, title="Confidence", highlight_class=None):
    """Create a horizontal bar chart of class probabilities."""
    colors = ["#ef4444" if CIFAR10_CLASSES[i] == highlight_class else "#3b82f6" for i in range(10)]
    fig = go.Figure(go.Bar(
        x=probs,
        y=CIFAR10_CLASSES,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Probability",
        yaxis=dict(autorange="reversed"),
        height=350,
        margin=dict(l=80, r=20, t=40, b=40),
        xaxis=dict(range=[0, 1]),
    )
    return fig


# --- Main App ---
def main():
    st.title("🛡️ Adversarial Robustness Benchmark")
    st.markdown(
        "Compare how **standard** and **adversarially trained** neural networks "
        "respond to adversarial attacks. Upload an image or pick a sample, "
        "choose an attack, and see the difference in real-time."
    )

    # Load models
    standard_model, std_ckpt = load_model(os.path.join(CHECKPOINT_DIR, "standard_best.pt"))
    robust_model, rob_ckpt = load_model(os.path.join(CHECKPOINT_DIR, "robust_best.pt"))

    if standard_model is None and robust_model is None:
        st.error(
            "⚠️ No model checkpoints found! Train models first:\n\n"
            "```bash\npython scripts/train.py --mode both --epochs 50\n```"
        )
        return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("Model Info")
        if std_ckpt:
            st.metric("Standard Model Acc", f"{std_ckpt.get('test_accuracy', 'N/A'):.1f}%")
        if rob_ckpt:
            st.metric("Robust Model Clean Acc", f"{rob_ckpt.get('test_clean_accuracy', 'N/A'):.1f}%")
            st.metric("Robust Model PGD Acc", f"{rob_ckpt.get('test_pgd_accuracy', 'N/A'):.1f}%")

        st.divider()

        st.subheader("Attack Parameters")
        attack_type = st.selectbox(
            "Attack Method",
            ["FGSM", "PGD", "DeepFool", "Carlini-Wagner", "HopSkipJump", "One Pixel"],
        )

        if attack_type in ("FGSM", "PGD"):
            epsilon = st.slider(
                "Epsilon (ε)",
                min_value=0.0,
                max_value=0.1,
                value=0.031373,
                step=0.002,
                format="%.3f",
                help="Maximum perturbation magnitude. 8/255 ≈ 0.031 is standard.",
            )
            st.caption(f"≈ {epsilon * 255:.1f}/255 pixel values")

        if attack_type == "PGD":
            pgd_steps = st.slider("PGD Steps", 1, 50, 20)
            pgd_alpha = st.slider(
                "Step Size (α)",
                min_value=0.001,
                max_value=0.02,
                value=0.007843,
                step=0.001,
                format="%.3f",
            )

        if attack_type == "Carlini-Wagner":
            cw_steps = st.slider("Binary-search steps", 1, 9, 4)
            cw_iters = st.slider("Adam iterations", 20, 300, 100, step=20)
            st.caption("Minimum-$L_2$ white-box attack. ~2–4 s per image.")

        if attack_type == "HopSkipJump":
            hsj_budget = st.slider("Query budget", 200, 3000, 1000, step=100)
            hsj_iters = st.slider("Iterations", 5, 30, 15)
            st.caption("Decision-based black-box (label-only). ~5–10 s per image.")

        if attack_type == "One Pixel":
            op_k = st.slider("Pixels (k)", 1, 5, 1)
            op_pop = st.slider("DE population", 50, 400, 150, step=50)
            op_iters = st.slider("DE generations", 10, 75, 20, step=5)
            st.caption("$L_0$ attack via differential evolution. Slow on free CPU (~15–25 s).")

        st.divider()
        st.caption(f"Device: `{DEVICE}`")
        st.caption("Architecture: ResNet-18 (CIFAR-10)")

    # --- Image Input ---
    st.subheader("📷 Input Image")

    input_method = st.radio(
        "Choose input method",
        ["Sample CIFAR-10 Images", "Upload Your Own"],
        horizontal=True,
    )

    image_tensor = None

    if input_method == "Sample CIFAR-10 Images":
        samples = load_sample_images()
        selected_class = st.selectbox("Pick a class", list(samples.keys()))
        image_tensor, true_label = samples[selected_class]
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dim
        st.image(tensor_to_pil(image_tensor), caption=f"True label: {selected_class}", width=256)
    else:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            image_tensor = transform(img).unsqueeze(0)
            st.image(tensor_to_pil(image_tensor), caption="Uploaded image (resized to 32x32)", width=256)

    if image_tensor is None:
        st.info("👆 Select a sample image or upload your own to get started.")
        return

    # --- Run Attack ---
    if st.button("🚀 Run Attack", type="primary", use_container_width=True):
        image_tensor = image_tensor.to(DEVICE)

        # Get clean predictions
        if standard_model:
            std_clean_class, std_clean_conf, std_clean_probs = get_prediction(standard_model, image_tensor)
        if robust_model:
            rob_clean_class, rob_clean_conf, rob_clean_probs = get_prediction(robust_model, image_tensor)

        # Get the predicted label for the attack
        with torch.no_grad():
            target_model = standard_model or robust_model
            pred_label = target_model(image_tensor.to(DEVICE)).argmax(dim=1)

        # Generate adversarial example
        with st.spinner(f"Generating {attack_type} adversarial example..."):
            if attack_type == "FGSM":
                adv_image = fgsm_attack(target_model, image_tensor, pred_label, epsilon, DEVICE)
            elif attack_type == "PGD":
                adv_image = pgd_attack(
                    target_model, image_tensor, pred_label,
                    epsilon=epsilon, alpha=pgd_alpha, steps=pgd_steps,
                    random_start=True, device=DEVICE,
                )
            elif attack_type == "DeepFool":
                adv_image, norms = deepfool_attack(target_model, image_tensor, device=DEVICE)
            elif attack_type == "Carlini-Wagner":
                adv_image, _ = cw_l2_attack(
                    target_model, image_tensor, pred_label,
                    binary_search_steps=cw_steps, max_iterations=cw_iters,
                    device=DEVICE,
                )
            elif attack_type == "HopSkipJump":
                adv_image, _ = hopskipjump_attack(
                    target_model, image_tensor, pred_label,
                    max_queries=hsj_budget, num_iterations=hsj_iters,
                    device=DEVICE,
                )
            elif attack_type == "One Pixel":
                adv_image, _ = one_pixel_attack(
                    target_model, image_tensor, pred_label,
                    k=op_k, pop_size=op_pop, max_iter=op_iters,
                    device=DEVICE,
                )

        # Compute perturbation
        perturbation = adv_image - image_tensor.to(DEVICE)
        l2_norm = perturbation.flatten().norm(2).item()
        linf_norm = perturbation.abs().max().item()

        # Amplified perturbation visualization
        pert_vis = perturbation.abs()
        if pert_vis.max() > 0:
            pert_vis = pert_vis / pert_vis.max()

        # --- Display Results ---
        st.divider()
        st.subheader("📊 Results")

        # Image comparison
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original Image**")
            st.image(tensor_to_pil(image_tensor.cpu()), width=256)
        with col2:
            st.markdown("**Adversarial Image**")
            st.image(tensor_to_pil(adv_image.cpu()), width=256)
        with col3:
            st.markdown("**Perturbation (amplified)**")
            st.image(tensor_to_pil(pert_vis.cpu()), width=256)

        # Perturbation metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Attack", attack_type)
        m2.metric("L₂ Norm", f"{l2_norm:.4f}")
        m3.metric("L∞ Norm", f"{linf_norm:.4f}")

        st.divider()

        # Model comparison
        if standard_model:
            std_adv_class, std_adv_conf, std_adv_probs = get_prediction(standard_model, adv_image)

        if robust_model:
            rob_adv_class, rob_adv_conf, rob_adv_probs = get_prediction(robust_model, adv_image)

        col_std, col_rob = st.columns(2)

        with col_std:
            if standard_model:
                st.markdown("### 🔴 Standard Model")
                st.markdown(f"**Clean:** {std_clean_class} ({std_clean_conf:.1%})")

                fooled = std_adv_class != std_clean_class
                status = "❌ FOOLED" if fooled else "✅ Correct"
                st.markdown(f"**Attacked:** {std_adv_class} ({std_adv_conf:.1%}) {status}")

                st.plotly_chart(
                    create_confidence_chart(std_adv_probs, "Standard Model (Attacked)", std_adv_class),
                    use_container_width=True,
                )

        with col_rob:
            if robust_model:
                st.markdown("### 🟢 Robust Model")
                st.markdown(f"**Clean:** {rob_clean_class} ({rob_clean_conf:.1%})")

                fooled = rob_adv_class != rob_clean_class
                status = "❌ FOOLED" if fooled else "✅ Correct"
                st.markdown(f"**Attacked:** {rob_adv_class} ({rob_adv_conf:.1%}) {status}")

                st.plotly_chart(
                    create_confidence_chart(rob_adv_probs, "Robust Model (Attacked)", rob_adv_class),
                    use_container_width=True,
                )
            else:
                st.info("Robust model not available. Train with:\n`python scripts/train.py --mode adversarial`")

    # --- About Section ---
    with st.expander("ℹ️ About This Demo"):
        st.markdown("""
        **What are adversarial attacks?**

        Adversarial attacks add small, carefully crafted perturbations to images that are
        imperceptible to humans but cause neural networks to make confident wrong predictions.

        **Attacks implemented (all from scratch):**
        - **FGSM** (Fast Gradient Sign Method): single-step $L_\infty$ attack. Fast but less powerful.
        - **PGD** (Projected Gradient Descent): multi-step iterative $L_\infty$ attack. The strongest first-order attack.
        - **DeepFool**: finds a small $L_2$ perturbation by linearizing the boundary.
        - **Carlini-Wagner**: optimization-based $L_2$ attack; the strongest minimum-perturbation attack.
        - **HopSkipJump**: decision-based black-box attack — uses only the predicted label, no gradients.
        - **One Pixel**: $L_0$ attack via differential evolution — changes only a handful of pixels.

        These span white-box, decision-based black-box, and the $L_\infty$/$L_2$/$L_0$ threat models.

        **Defense:**
        The robust model is trained using **PGD Adversarial Training** (Madry et al., 2018),
        which generates adversarial examples during training and teaches the model to be
        robust against them.

        **Research:**
        - Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (2014)
        - Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (2018)
        - Moosavi-Dezfooli et al., *DeepFool* (2016)
        - Carlini & Wagner, *Towards Evaluating the Robustness of Neural Networks* (2017)
        - Chen, Jordan & Wainwright, *HopSkipJumpAttack* (2020)
        - Su, Vargas & Sakurai, *One Pixel Attack* (2019)
        """)


if __name__ == "__main__":
    main()
