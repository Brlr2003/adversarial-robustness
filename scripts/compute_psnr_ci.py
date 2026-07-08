"""
PSNR of the adversarial perturbations, and 95% confidence intervals for the
subset-based success rates. Both were requested by the examiners
("PSNRs", "reliability of the results missing").

PSNR (higher = less perceptible) is computed directly for FGSM and PGD on a
1,000-image test subset, and derived from the reported median/average L2 for
DeepFool and Carlini-Wagner (PSNR = 20 log10(1 / (L2 / sqrt(3072))), images in
[0,1]). Wilson 95% intervals are reported for the small-subset rates.
"""
from __future__ import annotations
import os, sys, math, json
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.resnet import resnet18_cifar10
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
D = 3 * 32 * 32


def load(p):
    m = resnet18_cifar10(); ck = torch.load(p, map_location=dev, weights_only=True)
    m.load_state_dict(ck["model_state_dict"]); return m.to(dev).eval()


def psnr_batch(clean, adv):
    mse = ((clean - adv) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).mean().item()


def psnr_from_l2(l2):
    return 20 * math.log10(1.0 / (l2 / math.sqrt(D)))


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, (c - h)) * 100, min(1, (c + h)) * 100)


def main():
    tf = transforms.ToTensor()
    test = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(test), generator=g)[:1000].tolist()
    imgs = torch.stack([test[i][0] for i in idx])
    labels = torch.tensor([test[i][1] for i in idx])

    out = {"psnr_direct": {}, "psnr_from_l2": {}, "confidence_intervals": {}}
    for tag, ck in [("standard", "checkpoints/standard_best.pt"),
                    ("robust", "checkpoints/robust_best.pt")]:
        m = load(ck)
        with torch.no_grad():
            correct = m(imgs.to(dev)).argmax(1).cpu() == labels
        ci_img, cl = imgs[correct], labels[correct]
        fadv = fgsm_attack(m, ci_img, cl, 8 / 255, dev).cpu()
        padv = pgd_attack(m, ci_img, cl, 8 / 255, 2 / 255, 20, True, dev).cpu()
        out["psnr_direct"][tag] = {
            "fgsm": round(psnr_batch(ci_img, fadv), 1),
            "pgd": round(psnr_batch(ci_img, padv), 1),
            "n": int(correct.sum()),
        }
        print(f"{tag}: FGSM PSNR {out['psnr_direct'][tag]['fgsm']} dB, "
              f"PGD PSNR {out['psnr_direct'][tag]['pgd']} dB (n={out['psnr_direct'][tag]['n']})")

    # DeepFool / C&W from reported L2 (avg L2 for DeepFool, median L2 for C&W)
    for name, (ls, lr) in {"deepfool": (0.248, 0.892), "cw": (0.104, 0.758)}.items():
        out["psnr_from_l2"][name] = {"standard": round(psnr_from_l2(ls), 1),
                                     "robust": round(psnr_from_l2(lr), 1)}
        print(f"{name}: PSNR std {psnr_from_l2(ls):.1f} dB, robust {psnr_from_l2(lr):.1f} dB")

    # Wilson 95% CIs for the small-subset One Pixel success rates
    op = {"standard": (96, [26.0, 57.3, 66.7]), "robust": (88, [10.2, 21.6, 26.1])}
    for tag, (n, rates) in op.items():
        out["confidence_intervals"][f"onepixel_{tag}_n{n}"] = []
        for r, k in zip([1, 3, 5], rates):
            succ = round(k / 100 * n)
            lo, hi = wilson(succ, n)
            out["confidence_intervals"][f"onepixel_{tag}_n{n}"].append(
                {"k": r, "rate": k, "ci95": [round(lo, 1), round(hi, 1)]})
            print(f"One Pixel {tag} k={r}: {k}% (95% CI {lo:.1f}-{hi:.1f}, n={n})")
    # full-test PGD/FGSM: n=10000 -> half-width < ~1%
    print("Full-test (n=10000) rates have 95% half-width below ~1 point.")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/psnr_ci.json", "w"), indent=2)
    print("saved results/psnr_ci.json")


if __name__ == "__main__":
    main()
