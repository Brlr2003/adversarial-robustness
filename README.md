# 🛡️ Adversarial Robustness Benchmark

An end-to-end ML pipeline that trains, attacks, defends, and deploys deep neural networks to demonstrate adversarial robustness. Users can interact with a live demo to upload images, apply adversarial attacks in real-time, and compare standard vs. robust model predictions.

**Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/YOUR_USERNAME/adversarial-robustness-demo)

## 🎯 What This Project Demonstrates

- **Model Training**: Train ResNet-18 on CIFAR-10 from scratch using PyTorch
- **Adversarial Attacks**: Implement FGSM, PGD, DeepFool, HopSkipJump (decision-based black-box), and One Pixel ($L_0$) attacks
- **Adversarial Training**: Train a robust model using PGD-based adversarial training
- **Model Evaluation**: Compare accuracy, robustness, and confidence under attack
- **Experiment Tracking**: Log all experiments with MLflow
- **Deployment**: Serve models via FastAPI + Streamlit on HuggingFace Spaces
- **CI/CD**: Automated testing and linting with GitHub Actions

## 📁 Project Structure

```
adversarial-robustness-demo/
├── src/
│   ├── models/          # Model architectures
│   │   ├── __init__.py
│   │   └── resnet.py    # ResNet-18 for CIFAR-10
│   ├── attacks/         # Adversarial attack implementations
│   │   ├── __init__.py
│   │   ├── fgsm.py      # Fast Gradient Sign Method
│   │   ├── pgd.py       # Projected Gradient Descent
│   │   ├── deepfool.py  # DeepFool attack
│   │   ├── hopskipjump.py  # HopSkipJump (decision-based black-box)
│   │   └── one_pixel.py    # One Pixel attack (L_0, differential evolution)
│   ├── training/        # Training loops
│   │   ├── __init__.py
│   │   ├── standard.py  # Standard training
│   │   └── adversarial.py # Adversarial training (PGD-AT)
│   ├── api/             # FastAPI backend
│   │   ├── __init__.py
│   │   └── main.py
│   ├── utils/           # Utilities
│   │   ├── __init__.py
│   │   ├── data.py      # Data loading & transforms
│   │   └── metrics.py   # Evaluation metrics
├── frontend/
│   └── app.py           # Streamlit frontend
├── notebooks/
│   └── exploration.ipynb # Training & analysis notebook
├── configs/
│   └── config.yaml      # Hyperparameters
├── scripts/
│   ├── train.py             # Training entry point
│   ├── evaluate.py          # White-box evaluation entry point
│   └── evaluate_blackbox.py # Black-box (HSJ, One Pixel) evaluation
├── tests/
│   └── test_attacks.py  # Unit tests
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI
├── Dockerfile           # For HuggingFace Spaces
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🚀 Quick Start

### 1. Setup
```bash
git clone https://github.com/YOUR_USERNAME/adversarial-robustness-demo.git
cd adversarial-robustness-demo
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Train Models
```bash
# Train standard model
python scripts/train.py --mode standard --epochs 50

# Train adversarial model (PGD-AT)
python scripts/train.py --mode adversarial --epochs 50

# View experiments in MLflow
mlflow ui
```

### 3. Evaluate
```bash
python scripts/evaluate.py --model-path checkpoints/standard_best.pt --attacks fgsm pgd deepfool
python scripts/evaluate.py --model-path checkpoints/robust_best.pt --attacks fgsm pgd deepfool
```

### 4. Run Demo Locally
```bash
# Option A: Streamlit only (simple)
streamlit run frontend/app.py

# Option B: FastAPI + Streamlit (production-like)
uvicorn src.api.main:app --port 8000 &
streamlit run frontend/app.py
```

## 📊 Results

| Model    | Clean Acc | FGSM (ε=8/255) | PGD-20 (ε=8/255) | DeepFool |
|----------|-----------|-----------------|-------------------|----------|
| Standard | ~93%      | ~25%            | ~0.5%             | ~15%     |
| Robust   | ~84%      | ~55%            | ~48%              | ~52%     |

## 🛠️ Tech Stack

- **ML Framework**: PyTorch
- **Experiment Tracking**: MLflow
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Deployment**: HuggingFace Spaces / Docker
- **CI/CD**: GitHub Actions

## 📖 Research Context

This project is inspired by adversarial robustness research, particularly:
- Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (FGSM)
- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD)
- Moosavi-Dezfooli et al., "DeepFool: A Simple and Accurate Fooling Method" (DeepFool)

## 📄 License

MIT License
