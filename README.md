# Uncertainty

## 🚀 Installation

### 1. Installer uv (gestionnaire de dépendances)

Pour gérer les dépendances, ce projet utilise [uv](https://docs.astral.sh/uv), un gestionnaire de paquets Python bien plus rapide que pip qui gère automatiquement les environnements virtuels.

#### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 🪟 Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Cloner le Projet

```bash
git clone https://github.com/fab-toc/uncertainty.git
cd uncertainty
```

### 3. Installer les dépendances

Le choix des dépendances PyTorch dépend de votre configuration matérielle :

#### 🔥 Avec GPU NVIDIA

Identifiez d'abord votre version CUDA :

```bash
nvidia-smi
```

Puis installez selon votre version :

```bash
# CUDA 12.8
uv sync --extra cu128

# CUDA 13.0
uv sync --extra cu130
```

#### 💻 Autres Configurations

```bash
# CPU uniquement
uv sync --extra cpu
```
