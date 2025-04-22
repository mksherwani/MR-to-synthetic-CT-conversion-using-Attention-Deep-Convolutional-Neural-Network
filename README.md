# MR to Synthetic CT Conversion Using Attention-Enhanced DCNN 🧠➡️🩺

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-green)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Springer Paper](https://img.shields.io/badge/Springer-Chapter%2050-<COLOR>.svg)](https://link.springer.com/chapter/10.1007/978-3-030-64583-0_50)

**Official implementation**: **"MR to Synthetic CT Conversion Using Attention-Enhanced Deep Convolutional Neural Network"**. This repository extends traditional DCNNs with **attention mechanisms** to improve MR-to-CT synthesis for medical imaging applications.

---

## 📌 Table of Contents
- [Why Synthetic CT?](#-why-synthetic-ct)
- [Key Innovations](#-key-innovations)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset Preparation](#-dataset-preparation)
- [Attention-DCNN Architecture](#-attention-dcnn-architecture)
- [Training & Inference](#-training--inference)
- [Results & Comparisons](#-results--comparisons)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🏥 Why Synthetic CT?
CT scans are essential for radiation therapy planning but involve ionizing radiation. MR imaging is safer but lacks electron density data. **This work solves this** by generating synthetic CT (sCT) scans from MR images using an **attention-augmented DCNN**, enabling accurate radiotherapy dose calculations without additional radiation exposure.

---

## 🔥 Key Innovations
- **Attention-Enhanced DCNN**: Spatial and channel attention blocks to focus on anatomically critical regions.
- **Hybrid Loss Function**: Combines SSIM, MAE, and adversarial loss for structural accuracy.
- **Clinical Validation**: sCTs validated against real CTs using dosimetric metrics (Gamma Index, DVH).

---

## ⚙️ Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x/keras
- NVIDIA GPU (CUDA 11.0+ recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/mksherwani/MR-to-synthetic-CT-conversion-using-Attention-Deep-Convolutional-Neural-Network.git
   cd MR-to-synthetic-CT-conversion-using-Attention-Deep-Convolutional-Neural-Network
Install dependencies:

bash
pip install -r requirements.txt

Preprocessing Steps
Resampling: Align MR/CT voxel spacing (e.g., 1×1×1 mm³).

Intensity Normalization:

MR: Z-score normalization per modality.

CT: Clip to [-1000, 2000] HU → scale to [0, 1].

Patch Extraction: Extract 3D patches (e.g., 128×128×32) for memory efficiency.

🧠 Attention-DCNN Architecture
Key Components
Encoder:

4 convolutional blocks with Squeeze-and-Excitation (SE) attention.

Strided convolutions for downsampling.

Bottleneck:

Cross-Modality Attention for fusing multi-contrast MR data.

Decoder:

Transposed convolutions + Spatial Attention Gates.

Skip connections with attention-guided feature fusion.

Attention-DCNN Architecture "FIGure"


✍️ Citation
If you use this code or findings in your research, please cite:

bibtex
@InProceedings{Sherwani2021,
  author    = {Sherwani, Moiz Khan and others},
  title     = {Evaluating the Impact of Training Loss on MR to Synthetic CT Conversion},
  booktitle = {Proceedings of the International Conference on Advanced Machine Learning Technologies},
  year      = {2021},
  pages     = {491--500},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-64583-0_50}
}
🤝 Contributing
Contributions are welcome! Please:

Fork the repository.

Create a feature branch (git checkout -b feature/new-attention).

Commit changes (git commit -m 'Add 3D attention blocks').

Push to the branch (git push origin feature/new-attention).

Open a Pull Request.

📜 License
This project is licensed under the MIT License. See LICENSE for details.
