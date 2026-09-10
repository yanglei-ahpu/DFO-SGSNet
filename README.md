<div align="center">

# DFO-SGSNet

### Deep Fourth-Order Subspace-Guided Spectrum Learning for Unknown-Cardinality and Underdetermined Wideband DOA Estimation

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Task-Wideband%20DOA-4C78A8" alt="Wideband DOA">
  <img src="https://img.shields.io/badge/Setting-Unknown--K%20%7C%20Underdetermined-6F42C1" alt="Unknown-K and Underdetermined">
</p>

**Physics-guided source enumeration · Fourth-order virtual-array expansion · Spectrum refinement**

</div>

---

DFO-SGSNet is a **data-model hybrid framework** for wideband direction-of-arrival (DOA) estimation under **unknown source cardinality, underdetermined configurations, and limited observations**. It combines fourth-order virtual-array processing with cardinality-aware subspace learning and physics-constrained spectrum refinement, retaining the interpretability of classical subspace methods while introducing data-driven adaptation where conventional estimation becomes unreliable.

## Main Contributions

**1. Fourth-order virtual-array formulation for underdetermined wideband DOA estimation**  
Strict fourth-order cumulants are mapped to an extended virtual-lag domain and coherently fused across frequency to construct a structured virtual covariance matrix. The enlarged virtual dimension provides additional spatial degrees of freedom beyond the physical array aperture, supporting source configurations near and beyond the physical-array limit.

**2. Candidate-subspace representation learning for unknown source cardinality**  
DFO-SGSNet constructs candidate MUSIC spectra over multiple source-cardinality hypotheses and jointly evaluates them using candidate-wise structured scoring and global candidate-spectrum-bank representation learning. Explicit **MDL-inspired, eigengap, and peak-consistency evidence** is incorporated into the learned scores to retain established model-order information.

**3. Physics-constrained deep spectrum refinement**  
The physical MUSIC spectrum selected by the estimated source cardinality is refined using an angle-preserving dilated residual network. A bounded logit-domain correction suppresses finite-snapshot spectral distortion while maintaining the analytical spectrum as the physical reference.

---

## Repository Layout

```text
DFO-SGSNet/
├── fourth_order.py    # Fourth-order lag processing, focusing, covariance, and MUSIC bank
├── dataset.py         # Wideband data generation and dataset interface
├── model.py           # Candidate-subspace estimator and spectrum refiner
├── loss.py            # Source-cardinality and spectrum losses
├── train_test.py      # Training, validation, and testing
└── README.md
```

---

## Environment

The following versions are recommended for reproducing the implementation.

| Package | Recommended Version | Role |
|:--|:--:|:--|
| **Python** | 3.10+ | Runtime environment |
| **PyTorch** | 2.0+ | Network training, complex-valued tensor operations, and eigendecomposition |
| **NumPy** | 1.24+ | Numerical processing and evaluation |
| **SciPy** | 1.10+ | Hungarian assignment for DOA matching |
| **Matplotlib** | 3.7+ | Spectrum and result visualization |
| **CUDA** | Optional | GPU acceleration with a compatible PyTorch build |

Install the main Python dependencies with:

```bash
pip install torch numpy scipy matplotlib
```

> For GPU training, install the PyTorch build compatible with the local CUDA environment.

---

## Training

### Full Training

Run the complete training and evaluation pipeline with:

```bash
python train_test.py --profile target --regenerate
```

The command performs data generation and the complete staged optimization procedure defined in `train_test.py`.

The three core learning stages are:

| Stage | Trainable Component | Default Learning Rate |
|:--|:--|:--:|
| **I. Source-cardinality learning** | Source-order estimator | `8e-4` |
| **II. Spectrum refinement** | Spectrum-refinement network | `5e-4` |
| **III. Joint fine-tuning** | Source-order estimator + spectrum refiner | `1e-4` |

After the dataset has been generated once, `--regenerate` can be omitted to reuse the existing training, validation, and test sets.

### Paper-Aligned Example

The main experimental parameters can also be specified explicitly.

<details>
<summary><b>Show full training command</b></summary>

```bash
python train_test.py \
  --profile target \
  --regenerate \
  --num_sensors 8 \
  --min_sources 2 \
  --max_sources 10 \
  --snapshots 512 \
  --num_subbands 8 \
  --frequency_min 700 \
  --frequency_max 1300 \
  --angle_min -60 \
  --angle_max 60 \
  --snr_min 10 \
  --snr_max 10 \
  --coherence_min 0 \
  --coherence_max 0.35 \
  --train_samples 16000 \
  --val_samples 2000 \
  --test_samples 2000 \
  --grid_size 2401
```

</details>

> Experimental parameters such as the number of sensors, source-cardinality range, snapshots, frequency range, SNR, coherence level, dataset size, and angular grid can be changed directly through command-line arguments.

---

## Citation

This repository accompanies the paper:

> **DFO-SGSNet: Deep Fourth-Order Subspace-Guided Spectrum Learning for Unknown-Cardinality and Underdetermined Wideband DOA Estimation**

Citation information will be updated after publication.

---

## Contact

For questions regarding the implementation or experimental reproduction, please open a GitHub issue.
