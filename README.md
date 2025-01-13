# Generative Fractional Diffusion Models (GFDM)

This repository contains the implementation of **Generative Fractional Diffusion Models (GFDM)**, a novel approach to generative modeling that leverages **fractional Brownian motion (fBM)** to enhance diversity and quality in data generation. This work is based on the paper [Generative Fractional Diffusion Models](https://arxiv.org/abs/2310.17638).

---

## **Overview**

Generative Fractional Diffusion Models are designed to address the limitations of traditional diffusion models that rely on Brownian motion (BM). By using fractional Brownian motion (fBM) with its flexible and memory-aware properties, GFDM offers:

- **Higher Quality Outputs**: Realistic and sharp generated samples.
- **Increased Diversity**: Avoids mode collapse, producing varied outputs.
- **Efficient Sampling**: Requires fewer steps to generate data.

---

## **Key Features**

- **Fractional Noise**: Replaces BM with fBM, characterized by a tunable Hurst index (“H”), allowing control over the roughness or smoothness of the generated data.
- **Markov Approximation for fBM (MA-fBM)**: Approximates the complex fBM process using Ornstein–Uhlenbeck (OU) processes for computational tractability.
- **Augmented Score Matching**: Introduces an efficient loss function to train the score function for generating high-quality samples.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/dreamboat26/research-paper-implementation.git
   cd Generative-Fractional-Diffusion-Models
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **Training the Model**

Train the GFDM model on your dataset using the following command:
```bash
python train.py --dataset <dataset_name> --epochs 100 --hurst 0.7
```

- `--dataset`: Path to the dataset (e.g., MNIST, CIFAR-10).
- `--epochs`: Number of training epochs (default: 100).
- `--hurst`: Hurst index value (default: 0.7).


## **Key Components**

### **Forward Process**
Defines how data is transformed into noise using MA-fBM:
```python
# Forward SDE dynamics
def forward_process(data, hurst):
    ...
```

### **Reverse Process**
Defines the reverse-time dynamics for generating data:
```python
# Reverse SDE dynamics
def reverse_process(noise, trained_model):
    ...
```

### **Score Matching**
Implements the augmented score matching loss:
```python
# Augmented score matching loss
def augmented_score_matching_loss(predicted_score, true_score):
    ...
```

---

## **Results**

### Datasets:
- **Fashion MNIST**

### Metrics:
- **FID (Fréchet Inception Distance)**: Measures the quality of generated images.
- **VSp (Pixel-wise Diversity)**: Evaluates the variety in generated samples.

#### Sample Results:
- **Fashion MNIST**:
  - FID: **0.72** (compared to 1.44 with BM-driven models).
  - VSp: **24.18** (compared to 23.64 with BM-driven models).

---

---

## **References**

- Paper: [Generative Fractional Diffusion Models](https://arxiv.org/abs/2310.17638)
- Original Authors: Gabriel Nobis, Maximilian Springenberg, Marco Aversa, et al.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contributing**
Contributions are welcome! Please feel free to submit issues or pull requests to improve this repository.

