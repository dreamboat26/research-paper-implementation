## SD3 - Scaling Rectified Flow Transformers for Image

### Overview
The SD3 paper introduces **Rectified Flow Transformers (RFTs)** for image generation tasks. This approach optimizes the flow-based generative model by incorporating **rectified transformations** in the flow process to handle high-dimensional image data efficiently.

### Key Contributions:
1. **Rectified Flows**: 
   - Rectified flows are introduced to improve the efficiency and expressiveness of normalizing flows, especially when working with high-dimensional data like images.
   - These flows use **non-linear and monotonic transformations** to improve both model expressiveness and training stability.
   - The rectification process stabilizes training and enhances the model’s ability to capture complex dependencies in data.

2. **Flow-based Generative Model**:
   - Flow-based models map a simple distribution (like Gaussian) to a complex data distribution (e.g., image).
   - The SD3 approach enhances these models with more efficient transformations that can process large, high-dimensional images.

### Architecture:
- **Rectified Transformer Layers**: These layers replace standard transformer blocks with rectified transformations to improve performance in high-dimensional spaces.
- **Normalizing Flows**: Apply rectified transformations on latent space representations instead of pixel space to reduce computational complexity while maintaining high-quality image generation.

### Notes
Rectified Flows are a modification of the traditional Normalizing Flows technique, designed to improve the efficiency and expressiveness of generative models, especially when handling high-dimensional data like images. To understand rectified flows, it's important to first understand normalizing flows.
Normalizing Flows: A Brief Overview

Normalizing Flows (NF) are a class of generative models that learn to map a simple distribution (e.g., Gaussian) to a complex data distribution (e.g., the distribution of images). The idea is to use a series of invertible transformations (functions) to transform a simple random variable into a complex data distribution. The key property of flows is that they are invertible, which means that you can both sample from the model and compute the likelihood efficiently.
The Problem with Traditional Normalizing Flows

The main challenge with normalizing flows lies in how transformations are applied to high-dimensional data. Applying a series of transformations to data such as images often involves complex computations, leading to high computational cost and making it difficult to handle large images or datasets efficiently.
Rectified Flows

Rectified Flows improve upon traditional normalizing flows by introducing a rectification mechanism that helps handle complex data distributions more effectively. Instead of using traditional transformations, rectified flows use a transformation that rectifies the output in a way that better captures spatial dependencies and non-linearity in the data, leading to improved modeling of complex distributions.

The rectification process refers to adjusting the transformation applied to the data during the flow process to avoid numerical issues and improve the ability of the model to learn complex distributions. This rectification can be achieved through various mechanisms, such as:

- Non-linear transformations: These help capture complex dependencies in the data, which are often challenging to model with simple affine transformations.
- Monotonic transformations: These allow for smoother and more stable training, particularly in high-dimensional spaces, which helps improve the quality of generated samples.
- Better stability in training: Rectified transformations can help stabilize the training process by reducing issues such as mode collapse or vanishing gradients.

