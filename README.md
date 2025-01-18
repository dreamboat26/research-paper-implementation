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

