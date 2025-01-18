## SDEDIT - Guided Image Synthesis and Editing with Stochastic Differential Equations

### Overview
SDEDIT introduces a **guided image synthesis** and **editing framework** based on **Stochastic Differential Equations (SDEs)**. The paper focuses on improving the image generation process by incorporating **conditional guidance** during the diffusion process, allowing both **generation** and **editing** of images.

### Key Contributions:
1. **Stochastic Differential Equations (SDEs)**:
   - SDEs provide a continuous-time framework for modeling the diffusion process, where noise is gradually added and removed from data.
   - The paper adapts this framework to improve the **guidance** during the reverse diffusion process, enabling more controlled and efficient image synthesis.

2. **Conditional Guidance**:
   - The model can condition the image generation process on additional input signals, such as labels, sketches, or other images, to guide the generation process.
   - This enables the model to synthesize images that are not only high-quality but also closely match the given conditions (e.g., generating images of specific classes or editing existing images based on given prompts).

### Architecture:
- **SDE-based Diffusion Process**: The image generation process is treated as a **stochastic differential equation** where the model learns the reverse process of adding and removing noise from images.
- **Guided Diffusion**: The model incorporates **conditioning information** (like text or image prompts) to guide the generation or editing process during the reverse diffusion.
- **Continuous-time Modeling**: The SDE framework allows the model to operate in continuous time rather than discrete time steps, improving image synthesis and editing flexibility.

