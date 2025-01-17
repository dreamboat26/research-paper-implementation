# Paper Summary

## 2. **Bayesian Flow Networks**

Bayesian Flow Networks (BFNs) extend normalizing flows by incorporating a **Bayesian framework** to model uncertainty in the parameters of the flow. This enhances robustness in generative modeling by treating parameters as distributions instead of fixed values.

### Key Contributions:
- **Bayesian Treatment of Flow Parameters**: Instead of treating the parameters of the flow model as fixed, the model treats them as probabilistic variables with a prior and learns a posterior distribution over these parameters.
- **Normalizing Flows**: The paper builds on **normalizing flows**, a method for learning complex distributions through invertible transformations.
- **Variational Inference**: The model uses variational inference to approximate the posterior distribution of the flow parameters.

### Architecture:
- **Flow Architecture**: The model uses invertible neural networks (e.g., RealNVP, Glow) as building blocks for the flow transformations. Each transformation layer applies a non-linear mapping to the data while ensuring invertibility.
- **Uncertainty Propagation**: By incorporating the posterior distributions over parameters, the model generates multiple plausible samples, each corresponding to a different set of learned parameters.

### Use Cases:
- Robust density estimation
- Uncertainty quantification in generative models
- Probabilistic modeling and generative tasks requiring uncertainty handling

---

