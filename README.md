## **Masked Structural Growth for 2x Faster Language Model Pre-training**

### **Overview**:
The **Masked Structural Growth (MSG)** technique proposes a method for improving the efficiency of **large language model (LLM) pre-training**. By dynamically growing the model's architecture during training, this approach allows the model to reach large capacities only when necessary, resulting in faster pre-training and reduced computational costs.

### **Key Contributions**:
- **Dynamic Model Growth**: Instead of using a large model from the start, MSG begins with a smaller model and **grows the architecture dynamically** during training.
- **Faster Pre-training**: The method achieves **2x faster pre-training** compared to traditional methods by focusing on training smaller models and expanding as needed.
- **Efficient Use of Parameters**: The model grows in complexity only when required, optimizing resource usage.

### **Architecture**:
1. **Initial Smaller Model**:
   - The model starts small with fewer layers and attention heads.
   - **Masked Parameters**: Many parts of the architecture (such as layers or heads) are masked (i.e., inaccessible) initially.

2. **Masked Growth**:
   - During training, parts of the model are gradually unmasked, and new parts of the network are added based on their **importance**.
   - This expansion happens dynamically based on the model’s requirements, such as additional layers or attention heads to learn more complex features.

3. **Training with Growing Model**:
   - The model is trained progressively with a small architecture, and as the training advances, more complex parts of the model are unmasked, effectively expanding its capacity.
   - The expansion is guided by the training process, allowing the model to efficiently learn more complex patterns as needed.

### **Advantages**:
- **2x Faster Pre-training**: Pre-training time is reduced significantly compared to traditional methods.
- **Efficient Computation**: The model starts small and grows dynamically, ensuring that resources are used only where necessary.

### **Key Results**:
- The **dynamic growth** of the model leads to **faster convergence** and more efficient usage of computational resources during pre-training.

### Summary

- **Masked Structural Growth (MSG)** improves pre-training efficiency by dynamically expanding the model.


