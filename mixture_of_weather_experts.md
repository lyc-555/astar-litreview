# Mixture of weather experts

**Code:** https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather

*Weight maps are not part of output. Can ask Claude to output for you.

***

### Scale / shift
&gamma;, &beta; are learnable parameters.  
&gamma; = Scaling factor (Default 1)  
&beta; = Bias/Shift (Default 0)  

### MLP
A normal feedforward neural network  
Takes in lead time as a condition, embedded into a vector [1,0,1] for eg.  

### Scale
Scales output of self-attention.  
Determines whether to trust attention output more, or Residue connection more.  

***

### Important components:

#### 1: Objective of mixture of experts
Must outperform not just individual experts, but also a mean across experts.  

#### 2: Gating mechanism  
Bias allows for model to correct anu mistakes that all experts share.  
If all individual weather models predict 0.5C higher for an area, b will be -0.5.
