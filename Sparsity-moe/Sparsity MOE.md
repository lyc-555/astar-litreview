# **Spatial MOE**

**→** [https://arxiv.org/abs/2211.13491](https://arxiv.org/abs/2211.13491)  
→ [https://github.com/spcl/smoe](https://github.com/spcl/smoe)  
→ 25 citations   
—----------------—-------------—-------------—-------------—-------------—-------------—-----------  
<img src="image1.png" width="50%">

<br>

<img src="image3.jpeg" width="50%">

<br>

| **General Feature 1** | General Aim → To predict Spatial problems, such as weather forecasting. Input a grid, output a grid of exact same size.   |
  <img src="image2.png" width="50%"> 
  
| **General Feature 2** | An **Expert is just a CNN 3x3 Conv slider**, with different weights in slide. 
Each layer slides **all n Experts across entire grid** (Dense Experts). At each pixel, top K experts chosen ***If K\>1, each Expert maps to 1 channel.***“The outputs of each expert at a point are **concatenated**  to form the output channel dimension (of size E·F )”  → Activation Functions → Output  |
| **General Feature 3** | 16GB V100 GPUs, 30k GPU hours total |
| **General Feature 4** | “We use a batch size of 64, the Adam optimizer with an initial learning rate of of 0.001, which we divide by a factor of 10 if the validation loss has not improved for two epochs to a minimum learning rate of 1e-6, and early stopping if the validation loss does not improve for five epochs” |
|  |  |
| **Gating Mechanism 1** | Gate is **not updated by Backprop** from MSE, instead from **RC Loss** After main backward pass, each Gate looks at loss reaching each selected Expert. If loss exceeds a **quantile q compared to other gates**of loss  , output of selected Expert is deemed **incorrect**. Target is set to **0**. All other **unselected** Experts have Target set to N/E. 1Target\>0 *N \= Number of incorrectly-chosen Experts E \= Number of unselected Experts*  Weight for each Expert is updated by **BCE** loss, of value to Target. This is done **independently** for each pixel of grid map.  |
| **Gating Mechanism 2** | **k sparse probability vector** → Only k Experts are activated. `KeepTopK` k=3 → *(0, .5, .4, .1, 0, 0, 0, 0\)* → All other 5 experts have weight \= 0   **What this means**: For each *downstream task (i.e. Predicting a certain property)*, only k Experts will ever be used. The rest are completely ignored  |
| **Gating Mechanism 3** | Routing logic is **unaffected by inputs**. *“Tensor Routing”*  Useful because geographic locations are fixed. Experts do not need to look at inputs to decide, its own location suffices. |
|  |  |
| **Expert Feature 1** | Experts updated by MSE, through **Backprop**. MSE at all points of grid are summed  Note: Some points have 0 MSE as Experts were not selected there Error divided by N, total no. of grid points. This averaged error is used for Backprop **Implication**: If **Expert chosen less, gradient update is of smaller magnitude** |
| **Expert Feature 2** | **Expert Error Damping (0.1)**  If a selected Expert is judged to be **incorrect** at a point, its main loss *(Not RC Loss)* during backprop is reduced.  This **decreases weight updates** for said Expert’s **Dense Conv Weights** *(Aka updates the Expert, not the Gate weights)*, at that point, so it doesn’t force an Expert to learn something it will not be good at anyways.  Instead the **Gate will do the job of not choosing that Expert.**  |
|  |  |
| **Application 1** | → Applied SMoE to ResNet-based **weather prediction**   **Replaced certain Convolution layers with SMoE layers**. *(IN both, Grid Size is kept through `padding=1, Stride=1`)*  Number of filters in replaced filter \= No. of Experts chosen (K). This ensures channel count at each layer remains the same.  Routing between layers with same Expert count & output Channel count share the same routing weights  |
