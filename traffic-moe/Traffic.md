# **Traffic MOE**

<https://arxiv.org/abs/2403.02600>

<https://github.com/HyunWookL/TESTAM>

→ Cited by 76

### Output by Model

![](image1.png)

<br>

### Model Architecture Diagram

![](image2.png)

## **General Feature 1**

Tries to predict **traffic speed of a road for next 12h**, using speed of past hour.

3 Experts used, **each of a very different architecture**

## **General Feature 2**

***How is Forward Pass carried out?***

Concept of **Memory** (Meta Node Bank)
> **Memory**: **20 Learnable reference patterns** in a "latent space", representing various traffic conditions.

> Used to determine weights for Gating logic

1. Input *raw* traffic data is fed
   1. **1st path:** Goes into FCC + Experts
   2. **2nd path:** Goes into Memory ("Meta-node Bank")
2. **1st path** generates Expert Hidden Layer as output → Fed into "Expert Reformer" layer.
> This reforms Expert Hidden Layer output into same "latent space" as Memory
> **Self-attention**.
3. **2nd path** reforms input into "Input Reformer" layer.

> **Cross-attention** between input & Memory.

4. Self-attention output compared to Cross-attention output. Weight for each Expert determined by the **Cosine Similarity**. *(This may be what they meant by "Pairwise Distance?")*
5. Top-1 Expert selected → FF layers → Output speed predictions

## **General Feature 3**

RTX 3090GPU

<br>

## **Gating Feature 1**

`KeepTopK `= 1

Only one Expert chosen from 3.

## **Gating Feature 2**

Uses **Pseudo-Labels**. Lets use an example:

**Gate** Updating logic: *(No backward pass)*

1. Experts predict [50, 55, 80], truth = 60
   1. *This prediction is the output of FCN, after each Expert.*
2. Gate predicts [**0.6**, **0.3**, 0.1]
   1. *Selects Experts by comparing an **attention summary** of Expert's **hidden layer** output to its own **memory** of scenarios*, *where hidden layer output = **output before FCN**.*
   2. *Called **"Pairwise Distance"***
3. Errors = [10, 5, 20] → `gate[2]` is the worst choice.
4. `best_choice` = [0, 1, 0] *Pseudolabel*
5. `worst_avoidance` = [1,0,0] *Pseudolabel*
   1. *If Gate picked the worst (0.1) instead,* `worst_avoidance` *is spread across all non-worst Experts*
6. `best_choice` = −0.5 · log(**0.3**) → pushing `gate[1]` up.
7. `worst_avoidance` = −0.5 · log(**0.6**) → pushing `gate[0]` up, implicitly away from worst.

**Result**: Gate makes worst Expert less & less likely to be chosen, as **weight assigned to non-worst Experts keep increasing.**


## **Gating Feature 2b**

Above was oversimplification: There is no dedicated Gating Layer. **Gating is done in Memory**.

- Pseudolabels used to update "Input Reformer", "Expert Reformer" & Memories themselves
- Backprop gradient directly updates Memories too → Unrelated to Gating logic, as 1 Expert uses Memories to help with its own output

<br>

## **Expert Feature 1**

**Expert** Updating logic:

1. Compare Expert output against Ground truth
> Likely, this is done for all 3 Experts, NOT just the chosen one
2. Apply **backward pass**
3. Weights inside Experts are updated



## **Expert Feature 2**

**Warm-up** logic

- For first `warmup_epoch` epochs, **only Experts are updated**.
- **Gates' routing logic is not updated**. This allows for Experts to learn first, preventing instant sidelining of Experts.
