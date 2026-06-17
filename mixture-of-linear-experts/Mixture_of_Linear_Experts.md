# **Mixture of Linear Experts**

  

→ <https://arxiv.org/abs/2312.06786>

→ <https://github.com/RogerNi/MoLE>

→ 86 citations 

  
<img src="image1.jpg" width="70%">
  
  

**Aim & Structure** 

  

\> Paper nomenclature: Each ***‘Head’*** means Expert. 

  

**Aim**: To create fidelity for **Linear Models** in **long-term time series forecasting**, using MOE. Weather, Traffic, Electricity, etc.

  

**Forward pass:**

1.  \[Batch, seq\_len, Channel\] as input X  
2.  \[Batch, seq\_len, features\] as calendar features (x\_mark), on a *separate, parallel path*.

  

\> seq\_len = 1 time-step

\> Batch = Batch-size by DataLoader. Without this, input would be \[seq\_len, Channel\]. There is only one datapoint (EG. Speed) at each seq\_len. We split up input by every Batch. 

\>\> *EG. We take* *Channel**=1,* *\[(1,2,3,4,5,6,7,8)\]* *→* *\[(1,2,3,4), (5,6,7,8)\]* *for* *Batch**=4*

\> Channel = Different variables, that are all forecasted together. 

*\> \> EG. weather dataset's 21 meteorological variables - Rainfall/Temp/etc.*

  

\> Calender feature/x\_mark = Features for each set of \[Batch, seq\_len\] , describing temporal conditions of every point of data. 

\>\>  EG. *“Weekend/Weekday”, “Holiday”*, etc.

\>\> “...various temporal components of a datetime value are encoded into uniformly spaced values between \[−0.5,0.5\]”

  
  

3. Converts input into moving average. *(****DLinear*** *only)* 
      
    1.  Moving average becomes → **Trend** (Xtrend)
    2.  Original - Moving Average → **Seasonal**  (XSeasonal)

  

*\> This decomposition captures time-dependent anomalies, that an overall trend does not predict. So 2 processes will deal with it better.* 

*\> There is no decomposition for* ***RLinear & RMLP*** *(Elaboration at bottom).*

  

4. x\_mark is used to generate weights for each Expert, through 2 MLP layers. Reason is that timing will affect when each Expert is useful. t\_dim experts.
      
    1.  Model takes temporal characteristics **of first entry in batch**: x\_mark\[:, 0\] → Weights for all Experts generated for that 1 batch. 
    2.  Weights in different channels, for each Batch is independent.

  

*\> This means different batches, in the same channel have* ***different weights****.*

*\> This means weights in different channels are completely* ***different****.*

*\> However, Experts are shared.*

*\> Authors tried* ***Weight dropout*** *as regularisation (0.2).*  

  

5.  Y=WtrendXtrend+WseasonalXSeasonal

  

> All Experts are contained in a **single** **weight matrix**. The Experts are literally just a bunch of weights, *no bias.* 

  
  

6.  Output = Predict \[Batch, pred\_len, Channel\] Note that Channel remains, each input variable gets its own corresponding output.

  

7.  **MSE Error** used for Y vs Y

<br>
<br>

## General Feature 1

  - Input is decomposed into Linear & Seasonal in **DLinear** architecture

  - Not done so in RLinear & RMLP

  

> ***RLinear*** *is* *Y=RevI**n**denorm**(W.RevI**N**norm**(X))* 

>> *Denorm is reverse of normalisation.* 

>> *RLinear has trainable parameters & Denorm uses* ***same parameters*** *as Norm* 

  

> ***RMLP*** *incorporates an additional 2-layer MLP to the base RLinear mode*

>> *Y=RevI**n**denorm**(W(**X**norm**+MLP(**X**norm**)))* *, where* *X**norm* *=**RevI**N**norm**(X)**.*

>> *Performs better on* ***larger datasets***

<br>

## Gating Feature 1

  - All experts run during runtime. 
  - Not top-K, its **dense** → All Experts get a (Softmaxed) Weight.

  

## Gating Feature 2

  - Random expert **dropout** during training, where weight set to 0. Not used during inference. 
  - Chance for this can be set as a **hyperparameter.** 

  

## Gating Feature 3

  - Weights are channel specific. **Each batch in each channel** has weights’ sum set to 1. 

<br>  

*\> Meaning Expert* ***routing for different channels & batches are independent****, but the same* ***Experts are shared across channels and batches***  
  
## Expert Feature 1

  

  - Each Expert is linear nn.Linear(seq\_len, pred\_len) with no activation.
      
      - They are just weights, no bias

  
<br>  

## Findings 1

  - Regarding usefulness of x\_mark / Calendar Feature: It is more **useful when input sequence length** **Batch** **decreases** (Tested against random inputs for Calendar Feature)

 <img src="image2.jpg" width="70%">


\> “... as the input sequence lengthens, the linear layer has more past data to consider, and some patterns in this data can hint at upcoming changes in the series. Hence, we predict that the effect of conditioning on start time should be less beneficial.”

## Findings 2

  - Batch-size Tuning: Mid-range of batch sizes exhibits poor generalization, with low training loss but high test loss. Small or large are better. 

<img src="image3.jpg" width="70%"