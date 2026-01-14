Target:

Our target is to train a general and unconditional diffusion model in tabular data.

Given a noise in variable X with shape [B, N, C], N represent rows, C represent Columns/Features. 
In our setting , we set B, batch size as 1.
The the diffusion model output [B, N, C] will be a denoised version of the input X. 
In our case, the model will learn to reverse the diffusion process and recover the original data distribution from the noisy input.


We use the Casual Structure Model (refer to TabICL_Data_Generation_Overview.md)to randomly generate tabular dataset as the training data for our diffusion model.

There are several problems to solve.

1. how to design the model (I prefer to follow the TabICL base model, refer to tabICL_Architecture_Overview.md )

2. how to handle scalable and variable N and C.




--- 
there are several problem.

1. If I follow TabICL change stage 3 to a diffusion head/ the input is [B, N, 4] (4 CLS token), how to decode to [B, N, C] ?