# Customizing SGD to Make Training Faster

To recap, the vanilla SGD optimizer update the parameters' weight with the following formula
$$ weights \mathrel{-}= learning\_rate \times weights.gradient $$

Here, the $learning\_rate$ is a fixed number and same for all dimensions of the weights vector and $weights.gradient$ is determined solely on one point of the weight vector.

Vanilla SGD is vulnerable w.r.t gradient fluctuations. An exploding gradient(e.g. weight.gradient > $10^2$) would let the parameter make a huge change that causes divergence. On the contrary, a near-zero gradient would not make effective changes to the parameter. Both cases slow down the training.

(TODO) problem with the training data, the parameters, the architecture of the DNN, and the loss landscape


We can make these 2 components to adapt to different points and steps of the training process, here's a few tricks:

## 1. Momentum
Refine $weights.gradient$ with moving averages of weights, the formula becomes:
$$ weights\_avg = \beta \times weights\_avg + (1-\beta) \times weights.gradient $$
$$ weights \mathrel{-}= learning\_rate \times weights\_avg $$

This change can effectively prevent the training to be stuck at a local minimum (that have 0 gradients). (TODO, verify)
Note that we should record the average of each dimension of the weights vector and update each dimension accordingly, 
rather than averaging all the dimensions of the weights vector (that's more like a normalization).

## 2. RMSProp to prevent vanishing and exploding step size
### Rationale
RMSProp improves the vanilla SGD by allowing each parameter of the weights vector to update with step size that is stabilized by the gradient volatility.

Each parameter get its own $learning_rate$ controlled by a global $lr$, even when gradients vary drastically across parameters and iterations.
- Gradients of the same parameter fluctuate drastically due to "seeing" different mini-batch data in different iterations
- Gradients of different parameters changes differently due to their position in the deep neural network, eg. earlier layers changes less than later layers


## The Math
The core formula of RMSProp is to assign smaller learning rates for parameters with very large gradients^2 by dividing the global learning rate by the gradients^2, 
and vice versa for parameters with small gradients^2. To make alignment across iterations, RMSProp introduce 
the moving weighted average of the square of the gradients at each step to calibrate multiple iterations of signals:
$$ weights\_grad\_squared\_avg_{(t)} = \alpha \times weights\_grad\_squared\_avg_{(t-1)} + (1-\alpha) \times weights\_grad_{(t)}^2 $$
the $\alpha$ control how much historical gradient's squared average should be considered at the current step.

At step $t=1$, $ weights\_grad\_squared\_avg_{(1)} = (1-\alpha) \times weights\_grad_{(1)}^2$

And we adjust the vanilla SGD with the 
$$ weights \mathrel{-}=  \frac{learning\_rate} {\sqrt{weights\_grad\_squared\_avg + eps}} \times weights.gradient $$
Since the moving weighted average of the squared gradients sits in the denominator, it will enable:
1. if a parameter(a dimension of the weight vector) has a tiny moving average of gradients in the past few steps(indicating it was trapped at a local minimum or well optimized), we shall make its step size to be larger than the vanilla SGD. Note that since the `weights.gradient` is small, the adjusted step size won't be very huge. 
2. on the other hand, if a parameter has a big moving average of gradients in the past few steps(indicating this parameter is jumping everywhere, possibly diverging), let's decrease the step size to keep the updates stable.


## The Key to RMSProp
Does $ \frac{weights.gradient} {\sqrt{weights\_grad\_squared\_avg + eps}}$ equal to 1? If so, then each parameter would only step the global learning rate by each iteration? then what's the point of feeding gradients? 

For the 1st question, let's revisit the 
$$ weights\_grad\_squared\_avg_{(t)} = \alpha \times weights\_grad\_squared\_avg_{(t-1)} + (1-\alpha) \times weights\_grad_{(t)}^2 $$

If $weights\_grad_{(t)}$ is roughly the same for the past few iterations,
then $weights\_grad\_squared\_avg_{(t-1)} = weights\_grad_{(t)}^2$, if $\alpha=0.5$ then $\sqrt{weights\_grad\_squared\_avg_{(t)}}$ also close to $weights\_grad_{(t)}$
finally leading to 
$$ weights \mathrel{-}=  \frac{weights.gradient} {\sqrt{weights\_grad\_squared\_avg + eps}} \times learning\_rate = +- 1 \times learning\_rate $$

Here, $+-1$ reviles one of the critical things that gradients control: the direction of the updates. (which is one point to feed gradient)

By convention $alpha = 0.99$, 




This is the genius of RMSProp: it decouples "gradient magnitude" from "step size stability"—so neither parameter dominates the update process, and both converge at a reasonable rate.
Noted the $eps$ is to prevent zero denominator, it's usually 1e-8. By convention, $alpha = 0.99$.


## Causes and solutions with gradient vanishing and exploding
| Gradient Issue            | Common Causes                                                                                                                                                    | Common Fixes                                                                                                                                                                                                                                               |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Huge Gradients**        | 1.Input data have different magnitudes<br> 2. Initialized weights to large values $e.g. N(10,1)$ causing huge activations and derivatives<br> 3. outlier batches | 1. Normalize input data/features<br>2. Use He/Xavier weight initialization<br>3. Add gradient clipping (e.g., `torch.nn.utils.clip_grad_norm_`)<br>4. Remove outlier batches or use robust loss functions (e.g., Huber loss instead of MSE)                |
| **Near-Zero Gradients**   | 1. Activation function saturation, e.g. Sigmoid and Relu<br> 2. Flat regions in the loss landscape<br> 3. Overregularized                                        | 1. Replace sigmoid/tanh with Leaky ReLU/GELU/Swish<br>2. Reduce weight decay or dropout rate<br>3. Use residual connections (ResNets) to enable gradient flow through deep layers<br>4. Increase learning rate (if gradients are small due to convergence) |






