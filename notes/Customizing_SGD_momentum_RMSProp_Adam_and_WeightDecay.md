# Customizing SGD to Make Training Faster

The vanilla SGD optimizer update the parameters' weight with the following formula
$$ weights \mathrel{-}= learning\_rate \times weights.gradient $$

The $learning\_rate$ is a fixed number and same for all dimensions of the weights vector and $weights.gradient$ is determined solely on one point of the weight vector.

(TODO: common problems of Vanilla SGD)

We can make these 2 components to adapt to different points and steps of the training process, here's a few tricks:

## 1. Momentum
Refine $weights.gradient$ with moving averages of weights, the formula becomes:
$$ weights\_avg = \beta \times weights\_avg + (1-\beta) \times weights.gradient $$
$$ weights \mathrel{-}= learning\_rate \times weights\_avg $$

This change can effectively prevent the training to be stuck at a local minimum (that have 0 gradients). (TODO, verify)
Note that we should record the average of each dimension of the weights vector and update each dimension accordingly, 
rather than averaging all the dimensions of the weights vector (that's more like a normalization).

## 2. RMSProp to prevent vanishing and exploding step size
RMSProp improves the vanilla SGD by allowing each parameter of the weights vector get its own $learning_rate$ controlled by a global $lr$,
so that the step size of all the parameters would have roughly similar magnitude, even when gradients vary drastically across parameters and iterations.



Parameters that needs updates a lot get a bigger `lr` while the ones that are good enough gets a small one.
We shall observe consistent near-zero gradients for the former one and highly fluctuated gradients for the later case,
personally I think this is a bit counterintuitive, mostly due to the fact that a consistent close to zero gradients could also indicate this dimension has reached to global minimum, 
meaning it doesn't need to be updated further.


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

This is the genius of RMSProp: it decouples "gradient magnitude" from "step size stability"—so neither parameter dominates the update process, and both converge at a reasonable rate.
Noted the $eps$ is to prevent zero denominator, it's usually 1e-8. By convention, $alpha = 0.99$.







