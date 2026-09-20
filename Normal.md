# The Normal Distribution

Many RVs are "Normally Distributed", meaning they follow a Gaussian Probability Distribution. Some examples are shown in are shown in [](#fig:norm-everywhere).

::::{grid} 1 2 2 2
\label{fig:norm-everywhere}

:::{image} /figures/norm_bp.png
:::

:::{image} /figures/norm_weight.png
:::

:::{image} /figures/norm_height.png
:::

:::{image} /figures/norm_beta.png
:::

::::

The RVS plotted are blood pressures, baby birth weights, heights of English criminals in 1900, and the difference between the proton speeds measured with two different detectors. These are very different RVs, but when we plot their measured values, they all follow this same shape, with a symmetric distribution around a central value. Why?!

The reason for the apparently unrelated RVs in [](#fig:norm-everywhere) having the same underlying distribution is that **they do have something fundamental in common**: they are all the result of many interrelated factors, which makes them "sums" of different independent variables. We will see that the distribution of a sum will always tend towards a Gaussian distribution (the Central Limit Theorem).




- [] Explain why the CDF, rather than PDF, must be used for calculating probabilities for continous RVs
- [] Plot the Normal PDF and CDF 
- [] State the formula for calculating the Z value, and calculate Z values
- [] Describe the terms present in the Gaussian PDF
- [] Describe the location and scale parameters, and demonstrate the effect of changing them
- [] State the Central Limit Theorem