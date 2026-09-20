# Describing Data

## The Sample Mean $\overline{x}$ and the True Mean $E[X]$

The sample **Mean** of a RV is calculated as a normalised sum over a finite number of measurements: [](eq:mean). It is a summary statistic calculated from the data.


The **True Mean**, or **Expectation**, is a parameter of the true underlying probability distribution of our RV. 

### Discrete RV

For a Discrete RV we can calculate the Expectation in a similar way to [](eq:mean):

$$
\label{eq:expect}
E[X]  = \sum\limits_i^\infty  x_i p_i
$$

* The sum is to infinity because in Truth, there are $\infty$ possible values for X.
* The Probability of observing $x_i$ is $p_i$. Because Probabilities must sum (or integrate) to 1 ([Kolmogorov](#the-kolmogorov-axioms)), the normalisation factor $\dfrac{1}{N}$ is not needed. We are normalising every term in the sum individually, and will get the same result.

### Continuous RV

If our random variable RV is Continuous rather than discrete, then it does not make sense to "sum over all possible values", because continuous RVs have an Uncountable Infinity of possible values. Instead, we integrate:

$$
\label{eq:expectC}
E[X]  = \int\limits_{-\infty}^\infty  x_i f_X dx
$$

This looks different from the sum in [](eq:expect) because we have $f_X$ 
 instead of $p_i$. The $p_i$ are individual probabilities, but the $f_X$
 is a **Probability Density Function (PDF)**. It serves exactly the same purpose as the $p_i$ in the sum form for Discrete RVS.


:::{figure} 
:label: fig:mean-expect-mp4
:align: left
![](figures/MeanExpect.mp4)

A series of images showing a data histogram in grey and the true underlying PDF as a red dashed line. The size of the dataset is increased by a factor 10 for each step in the sequence of images.
:::

## Notation: $E[X]  \equiv \mu$

> An alternative notation for the expectation E[X] is $\mu$. They have exactly the same meaning. I'm sorry this is confusing, but I think there are strong (pedagogical) arguments for using both forms.

$$
\label{eq:expectMu}
E[X]  \equiv \mu
$$

## The Law of Large Numbers (LLN)

In [](#fig:mean-expect-mp4) we see that the mean of a data sample measuring some RV will not exactly correspond to the expectation, which is the true, theoretical mean of the RV.

The **Law of Large Numbers (LLN)**[^1] tells us that if we were to increase the size of the data sample to infinity (not possible in real life), the sample mean would approach the expected value with high probability.

[^1]: This is the Weak LLN. The Strong LLN is subtly different, explained quite nicely on [wikipedia](https://en.wikipedia.org/wiki/Law_of_large_numbers).

$$
\label{eq:lln}
\lim\limits_{N\to \infty} P(| \overline{x} - \mu | \geq \alpha) =0\;\; \mathsf{for}\;\; \alpha > 0 
$$

The sample mean is $\overline{x}$ and the true mean is $\mu \equiv E[X]$. The symbol $\alpha$  is a small number of our choosing, which we can think of as a "tolerance level" for the difference between the sample mean and the true mean.

As is often the case with Statistics, it is helpful to think about what the LLN is **not** telling us. 

>The LLN does not imply that if we add some new measurements to a dataset, the mean will get closer to the expectation. We will observe fluctuations of $\overline{x}$ away from $\mu$ for any number of measurements $N<\infty$. 

We can see a fluctuation in the image series [](#fig:mean-expect-mp4) when we compare the $N=1k$ dataset with the $N=10k$ dataset. This is examined in [](#fig:fluc).


::::{grid} 1 1 2 2
\label{fig:fluc}

:::{image} /figures/PLOTDAT2-MeanAndExpec_Norm_Mu0_Sigma1_N1000.png

For $N=1k$ measurements, we observe $|\overline{x} - \mu |=0.0044$.
:::

:::{image} /figures/PLOTDAT2-MeanAndExpec_Norm_Mu0_Sigma1_N10000.png

For $N=10k$ measurements, we observe $|\overline{x} - \mu |=0.0088$.
:::

::::

* The mean measured in the smaller 1k sample is closer to the expectation $\mu$ than the mean of the 10k sample. 
* This is a natural **Statistical Fluctuation**, and does not imply that the LLN is wrong.


## Sample Variance and Standard Deviation

The sample **Variance** [](#eq:variance) is the square of the sample **Standard Deviation** [](#eq:std). You would be forgiven for wondering why we don't just pick one of these summary statistics and ditch the other for simplicity. We keep this redundancy because they are each crucial in their own worlds, as we shall see.

The problem with the sample variance is that we can't relate it directly to the measurement, because it has units of $x^2$ rather than $x$. As such, we cannot draw the sample variance on axes with units of $x$. But we can draw the sample Standard Deviation $\sigma_x = \sqrt{V[x]}$.


:::{figure} 
:label: fig:var-std-cartoon
:align: left
![](figures/variance-cartoon)

Cartoon showing $N=5$ measurements of X, with the standard deviation indicated on the plot.
:::

It is the sample Standard Deviation that we use to quantify the spread of our data points, giving us an intrinsic **Uncertainty** on each measurement.


## True Variance and Standard Deviation

* $V[x]  = \frac{1}{N} \sum\limits_i^N  (x_i - \overline{x})^2$: the sample variance in terms of the sample mean.
* $V[x]  = \sum\limits_i^N  (x_i - \overline{x})^2 p_i$ in terms of the sample mean and the measurement probabilities $p_i$.

To write down a form for the True Variance, we can follow a few logical steps:

Sample Mean to True Mean
: Use the expectation $E[x]\equiv \mu$ in place of the sample mean $\overline{x}$

$(x_i - \overline{x})^2\;\; \rightarrow \;\;(x_i - \mu)^2$

Known probabilty and Infinite data
: Use the probability $p_i$ instead of normalising by the number of events, and let $N=\infty$

$\frac{1}{N} \sum\limits_i^N (x_i - \mu)^2 \;\; \rightarrow \sum\limits_i^\infty (x_i - \mu)^2 p_i$

Expectation definition
: Use [](#eq:expect) to note that:

$E[X^2]  = \sum\limits_i^\infty  x^2_i p_i \;\; \therefore \;\; E[(X-\mu)^2]  = \sum\limits_i^\infty  (x-\mu)^2 p_i$

The above steps allow us to write the true variance in terms of the expectation:

$$
\label{eq:truevar}
V[X] = E[(X-\mu)^2]
$$

Another useful form for the true variance in terms of the expectation is:

$$
\label{eq:truevar2}
V[X] = E[X^2] - E^2[X]
$$

:::{dropdown} Proof that [](#eq:truevar1) and [](#eq:truevar2) are equivalent
$$
\begin{aligned}
V[X] & = E[\, X^2 + E^2[X] - 2X\,E[X] \,]\\
	& = E[\, X^2\,] + E[\, E^2[X] \,]- E[\,2X\,E[X] \,]\;\; because\; E[A+B] = E[A] + E[B] \\
	& = E[ X^2] + E^2[X] - 2E[X]  E[X]\;\; because\; E[E[A]] = E[A]\\
	& = E[ X^2] + E^2[X]  - 2E^2[X]   \\	
	& = E[ X^2] - E^2[X]   \\
\end{aligned}
$$
:open:

:::

