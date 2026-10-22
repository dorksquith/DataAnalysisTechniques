---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: 'Python 3'
---

(chapter:EstimatorsIntro)=
# Estimators Intro
<!--Week7-Tools/Tools.tex-->
<!--Week8-Tools/MaxLike.tex-->
<!--Week9-Fits-Tests/LeastSquares.tex-->

<!--
Describe Point and Interval Estimation with examples
Explain the Bootstrap Method
Implement the Bootstrap Method to esimate uncertainties
Define Consistency, Bias, and Minimum Variance and explain their importance for estimators
-->
I have mentioned Estimates a couple of times, most recently in [](#sec-sem), where I stated that the sample standard deviation is a good estimate for the true standard deviation when we want to calculate the SEM.

In every day life, when we say "estimate" we mean "educated guess". In statistics it is similar, but we have a very clear framework for how to go about making the educated guess. An Estimator should be **Consistent**, **Unbiased**, and **Minimum Variance**.

Consistent
: The estimate should approach the true value in the limit of infinite sample size.

  $\hat{\theta} \rightarrow \theta\;\; as \;\; N\rightarrow \infty$

Unbiased
: The bias $b$ is the difference between the estimate and the true parameter. This should be small, preferably zero.

  $b = E[\hat{\theta}] - \theta$

Minimum Variance
: We don’t want too much spread in the estimated values (we want consistency, but we used that word already). We want **precision**.

  $V[\hat{\theta}] = E[\hat{\theta}^2] - E^2[\hat{\theta}] \equiv E[ (\hat{\theta}-\mu)^2 ]$  

:::{figure} 
:label: fig:bias-var
![](figures/Goldilocks.png)

A biased estimator is innaccurate. A large-variance estimator is imprecise.
:::

## Point & Interval Estimation

Estimating a parameter of a true distribution using data is known as Point Estimation because we are estimating a single value: a point. Providing a range of values in which we have some confidence is Interval Estimation; for example a Confidence Interval represents a range of values in which we expect some proportion of repeated experiments to fall. The proportion is the Confidence Level (CL), and can be quoted as a percentage eg $95\%$, or in terms of standard deviations eg $1\sigma$.

### The Sample Mean 

The Sample Mean $\overline{x}$ is an Estimator $\hat{\mu}$ for the True Mean $\mu$.

Consistency Check
: Does $\hat{\mu} \rightarrow \mu\; as \; N\rightarrow \infty$?

The [Law of Large Numbers](#eq:lln) tells us that **the Sample Mean is a consistent estimator for the true mean**. LLN reminder: If every possible measurement of $x$ is included in the calculation of $\overline{x}$, the probability of there being any measurable difference between $\overline{x}$ and $\mu$ is zero.

Bias Check
: What is the bias $b = E[\hat{\mu}] - \mu$?


Let's calculate the expectation of the sample mean:

```{math}
:label: eq:expmean

\begin{aligned}

E\left[\, \overline{x}\, \right]  & = E\left[ \dfrac{1}{N} \sum\limits_i^N x_i \right] \textcolor{blue}{\; because\; \overline{x}  = \frac{1}{N} \sum \limits_i^N x_i}\\ 
 
  & = \dfrac{1}{N} \sum\limits_i^N E\left[ x_i \right] \textcolor{blue}{\;because\; E[sum] = sum(E)} \\
 & = \dfrac{1}{N} \sum\limits_i^N E[X] \textcolor{blue}{\;because\; x_i\;are\; iid\; so\; E[x_i] = E[X]\; for\; all\; x_i}\\
  & = \dfrac{1}{N} N E[X]= E[X] \\                                                
\end{aligned}
```

We have shown that $E\left[\, \overline{x}\, \right] = E[X]$, which with the notation $\mu \equiv E[X]$ and for this choice of estimator $\hat{\mu} =\overline{x}$, shows that **the Sample Mean is an unbiased estimator for the True Mean**.

Variance Check
: Is the sample mean the minimum variance estimate for the true mean? Proving that an estimator is the minimum variance of all possible estimators is beyond the scope of this module, but we will look at the variance of $\overline{x}$, and I will state without proof that $\overline{x}$ is the Minimum Variance Unbiased Estimator [MVUE](https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem) for $\mu$. 

Let's calculate the variance of the sample mean:

```{math}
:label: eq:varmean

\begin{aligned}
V\left[\, \overline{x}\, \right]  & = V\left[ \dfrac{1}{N} \sum\limits_i^N x_i \right] \textcolor{blue}{\; because\; \overline{x}  = \frac{1}{N} \sum \limits_i^N x_i}\\ 
 &= \sum\limits_i^N  V\left[ \dfrac{1}{N}  x_i \right] \textcolor{blue}{\;because\; V[sum] = sum(V)\; for\; iid\; x_i}\\
  & = \dfrac{1}{N^2}  \sum\limits_i^N V\left[ x_i \right] \textcolor{blue}{\;because\;V\left[ax\right] = a^2 V\left[x\right] }\\
  & = \dfrac{1}{N^2} N V[X]  \textcolor{blue}{\;because\;V[X] = \frac{1}{N} \sum\limits_i^N (x_i-\overline{x})^2 
                          = \frac{1}{N} \sum\limits_i^N V[x_i]   }\\
  & = \dfrac{1}{N} V[X] \\                                               
\end{aligned}
```

The variance of the sample means is $\dfrac{1}{N}$ times the true variance, $V[X]$. This gives us the standard deviation of the sample means, referred to as the Standard Error on the Mean [(SEM)](#sem), because $\sigma_x = \sqrt{V[X]}$.

### The Sample Variance

The Sample Variance $v_x$ is an Estimator for the True Variance $V[x]$. $\hat{V}[x] = v_x$.

To check the Consistency, Bias, and Variance of the Estimator, we need to write down $\hat{V}[x] = v_x$ in terms of the True Variance, $V[X]$.

This is a long proof, my version of which is [here](#fig:lilyproof_varbias).  The conclusion is:

```{math}
:label: eq:vartruesamp
\hat{V}[X] = v_x = \dfrac{N-1}{N} V[X]
```

Consistency Check
: Does $\hat{V}[X] \rightarrow V[X]\; as \; N\rightarrow \infty$?

In the **Asymptotic Limit** ($N\to\infty$), $ \dfrac{N-1}{N} \to 1$. **The Sample Variance is a consistent estimator for the true variance**.



Bias Check
: What is the bias $b = E[\hat{V}[X]] - V[X]$?

```{math}
:label: eq:varbias
\begin{aligned}
b &= E[\hat{V}[X]] - V[X]\\
  &= \hat{V}[X] - V[X]\textcolor{blue}{\; because\; E[number] = number}\\
  &= \dfrac{N-1}{N} V[X] - V[X] \\
\end{aligned}
```
The Bias $b$ is non-zero, so **the Sample Variance is a biased estimator for the true variance**.

This is why we have the factor $\dfrac{1}{N-1}$ as opposed to $\dfrac{1}{N}$ in the [definition](#eq:varianceU) for the **Unbiased Variance**.

:::{note}
Bias is not the only thing that matters. Sometimes, the Best Estimator is not the zero bias estimator.
:::

If we continue simplifying [](#eq:varbias), we see that $b = \dfrac{-1}{N} V[X]$: there is a negative bias. This means that our Estimator $\hat{V}[X]$ is smaller than the true variance $V[X]$. This makes sense, because in our Estimator we have used a finite number of measurements, which will be clustered around the sample mean $\overline{x}$. The True Variance uses all possible measurements, which must be clustered less tightly about the mean. So it makes sense that the True Variance is larger than the sample variance. The larger we make $N$, the closer the Estimator gets to the True Variance.



## Interval Estimation

A **Confidence Interval** is an estimated range of values such that we expect CL% of repeated experiments to yield a value within that range. CL is the Confidence Level, for example 95% CL means that in 95% of repeated experiments, we expect the value to fall in the Confidence Interval. The Confidence Interval is a Frequentist construct. Like all frequentist statistics, it tells us nothing about a single experiment.

A **Credible Interval** is, in my opinion, a much more intuitive interval. It represents the most probable range of outcomes for a given single experiment. The Credible Interval is a Bayesian construct. It uses Prior Knowledge to allow us to estimate the range of probable outcomes for a single experiments.

:::{note}
People such as myself (particle physicists or similarly dorky types) might get home from work to find the oven on, and declare "I am 3 sigma confident that I did not leave the oven on". Because Confidence Levels are frequentist, this should be interpreted as "In 741 days, with conditions identical to today, where I came home to find the oven on, only 1 of these occasions would be my fault." Of course this is not actually what I mean when I say that. I mean I am 99.73% sure I did not leave the oven on this morning. So I should really say "I did not leave the oven on- 3 sigma credible interval". 
:::

We have seen [](#critical-region-and-significance-level) that the size of the confidence interval is set using a predefined significance level $\alpha$, for example $\alpha=5\%$ gives us a 95% CL, and the confidence interval represents the central 95% of a normal distribution centered on the estimated mean $\hat{\mu} = \overline{x}$. The 95% CL can also be expressed as $\pm 1.64\sigma$, which means the same thing. Note that the Confidence Interval is predefined - it does not rely on the measured p value and Z value from the data.

We have also seen that the uncertainty on the estimated mean reduces with the number of measurements used [](#sem).

Sometimes, we will find that we have an estimate for the mean and cannot use the SEM for the uncertainty, because our data is not normally distributed. 

Given that measurements without quoted uncertainties are meaningless, we must estimate the uncertainty. The [**Bootstrap Method**](#chapter:Resampling) is an excellent way of doing this.








