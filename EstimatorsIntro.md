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

Mean Squared Error (MSE)
: The **MSE** combines Bias and Variance into a single measure of estimator quality:

  ```{math}
  :label: eq:mse
  \text{MSE}[\hat{\theta}] = E\!\left[(\hat{\theta} - \theta)^2\right] = V[\hat{\theta}] + b^2
  ```

  An ideal estimator has zero bias ($b=0$) and small variance. The MSE makes clear that these two goals can trade off: a slightly biased estimator may have a lower MSE than an unbiased one if its variance is much smaller. This is the **bias–variance tradeoff**.

:::{figure} 
:label: fig:bias-var
![](figures/Goldilocks.png)

A biased estimator is innaccurate. A large-variance estimator is imprecise.
:::

Estimating a parameter of a true distribution using data is known as **Point Estimation** because we are estimating a single value: a point. Providing a range of values in which we have some confidence is **Interval Estimation**; for example a Confidence Interval represents a range of values in which we expect some proportion of repeated experiments to fall. The proportion is the Confidence Level (CL), and can be quoted as a percentage eg $95\%$, or in terms of standard deviations eg $1\sigma$.

(sec:mean-est)=
## The Sample Mean as an Estimator for the True Mean

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
  & = \dfrac{1}{N^2} N V[X]  \textcolor{blue}{\;because\; x_i\; \text{are iid, so}\; V[x_i]=V[X]\; \text{for all}\; i}\\
  & = \dfrac{1}{N} V[X] \\                                               
\end{aligned}
```

The variance of the sample means is $\dfrac{1}{N}$ times the true variance, $V[X]$. This gives us the standard deviation of the sample means, referred to as the Standard Error on the Mean [(SEM)](#sem), because $\sigma_x = \sqrt{V[X]}$.


````{card}
For a single RV, we can Estimate the mean and its uncertainty directly from the data:

```{math}
:label: eq:mean-est-uncert
\begin{aligned}
\hat{\mu} \pm \hat{\sigma_{\mu}} & = \overline{x} \pm \sigma_{\overline{x}} \\ \\
                                 & = \overline{x} \pm \mathsf{SEM} \\ \\
                                 & = \overline{x} \pm \dfrac{ \sigma_{x} }{ \sqrt{N} } \\                                
\end{aligned}
```

````


(sec:var-est)=
## The Sample Variance as an Estimator for the True Variance

The Sample Variance $v_x$ is an Estimator for the True Variance $V[x]$. $\hat{V}[x] = v_x$.

To check the Consistency, Bias, and Variance of the Estimator, we need to write down $\hat{V}[x] = v_x$ in terms of the True Variance, $V[X]$.

This is a long proof, my version of which is [here](#fig:lilyproof_varbias).  The conclusion is:

```{math}
:label: eq:vartruesamp
E\!\left[\hat{V}[X]\right] = \dfrac{N-1}{N} V[X]
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
  &= \dfrac{N-1}{N} V[X] - V[X] \textcolor{blue}{\;\text{substituting}\; E\!\left[\hat{V}[X]\right] = \dfrac{N-1}{N}V[X]}\\
\end{aligned}
```
The Bias $b$ is non-zero, so **the Sample Variance is a biased estimator for the true variance**.

This is why we have the factor $\dfrac{1}{N-1}$ as opposed to $\dfrac{1}{N}$ in the [definition](#eq:varianceU) for the **Unbiased Variance**.

:::{note}
Bias is not the only thing that matters. Sometimes, the Best Estimator is not the zero bias estimator.
:::

If we continue simplifying [](#eq:varbias), we see that $b = -\dfrac{V[X]}{N} $: there is a negative bias. This means that our Estimator $\hat{V}[X]$ is smaller than the true variance $V[X]$. This makes sense, because in our Estimator we have used a finite number of measurements, which will be clustered around the sample mean $\overline{x}$. The True Variance uses all possible measurements, which must be clustered less tightly about the mean. So it makes sense that the True Variance is larger than the sample variance. The larger we make $N$, the closer the Estimator gets to the True Variance.

:::{tip}
I acknowledge that thinking about the "variance on the variance" can cause issues in terms of making new neural connections[^neurons] in the old wheelbox[^wot]. The approach that works for me is to think of the variance as a parameter of the distribution; some number. All numbers have some uncertainty.
:::

[^neurons]: Learning new things means that neurons in our brains have formed new and lasting connections with other neurons.

[^wot]: The "old wheelbox" is a colloquialism; this is the affectionate yet somewhat frustrated way in which I refer to my brain, as a slightly dusty and creaky machine that nonetheless does what it I want it to most of the time.


Variance Check
: Calculating the variance on the sample variance is not as straightforward as [](#eq:varmean), especially in cases where the data are not normally distributed. 


(sec:central-moments)=
### Aside: Central Moments

:::{dropdown} For those who are interested

We saw that we can write the Variance in terms of the Expectation as [](#eq:truevar). The Variance is the "second central moment", where the central moments $\mu_n$ are as follows:

$\mu_0 = E[(X-\mu)^0] = 1$
: The zeroth central moment is 1, because any number to the power zero is 1, and the expectation of a number is just the number.

$ \mu_1 = E[(X-\mu)^1] = 0$
: The first central moment is 0, because $E[(X-\mu)] = E[X] - E[\mu] =  \mu - \mu$, where we used  E(sum) = sum(E) [](#eq:expect_sum).

$\mu_2 = E[(X-\mu)^2] = \sigma^2$
: The second central moment is the **Variance**, as shown in [](#sec:truvar). This quantifies the **spread** of the distribution

$\mu_3 = E[(X-\mu)^3] = \sigma^3 \gamma_1 $
: The third central moment defines the **Skewness**, $\gamma_1$. This quantifies the **symmetry** of the distribution.

$\mu_4 = E[(X-\mu)^4] = \sigma^4 k $
: The fourth central moment defines the **Kurtosis**, $k$. This quantifies the relative **curvature** of the distribution. Positive kurtosis: fatter tails.

:::

(sec:sev)=
### Aside: Standard Error on the Variance

:::{dropdown} For those who are interested

In the **special case that the data are Normally Distributed**, the SEV is:

```{math}
:label: eq:sev
\mathsf{SEV} = \sqrt{ \dfrac{2\,\sigma_{\mathsf{true}}^4}{N-1} }
```

If our data are normally distributed we can estimate the SEV from data, using the sample standard deviation $\sigma_x$ as an estimate for the true standard deviation $\sigma$, just as we can do for the [SEM](#eq:sem).


Generally, with no assumption of Normality, the SEV is:

```{math}
:label: eq:sev-gen
\mathsf{SEV} = \sqrt{ \dfrac{1}{N} \left( \mu_4 - \dfrac{N-3}{N-1} \sigma^4 \right) }
```
where $\mu^4$ is the fourth central moment, $\mu_4 = E[(X-\mu)^4]$. There is a somewhat accessible proof of [](#eq:sev-gen) on mathematics stack exchange [here](https://math.stackexchange.com/questions/72975/variance-of-sample-variance).


If all this is making you a little queasy, worry not! Estimating the SEV from the variance is a piece of cake using eg the [**Bootstrap Method**](#chapter:Resampling), as we will discover shortly.

:::

````{card}
For a single RV, we can Estimate the variance and its uncertainty directly from the data:

```{math}
:label: eq:var-est-uncert
\begin{aligned}
\hat{V} \pm \hat{\sigma}_{V} & = v_x \pm \sigma_{v_x} \\ \\
                                 & = v_x \pm \mathsf{SEV} \\                                
\end{aligned}
```

where the SEV is the Standard Error (uncertainty) on the Variance, and is most easily and efficiently calculating using e.g. the [**Bootstrap Method**](#chapter:Resampling), as we shall see.

````



(sec:binned-data)=
## What to do when Data are Binned?

Sometimes we do not have access to individual measurements. Instead, the data arrive already summarised in a **histogram**: we know the number of measurements $N_j$ falling in each bin, centred on bin centre $x_j$, for bins $j = 1, 2, \ldots, B$.

The total number of measurements is $N = \sum_{j=1}^{B} N_j$.

We can still estimate the mean and variance by treating each count $N_j$ as $N_j$ measurements all equal to $x_j$:

```{math}
:label: eq:binned-mean
\hat{\mu} = \overline{x} = \dfrac{\displaystyle\sum_{j=1}^{B} N_j\, x_j}{N}
```

```{math}
:label: eq:binned-var
\hat{V} = v_x = \dfrac{1}{N-1}\sum_{j=1}^{B} N_j\, (x_j - \overline{x})^2
```

The SEM is unchanged: $\hat{\sigma}_\mu = v_x^{1/2}/\sqrt{N}$.

:::{note}
These estimators treat each bin as if all its measurements were exactly at the bin centre. This introduces a small **binning bias**, which decreases as the bin width $\Delta x$ decreases.
:::

:::{tip}
A more powerful approach when working with histograms is to fit the parameters of a probability distribution directly to the histogram using the Maximum Likelihood Method or Least Squares — see [](#sec:general-param-est).
:::



(sec:general-param-est)=
## Parameter Estimation Generally


In the two examples above, we looked at using Summary Statistics ($\overline{x}$ and $v_x$) as Estimators ($\hat{\mu}$ and $\hat{V}$) for the parameters of the True Underlying Distribution ($\mu$ and $\sigma^2$).

For a single RV following a (e.g.) Normal distribution, the parameters of the true distribution can be accessed directly from the data in this way. We saw that in the case of $\hat{\mu}$ it is possible to extract the **Uncertainty on the Estimated Mean**, using the [SEM](#sem). 

We can similarly extract the **Uncertainty on the Estimated Variance**, [SEV](#eq:sev-gen) if the data are normally distributed, but this is not quite as straightforward as the SEM; 

The parameters of the true underlying distribution will not always be so easily accessible from the data, so it is helpful to explore general methods for parameter estimation. 

The parameter estimation methods we will explore are the Maximum Likelihood Method and the Method of Least Squares. Both methods compare the data with true distributions having different parameter values, and find the parameter values that minimize the differences between data and truth. This process of minimising the differences is known as **Fitting**, or **Regression**, thus both these terms mean **Parameter Estimation**.


Before we go further, we must define the **Likelihood**.


(sec:likelihood-est)=
## Likelihood

We calculate probabilities by drawing a sample from a given distribution with known parameter(s) $\theta$. We can clearly only do this if we know the form of the PMF $p_K$ or PDF $f_X$, and the value(s) of its parameter(s), $\theta$.

Discrete RV $K$: probability $K=k$ from PMF
: $P(K=k | \theta) = p_K(k; \theta)$

Continuous RV $X$: probability $X\leq x$ from CDF
: $P(X\leq x | \theta) = \int\limits_{x\in S} f_X(x;\theta) dx$


* We know the true parameter.
* We want the probability of finding the data with some value (or some range of values).

**We calculate the probability of measuring our data in a given range by relying on what the true distribution is telling us.**


If we are trying to estimate the unknown underlying true distribution parameter(s) using the data, we clearly cannot use the CDF. We need to turn it on its head:

* We know the data distribution.
* We want the probability of finding the true parameter with some value.

How to do this?

<!--
We could try turning around the PMF definition, letting the parameter $\theta$ be the RV, as this is the thing we seek to measure, and using the data as the conditional:

$P(\theta=\theta_a | k) = p_\theta(\theta_a; k)$
-->

Let's consider a single measurement, $k_1$, and two different PMFs having the same form but different parameter values. One has $\theta=\theta_a$ and the other has $\theta=\theta_b$. 

*Plot showing two PMFs with $\theta_a$ and $\theta_b$ and a single data measurement $k_1$*

Based on this data point $k_1$ alone, I would infer that $\theta_a$ is the better parameter estimate, ie that $\theta_a$ is **more likely** to be the true parameter of the data than $\theta_b$.

I will define the **Likelihood** of the data being described by $\theta_a$:

$\mathcal{L_{\theta}}(\theta_a; k_1) = p_K(k_1; \theta_a)$

The Likelihood of true parameter $\theta_a$, given the single-valued data point $k_1$ is equal to the probability of measuring $k_1$, given the single-valued true parameter is $\theta_a$.

I find $\mathcal{L_{\theta}}(\theta_a; k_1) > \mathcal{L_{\theta}}(\theta_b; k_1)$, and therefore conclude that $\theta_a$ is the most likely of the two parameter values I considered.

This is a toy version of the **Maximum Likelihood Method** for parameter estimation. The "toy" aspect arises from an important issue with the above example: we only used one data point. **We are very unlikely to successfully estimate a parameter using a single data point!**. The uncertainty on such an estimate would be so large as to make the estimate useless.


<!--
, with two **important caveats**: 

Problem 1: Validity
: In the thought experiment above, we have not yet considered the strict rules around what we can call a probablity.

**Is this so-called PMF $p_{\theta}$ a valid probability? No!**

To write down a valid PMF, we would need to insist that $\sum\limits_{\theta \in S} p_{\theta}(\theta; k_1 ) =1 $, with S as the support for $\theta$: the set of parameter values that result in a non-zero probability for this data point $k_1$.

We cannot do this. For a start, there is no way to know if the parameter follows a discrete or continuous distribution, and it doesn't make sense to ask the question. The parameter has a value, not a distribution. 

But we don't have to throw away our thought experiment, we just cannot refer to thing we constructed as a probability. Instead, we will call it a **Likelihood**.

```{math}
:label: eq:like_is_pmf
\mathcal{L}_{\theta}(\theta_a; k_1) = p_K(k_1; \theta_a)
```

Note that for this single data point $k_1$ (or for any single data point), the Likelihood $\mathcal{L_{\theta}}$ is equal to the PMF $p_K$. The validity issue arises when we consider more than one data point.

Problem 2: The Goal
: **We are very unlikely to successfully estimate a parameter using a single data point!**. The uncertainty on such an estimate would be so large as to make the estimate useless.
-->

Let's consider two data points, and just consider a single parameter value, $\theta_a$. This is the parameter value for which we are going to calculate the Likelihood.

*Plot showing two PMFs with $\theta_a$ and $\theta_b$, and two single data measurements $k_1$ and $k_2$ being used to determine the probability of $\theta_a$ and $\theta_b$*

Now we have:

$\mathcal{L_{\theta}}(\theta_a; k_1) = p_K(k_1; \theta_a)$

$\mathcal{L_{\theta}}(\theta_a; k_2) = p_K(k_2; \theta_a)$

The question now is: "What is the Likelihood that the parameter value is $\theta_a$, given **both** data measurements $k_1$ and $k_2$?"

This is a **Joint Probability**: $P( k_1 \cap k_2 | \theta_a)$. Noting that the two measurements are IID, we can use the [Multiplication Rule](#sec:multrule) to write the Joint Probability of these two measurements as a product [](#eq:multrule-twodata):

```{math}
:label: eq:multrule-twodata
\mathcal{L_{\theta}}(\theta_a; k_1, k_2) = P( k_1 \cap k_2 | \theta_a) = P( k_1 | \theta_a )\, P( k_2 | \theta_a ) = p_K(k_1; \theta_a)\, p_K(k_2; \theta_a)
```

This is nice because it allows us to extrapolate to the entire dataset $k$ of $N$ measurements $k_1, k_2, ..., k_N$:

```{math}
:label: eq:multrule-alldata
\mathcal{L_{\theta}}(\theta_a; k) = \prod_{i}^{N} p_K(k_i; \theta_a) 
```

**The Likelihood of the data $\mathcal{L_{\theta}}$** having an underlying probability distribution with true parameter $\theta_a$ **is the product of the probabilities $p_K$** evaluated at each data point, where the PMF $p_K$ has parameter $\theta_a$.

If the RV in question is continuous, exatly the same definition holds, with the PMFs (probabilities) replaced by PDFs (densities): The Likelihood of the data $\mathcal{L_{\theta}}$ is the product of the densities $f_X$.


We can use the Likelhood to make a statement about the probability of a given estimate $\theta_a$.

For a discrete RV $K$:

$P(\theta_a | K ) =  \dfrac{ \mathcal{L_{\theta}} ( K | \theta_a ) } { \sum\limits_j^Z \mathcal{L_{\theta}} ( K | \theta_j ) }$

This is a valid probability, because the numerator is a Likelihood for the data (a positive number, but with no requirement that it is in 0,1 ) and the denominator is the sum of all possible Likelihoods for the data. (This is equivalent to a Bayesian posterior under a **uniform prior** over all candidate parameter values: the prior probability $P(\theta_j)$ is constant for all $j$, so it cancels from numerator and denominator.) 


For a continuous RV $X$ we replace the sum over all possible parameter values with an integral over all possible parameter values:

$P(\theta_a | X ) =  \dfrac{ \mathcal{L_{\theta}} ( X | \theta_a ) } { \int\limits_j^Z \mathcal{L_{\theta}} ( X | \theta_j ) d\theta }$

The probability that this parameter value (given the data) is the best parameter estimate is equal to the Likelihood of the data for this parameter divided by the Total Likelihood for all possible parameter values (given the same data).


(sec:interval-est)=
## Interval Estimation

A **Confidence Interval** is an estimated range of values such that we expect CL% of repeated experiments to yield a value within that range. CL is the Confidence Level, for example 95% CL means that in 95% of repeated experiments, we expect the value to fall in the Confidence Interval. The Confidence Interval is a Frequentist construct. Like all frequentist statistics, it tells us nothing about a single experiment.

A **Credible Interval** is, in my opinion, a much more intuitive interval. It represents the most probable range of outcomes for a given single experiment. The Credible Interval is a Bayesian construct. It uses Prior Knowledge to allow us to estimate the range of probable outcomes for a single experiments.

:::{note}
People such as myself (particle physicists or similarly dorky types) might get home from work to find the oven on, and declare "I am 3 sigma confident that I did not leave the oven on". Because Confidence Levels are frequentist, this should be interpreted as "In 741 days, with conditions identical to today, where I came home to find the oven on, only 1 of these occasions would be my fault." Of course this is not actually what I mean when I say that. I mean I am 99.73% sure I did not leave the oven on this morning. So I should really say "I did not leave the oven on- 3 sigma credible interval". 
:::

We have seen [](#critical-region-and-significance-level) that the size of the confidence interval is set using a predefined significance level $\alpha$, for example $\alpha=5\%$ gives us a 95% CL, and the confidence interval represents the central 95% of a normal distribution centered on the estimated mean $\hat{\mu} = \overline{x}$. The 95% CL can also be expressed as $\pm 1.96\sigma$, which means the same thing. Note that the Confidence Interval is predefined - it does not rely on the measured p value and Z value from the data.

We have also seen that the uncertainty on the estimated mean reduces with the number of measurements used [](#sem).

Sometimes, we will find that we have an estimate for the mean and cannot use the SEM for the uncertainty, because our data are not normally distributed. Other times, we will be estimating some parameter other than the mean, and it is not clear how to quantify the uncertainty from our data. 

Given that measurements without quoted uncertainties are meaningless, we must estimate the uncertainty. The [**Bootstrap Method**](#chapter:Resampling) is an excellent way of doing this.







