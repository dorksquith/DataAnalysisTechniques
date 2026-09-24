# Normal Tests (```wip```)

## Distribution of Means

Real world measurements do not have infinite datasets. One consequence of this is that the variance in our small datasets will not accurately reflect the true variance of the underlying PDF.

<!--A set of 50 measurements of X will have some mean, $\overline{x}$, which is a summary statistic of the measurements. A different set of 50 measurements of X will have some other mean, because real world measurements are samples from a distribution (PDF or PMF) of an infinite number of measurements, languishing in the theoretical universe.-->

We know from the [CLT](#CLT) that if we measure the mean from many independent samples, then plot the measured means, we will get a Gaussian distribution (given large enough sample sizes, although the  $N\rightarrow \infty$ criterion in the CLT is clearly impossible to achieve in real life!).

<!--Note: I am using $N$ to denote the size of each sample, which is what the CLT cares about. I am using $n$  for the number of samples, which is the number of means we will have to plot.-->

**Example**:

I have $n=80$ employees, and send each of them to a different town in the UK and ask each of them to record the number of text messages sent in the last day by $N=50$ adult women, randomly selected on the streets. At the end of this endeavour, each employee will have an [IID](#IID) sample with some mean and variance, such that my combined sample will be formed of $n=80$ means. The distribution of these mean values will resemble a Gaussian, even though the sample sizes were $N=50$, which is nowhere near the $N\rightarrow \infty$ suggested by the CLT.

> **IID Reminder**:  They are **Independent**, because the mean measured in Stockport will have no effect on the mean measured in Brixton.  They are **Identically Distributed** because we are asking the same question everywhere: how many texts have you sent in the last 24H.




**Question**:  How confident are we in this resulting Gaussian-like distribution of means?


If I did this exact same experiment again on a different day in 80 other towns, I would not get exactly the same distribution of means. Would the mean of my resulting distribution be fairly stable, or would it be likely to fluctuate significantly? If the variable is IID, then this depends only on the number of people asked (in each town) $N$.

We quantify the uncertainty on the mean of the Gaussian using the [**Standard Error on the Mean (SEM)**](#sem).


### Expectation $E[\overline{x}] = E[X]$

The mean of our measurements (yes, it is a mean of means), $\overline{x}$, will not be the same as the True Mean , E[X], but they will be similar according to the [LLN](#LLN).

We can [show](#eq:expect_mean) that the Expectation of (each of) the sample means E[$\overline{x}$] is equal to the True Mean E[X], where we have used [](#eq:expect_sum) for the second step.

```{math}
:label: eq:expect_mean

\begin{align*}

\mathsf{E\left[\,\overline{x}\,\right] }&
\mathsf{= E\left[\dfrac{1}{N}\sum\limits_i^N x_i\right] }
 &\;\; \mathsf{\textcolor{blue}{because} \;\;} 
& \mathsf{\overline{x}  = \frac{1}{N} \sum \limits_i^N x_i}\;\;\\

& = \mathsf{\dfrac{1}{N} \sum\limits_i^N E\left[ x_i \right] }
 &\;\; \mathsf{\textcolor{blue}{because} \;\;} 
& \mathsf{E[sum] = sum(E)}
\\
  & = \mathsf{\dfrac{1}{N} \sum\limits_i^N E[X] }
& \;\; \mathsf{\textcolor{blue}{because\;x_i\;are\; iid,\; so\;} \;\;} 
& \mathsf{E[x_i] = E[X]\; \forall\; x_i}
\\
  & = \mathsf{\dfrac{1}{N} N E[X]= E[X]}
\end{align*}

```


### Variance $v_{\overline{x}} = \dfrac{1}{N}V[X]$

The Standard Deviation of our measurements (the SEM) will also not be the same as the True Standard Deviation , and we don't expect it to ever be, even if each employee asks a million people, because they are different things. The SEM is the measured variation of the means, which will depend on $N$ (how many people each of my employees ask for their data), whereas the True Standard Deviation is a fixed value, a parameter of the underlying distribution.


We can show [](#eq:var_mean) that the Variance of the sample means $v_{\overline{x}} \equiv V[\overline{x}]$ is equal to the True Variance[X] divided by the number of measurements $N$, where we have used [](#eq:var_sum) and [](#eq:var_multc).

<!--The Variance on the measured means is $\mathsf{V[\,\overline{x}\,] = \sigma^2_{\overline{x}} = \mathsf{SEM}^2}$.

The True Variance is $\mathsf{V[X] = \sigma^2_{true} }$.-->


```{math}
:label: eq:var_mean

\begin{align*}
\mathsf{V\left[\,\overline{x}\,\right] }&
\mathsf{= V\left[\dfrac{1}{N}\sum\limits_i^Nx_i\right] }
 &\;\; \mathsf{\textcolor{blue}{because} \;\;} 
& \mathsf{\overline{x}  = \frac{1}{N} \sum \limits_i^N x_i}\;\;\\

& \mathsf{=  \sum\limits_i^N V\left[ \dfrac{1}{N}x_i \right] }
 &\;\; \mathsf{\textcolor{blue}{because} \;\;} 
& \mathsf{V[sum] = sum(V)} \mathsf{\;for\;iid\;}x_i
\\
  & = \mathsf{\dfrac{1}{N^2} \sum\limits_i^N V[x_i] }
& \;\; \mathsf{\textcolor{blue}{because\;} \;\;} 
& \mathsf{V[ax] = a^2\,V[x]}
\\
  & \mathsf{= \dfrac{1}{N} V[X] }
& \;\; \mathsf{\textcolor{blue}{because\;} \;\;} 
& \mathsf{V[X] = \dfrac{1}{N}\,V[x_i]}
\\

\end{align*}

```


The [proof](#eq:var_mean) tells us that the variance on the measured means, $v_{\overline{x}} \equiv V[\overline{x}]$, is **smaller than the True Variance, V[X] by a factor $N$**.

```{math}
:label: eq:var_mean1

v_{\overline{x}} = \dfrac{1}{N} V[X]

```

<!--If my employees asked ten times as many people for their data, the means they get will be more reliable, and the variances would become ten times smaller.

The True Variance is irreducible. It is a property of the underlying PDF that lives in the theoretical universe.-->

The Variance on the  Means is a (linear) function of this True Mean, and also has a strong inverse dependence on the number of measurements we take. In a nutshell, **the more measurements we take, the smaller the uncertainty on our measurement becomes**.

(sem)=
### Standard Error on the Mean $\mathsf{SEM}$

The Standard Error on the Mean is the square root of the Variance on the sample means:

$$\label{eq:SEM} \mathsf{\mathsf{SEM} = \sigma_{\overline{x}}  = \dfrac{\sigma_{true} }{\sqrt{N}}    }$$


This is an important result for Normal Tests, as we shall see.

```{card} $\sigma_x$ and $\sigma_{\overline{x}}$
The difference between $\sigma_x$ and $\sigma_{\overline{x}}$ is of crucial importance!

* $\sigma_x$ is a summary statistic for the sample, telling us how much variability is within the sample. 

* $\sigma_{\overline{x}}$ is not a summary statistic; it is a theoretical statistic, and it tells us how precise our mean measurement will be for a given number of measurements, N , and a true underlying variability, $\sigma_{true}$.
```



## Worked Example: A Tale of Three Cities

Let's consider the data displayed in [](#fig:sem1) below. This is simulated data for a scenario where I have asked people in three different cities to collect data on the number of hours slept the previous night. In this example, each of them have asked N=2 people.

:::{figure} 
:label: fig:sem1
![](figures/MDA-demo-N2-semFalse-seed1.png)

My fake data for the hours slept each night in three cities, with $N=2$. The green star and shaded bar at the top shows the true distribution I used to generate the data. I used Norm( $\mu$ =7.5H , $\sigma$=1.2H). The blue crosses show the data points, the circles show their means, and the shaded blue-grey bars show the $1\sigma$ ranges.

:::

**Observations**:

The two people asked in Glasgow both gave very similar answers, which has resulted in a small [standard deviation](#eq:std) on the Glasgow sample. **Using the standard deviation for the shaded "error" bar is misleading** - it suggests that we have more confidence in the Glasgow measurement, when in fact they are all statistically equivalent in their uncertainty, which is large.

This becomes apparent as we take more data, and the means start to approach the expected value of 7.5 H. The mean number of hours slept in three different cities, using sample sizes of 3, 30, and 100, are shown in [](#fig:sem2), [](#fig:sem3), and [](#fig:sem4) respectively. The blue shaded bars represent the standard deviations on the samples, which is **not** a good estimate for the uncertainty on the mean, as it represents the range in which 68\% of the data fall. 


:::{figure} 
:label: fig:sem2
![](figures/MDA-demo-N3-semFalse-seed1.png)

Fake data, N=3.
:::

:::{figure} 
:label: fig:sem3
![](figures/MDA-demo-N30-semFalse-seed1.png)

Fake data, N=30.
:::

:::{figure} 
:label: fig:sem4
![](figures/MDA-demo-N100-semFalse-seed1.png)

Fake data, N=100.
:::


We have two problems:

1. How do we factor in our increased confidence as the number of measurements increases?

2. How to we express the uncertainty on the mean when we have a tiny number of measurements?



### The Uncertainty on the Mean

The solution to problem 1 is to use the [SEM](#eg:SEM), but we must assume we do not know the true standard deviation  $\sigma_{true}$ and instead use the measured sample standard deviation $\sigma_{x}$  as our best **Estimate** of the SEM[^estimators].

[^estimators]: we will cover estimation at length later [](#sec:Estimators).

 $\mathsf{SEM} =  \dfrac{\sigma_{true} }{\sqrt{N}}$: the Standard Error on the Mean requires knowledge of the true standard deviation.

 $\mathsf{\widehat{SEM}} =  \dfrac{\sigma_{x} }{\sqrt{N}}$: the **Estimated** Standard Error on the Mean is denoted with a hat symbol $\widehat{}$ .


I have replaced the standard deviation with the $\mathsf{\widehat{SEM}}$ for the blue shaded barsin the  plots for [N=3](#fig:sem5), [N=30](#fig:sem6), and [N=100](#fig:sem7).[^estsem]

[^estsem]: $\mathsf{\widehat{SEM}}$ is a good estimate for the uncertainty on the mean if the sample size is not very small. A rule of thumb for "very small" is $N \lessapprox 30$. 


:::{figure} 
:label: fig:sem5
![](figures/MDA-demo-N3-semTrue-seed1.png)

Fake data, N=3, with the shaded blue bar now giving what would be a sensible estimate of the uncertainty if we weren't suffering from such devestatingly small dataset [](#sosmall).
:::

:::{figure} 
:label: fig:sem6
![](figures/MDA-demo-N30-semTrue-seed1.png)

Fake data, N=30, with the shaded blue bar now giving a sensible estimate of the uncertainty.
:::

:::{figure} 
:label: fig:sem7
![](figures/MDA-demo-N100-semTrue-seed1.png)

Fake data, N=100, with the shaded blue bar now giving a sensible estimate of the uncertainty.
:::


(sosmall)=
### Small Datasets


Using the estimated SEM does not help us with tiny samples. A standard deviation from 2 or 3 measurements is not going to be accurate, and the SEM is just the standard deviation reduced in size by a factor $\dfrac{1}{\sqrt{N}}$, making it even more misleading than $\sigma_x$ for small sample sizes.


If we have only a handful of measurements, and there is no way to collect more data, how should we express the uncertainty on the mean?

**Option 1 - Don't do this**
: I will simply quote my raw measurements, making it clear there are only N, and let the reader decide the uncertainty.

  Dangerous! Almost everyone has almost no understanding of statistics and uncertainty. Because it is both hard and boring :).

**Option2 - The Frequentist Approach**
: I cannot know my uncertainty from such a small sample, so I will not share the incomplete measurement of the mean.

  Sad but safe.

**Option3 - The Bayesian approach**
: Use "common sense": my own experience of sleeping tells me that there can be natural +/- 1.5 H fluctuations on the number of hours I sleep each night. I expect that this could be as high as +/- 3H  in some people. I will use $\mathsf{\widehat{\sigma} = \mathsf{3H}}$. 
  
  Less sad and safe than the frequentists' "abstinence" approach, but reasonable in my opinion, given the very conservative (over-inflated, really) estimate. And it just feels wrong to not share my data at all, given that there is some limited information in it.


## The Z Test: compare data mean with hypothesis mean

```{card} Z Test
**Example usage**
: Test if the sample mean $\overline{x}$ equals a hypothesised mean $\mu_0$.

**Restrictions**
: Data are $\mathsf{X\sim Norm(\mu,\sigma)}$
  True Variance $\sigma^2$ is known[^wot].
```

[^wot]: Why on earth would we know the true variance? This is very unrealistic, so is not used irl. But this is the "official" definition, so I am sharing it with you as-is.

Recall that the [Z value](#eq:z) for a measurement x tells us how many standard deviations away from the mean that value is.

The Z Test defines this statistic slightly differently, because we no longer want to compare a measurement (x) to a summary statistic ($\overline{x}$). Now, we want to compare a summary statistic ($\overline{x}$) to a hypothesis ($\mu_0$):

```{math}
:label: eq:zstat

\mathsf{Z\; Test\; Statistic = \dfrac{ \overline{x}-\mu_{_0} }{ \mathsf{SEM}} }

```

The measurement (x) is replaced with the sample mean, $\overline{x}$,  and the reference ($\overline{x}$) is replaced with the True Mean of the Null Hypothesis $\mu_0$ with which we are comparing our data. It would not make sense to use the sample standard deviation in the denominator, because we are comparing means rather than values, so the correct thing to use in the denominator is the SEM.

(ZTestExample)=
### Z Test Example: Hours slept in Stockport

**Null Hypothesis:**
: $H_0$: the mean number of hours slept by adult women is $\mathsf{\mu_0 = 7.33\,H}$.

**Research Question:**
: Is the mean sleep achieved by residents of Stockport equal to that predicted by the **Null Hypothesis**?


```{card} [1] Design Test
- Significance Level : $\alpha=0.05$
- Alternate hypothesis, $H_1$:  $\mathsf{\mu_0 \textcolor{red}{\neq} 7.33\,H}$ : this means the test is two-sided.
```

```{card} [2] Look at Data:
- Sample mean: $\overline{x} = \mathsf{7.40\,H}$
- Sample variance: $V[x] = \mathsf{1.68\,H^2}$
- Sample size: $N=100$
```

```{card} [3] Calculate [Test Statistic](#eq:zstat)
- The formula for the denominator is $\mathsf{SEM} = \sigma_{\overline{x}}  = \dfrac{\sigma_{true} }{\sqrt{N}}$
- We don't know $\sigma_{true}$ so we will **Estimate** it[^note1]. We know the variance of the sample, so we will use that, and remind ourselves these are Estimates by giving them little hats to wear:
$\mathsf{\widehat{SEM} = \hat{\sigma}_{\overline{x}} = \dfrac{\sigma_{x} }{\sqrt{N-1}}   }$[^note2]
- $V[x] = \mathsf{1.68\,H^2}$ so the square root of this gives us our numerator, the sample standard deviation.

- $\mathsf{ \widehat{SEM} =\dfrac{\sigma_{x} }{\sqrt{N-1}}   =  \dfrac{\sqrt{1.68\,H^2}}{\sqrt{99}}    \approx 0.130\,H  }$
-$\mathsf{Estimated\;z\; test\; statistic= \dfrac{ \overline{x}-\mu_{_0} }{\mathsf{SEM}} 
= \dfrac{ 7.40\, H - 7.33\, H }{0.13\, H} \approx  0.538}$

```

[^note1]: By using the data to **Estimate** the true variance, and therefore the SEM, this is not really a Z Test any more. It is now the same form as Students' T Test (next), but Student's T has a key difference as we shall see.

[^note2]: Why did the denominator change from $\mathsf{\sqrt{N} \rightarrow \sqrt{N-1}}$? The short answer is that it is because we are using the sample variance to estimate the true variance, and this estimate is known to be Biased. I did a [proof of this](#fig:fig:lilyproof_varbias) (bottom of this page) last year when I first started teaching this module because I could not find one anywhere. It is long. If you can find a shorter way, please send it to me.

<!--
$\mathsf{V[X] = \dfrac{N}{N-1}\; V[x]}$
: The true variance V[X] is slightly underestimated by the sample variance V[x].

$\mathsf{\sigma_{true} = \sqrt{ \dfrac{N}{N-1}}\; \sigma_{x}}$
: The standard deviation is the square root of the variance.

$\mathsf{\dfrac{\sigma_{true}}{\sqrt{N}} =  \dfrac{1}{\sqrt{N}} \dfrac{\sqrt{N}}{\sqrt{N-1}}\; \sigma_{x}
=  \dfrac{\sigma_{x}}{\sqrt{N-1}}\; }$
: We recover our Unbiased Estimate of the True Variance.
-->


An estimated  test statistic of 0.538 means that our data mean is 0.538 standard deviations above the value the Null Hypothesis claims is the true mean.




To [convert this into a p value](#pfromz) that we can use directly to reject or not-reject $H_0$, we use python: 

```{code-cell} python
from scipy.stats import norm

# the two-sided p-value is 2* the survival function

p_value = 2*norm.sf( abs(0.538) )

print(f"P value: {p_value}")

```

This p value is (much) larger than our pre-decided significance level, $0.59 > 0.05$ , so  **the result is Negative, we do not reject the null hypothesis**.

> It may be helpful to keep a reference such as [](#fig:pvalzval) to hand to help with getting an understanding of how Z values and p values are related. It is very simple to convert one to another, but even better to have an instinct such as knowing a Z value of 0.5 puts us right in the peak of the normal distribution.

:::{figure} 
:label: fig:pvalzval
![](figures/StandardNormPvalZval.png)

The p value and Z value.
:::


## The Student's T Test

The one-sample T Test is very similar to the Z Test, but is **particularly well suited to small datasets**. This is because instead of using the Normal distribution, it uses the Student's T[^student] distribution, which is like the Normal distribution but with fatter tails [](#fig:tpdf).

[^student]: William Gosset is the Student. He used a code name at his employers' (Guiness) request.

The T distribution has one parameter: the number of degrees of freedom, with symbol [$\nu$](#eq:Tnu). The T PDF $\nu=1$ corresponds to a sample size of $N=2$, because we are measuring a single parameter, the mean. For reasonably sized datasets of $N \gapprox 30$, the T distribution and Norm are barely distinguishable.

```{math}
:label: eq:Tnu

\nu = \mathsf{N_{measurements} - N_{parameters}}

```


:::{figure} 
:label: fig:tpdf
![](figures/StudentsT2.png)

The T PDF for different values of the parameter $\nu$ (Number of degrees of freedom) on a linear scale (left) and on a log scale (right). The standard normal distribution Norm(0,1) is also shown as a red dashed line.
:::



### T Test Example: Hours slept in Brixton

**Null Hypothesis:**
: $H_0$: the mean number of hours slept by adult women is $\mathsf{\mu_0 = 7.33\,H}$.

**Research Question:**
: Is the mean sleep achieved by residents of Brixton equal to that predicted by the **Null Hypothesis**?


```{card} [1] Design Test
- Significance Level : $\alpha=0.05$
- Alternate hypothesis, $H_1$:  $\mathsf{\mu_0 \textcolor{red}{\neq} 7.33\,H}$ : this means the test is two-sided.
```

```{card} [2] Look at Data
- Sample mean: $\overline{x} = \mathsf{8.14\,H}$
- Sample variance: $V[x] = \mathsf{0.31\,H^2}$
- Sample size: $N=3$
```

```{card} [3] Calculate [Test Statistic](#eq:zstat)
- $\mathsf{ \widehat{SEM} =\dfrac{\sigma_{x} }{\sqrt{N-1}}   =  \dfrac{\sqrt{0.31\,H^2}}{\sqrt{3}}    \approx 0.321\,H  }$
-$\mathsf{Estimated\;T\; test\; statistic= \dfrac{ \overline{x}-\mu_{_0} }{\mathsf{SEM}} 
= \dfrac{ 8.14\, H - 7.33\, H }{0.321\, H} \approx  2.52}$

```


To [convert this into a p value](#pfromz) we use python again, taking care to use the T distribution rather than the Normal distribution: 

```{code-cell} python
from scipy.stats import t # <= t is scipy stats name for the T distribution

# for norm.sf we passed only Z, but for t we must also pass the parameter nu of the T PDF, which is 3-1=2 for this N=3 dataset.

p_value = 2*t.sf( abs(0.538), df = 2 ) # <= df is scipy's name for the parameter nu

print(f"P value: {p_value}")

```


## The Student's T Test: two samples

Incoming...

## The F Test (one-way ANOVA)

Incoming...

## Two-sample F Test Example

Incoming...



:::{figure}
:label: fig:lilyproof_varbias

![](/figures/var-proof-long.png)

Proof that the sample variance is a biased estimator for the true variance.
:::