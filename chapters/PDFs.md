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

(chapter:pdf)=
# Continuous Probabilities & Likelihoods

I strongly recommend the [Probability Playground](https://probabilityplayground.com/standarduniform) to help become familiar with special distributions. It has all of the distributions we will be thinking about, plus more!


* `scipy.stats` distributions: `uniform, expon, chi2 `


---

(sec:pdf_whipdf)=
## Which PDF?

**Continuous RVs** model measurements, waiting times, proportions. The most commonly used continuous distributions are: Uniform, Exponential, Normal, Chi-squared, Gamma, Beta, Pareto, Dirichlet.


| Situation | Distribution |
|-----------|-------------|
| Equally likely values on [a, b] | [Uniform](#sec:pdf_uni) |
| Waiting time between events | [Exponential](#sec:pdf_exp) |
| Symmetric, bell-shaped variation | [Normal](#sec:pdf_norm) |
| Sum of squared standard normals | [Chi-squared](#sec:pdf_chi2) |
| Flexible positive-valued data | Gamma |
| Values constrained to [0, 1] | Beta |
| Heavy-tailed (power-law) data | Pareto |
| Proportions summing to 1 | Dirichlet |


---

(sec:pdf_uni)=
## The Uniform Distribution

A Continuous RV with a flat probability distribution is 

$X\sim$ Uniform($a,b$).

The parameters are $a,b$: indicating the lower and upper bound of the support.

---

(sec:pdf_unipk)=
### PDF & Kernel

The Uniform PDF is:

```{math}
:label: eq:pdf_unipdf
f_X(x) = \dfrac{1}{b-a} \;\; ; \;\;  a\leq x \leq b
```

A plot of the Uniform PDF for chosen parameter values $a=-2,b=2$ is shown in [](#fig:pdf_uni) along with the Cumulative Distribution Function (CDF).


:::{figure}
:label: fig:pdf_uni

![image](../figures/Uniform.png)

The Uniform PDF is shown in blue with reference to the left-hand $y$-axis, and the CDF is shown in red with reference to the right-hand $y$-axis. Because $X$ is a continuous RV, the left-hand $y$-axis is the Density rather than the Probability. The CDF is the Cumulative probability, which must total 1 as a sum over all possible outcomes.
:::


For a PDF to be valid, it must satistify [Kolmogorov's normalisation axiom](#eq:pe_kol2).

Is $\int\limits_{S}\, f_X(x)\, dx =1$ ?

Normalisation check
: $\int\limits_{x=a}^b\, \dfrac{1}{b-a}\, dx = \dfrac{b-a}{b-a} = 1$: the Uniform PDF is valid (by definition).

---

(sec:pdf_uniprob)=
### Probabilities

The Cumulative Probability (CDF) for a continuous RV is the integral over the range of the lower bound of the support to the value at which we want to know the CDF.

The integral is over the PDF, from the lower bound of the support (or $-\infty$, equivalently) to the value we seek to calculate, $x_1$ :

```{math}
:label: eq:pdf_contcdf
F_X(x_1) = P(X\leq x_1) = \int\limits_{x=-\infty}^{x_1}\, f_X(x) \, dx
```

The Uniform CDF is:

```{math}
:label: eq:pdf_unicdf
F_X(x_1) = P(X\leq x_1) = \int\limits_{x=a}^{x_1}\, \dfrac{1}{b-a} \, dx = \dfrac{x_1-a}{b-a}
```

---

(sec:pdf_uniev)=
### Expectation & Variance

The Uniform PDF has
: $$ \label{eq:pdf_uniexp} E[X] =  \dfrac{b+a}{2}$$
  
  $$ \label{eq:pdf_univar} V[X] =  \dfrac{(b-a)^2}{12}$$


The **Expectation** is calculated as:

$E[X]= \int\limits_{a}^{b}\, \dfrac{x}{b-a} \, dx = \dfrac{b^2-a^2}{2(b-a)}= \dfrac{b+a}{2}$

The expectation of the square is:

$ E[X^2] = \int\limits_{a}^{b}\, \dfrac{x^2}{b-a} \, dx =  \dfrac{b^3-a^3}{3(b-a)} =  \dfrac{b^2+ab+a^2}{3}$

The **Variance** is calculated as:

$V[X]  = E[X^2] - E^2[X] = \dfrac{b^2+ab+a^2}{3} -  \dfrac{b^2 + 2ab + a^2}{4} = \dfrac{(b-a)^2}{12}$


---

(sec:pdf_unipy)=
### Python

```{code-cell} python

def ExploreUniform(a,b):

  # import the uniform distribution
  from scipy.stats import uniform

  # set up an instantiation of the distribution with our chosen parameter values a,b
  uni =  uniform(loc=a, scale=b-a) 

  # access the lower bound and upper bound of the support
  lb, ub = uni.support()

  print(f"Support is {lb} <= x <= {ub}")

  # calculate the cumulative probability of x<=(a+b)/2
  cdf1 = uni.cdf((a+b)/2)
  print(f"P(X<={(a+b)/2}) = {cdf1}")

  # extract the true mean and variance
  mean, var = uni.stats(moments='mv')
  print(f"Expectation = {mean}, Variance = {var}")

ExploreUniform(0.45,5)

```

---

(sec:pdf_unilike)=
### Likelihoods

Suppose we observe $N$ measurements $\mathbf{x} = (x_1, x_2, \ldots, x_N)$ drawn independently from Uniform($a,b$), but the parameters $a$ and $b$ are unknown.

The Likelihood is the joint PDF evaluated at the observed data:

```{math}
:label: eq:pdf_unilik
\mathcal{L}(a,b; x) = \prod_{i=1}^{N} \dfrac{1}{b-a} = \dfrac{1}{(b-a)^N}

```

Where $a \leq x_i \leq b \;\forall\; i$


The Log Likelihood is then:

```{math}
:label: eq:pdf_unill
\ln \mathcal{L}(a,b; x) = -N\ln(b-a)
```

Unlike for the PMF likelihoods, **we can't differentiate** [](#eq:pdf_unill) to find the MLE. This is because the support itself depends on the parameters: the constraints $a \leq x_i \leq b$ mean the likelihood is zero for any $a > x_{(1)}$ or $b < x_{(N)}$, where $x_{(1)} = \min(x_i)$ and $x_{(N)} = \max(x_i)$ are the **order statistics**.

Within the feasible region, $\ln\mathcal{L}$ is maximised by minimising $(b-a)$. The tightest feasible interval is:

```{math}
:label: eq:pdf_unimle
\hat{a} = x_{(1)} = \min(x_i), \qquad \hat{b} = x_{(N)} = \max(x_i)
```

This makes intuitive sense: the widest the interval needs to be is just enough to include all observed values.

---

(sec:pdf_exp)=
## The Exponential Distribution

A Continuous RV with an Exponential probability distribution is:

$X\sim$ Expon($\lambda$)

The parameter is $\lambda$, which is a rate parameter (similar to [Poisson($\lambda$)](#sec:pmf_poi) ). 

$X\sim$ Expon($\lambda$)
: the RV $X$ is a **time for a given count** (1/rate, continuous)

$K\sim$ Poisson($\lambda$)
: the RV $K$ is the **count in a given time interval** (a rate, discrete)

---

(sec:pdf_exppk)=
### PDF & Kernel

The Exponential PDF is:

```{math}
:label: eq:pdf_exppdf
f_X(x) = \lambda e^{-\lambda x} \;\; ; \;\;  0\leq x \leq \infty
```

The **Kernel** of the Exponential PDF is given in [](#eq:pdf_expkern).

```{math}
:label: eq:pdf_expkern
 e^{-\lambda x}
```

A plot of the Exponential PDF for chosen rate parameter value $\lambda=1.3$ is shown in [](#fig:pdf_exp) along with the Cumulative Distribution Function (CDF).

:::{figure}
:label: fig:pdf_exp

![image](../figures/Expon.png)

The Exponential PDF is shown in blue with reference to the left-hand $y$-axis, and the CDF is shown in red with reference to the right-hand $y$-axis. The (rate) parameter value is $\lambda=1.3$, meaning the expected waiting time between successful events is the inverse of this: $1/1.3$, in units of time. The CDF is the Cumulative probability, which must total 1 as an integral over all possible outcomes.
:::


For a PDF to be valid, it must satistify [Kolmogorov's normalisation axiom](#eq:pe_kol2).

Is $\int\limits_{x_i \in S}\, f_X(x_i) \, dx =1$ ?

Because we have a nice easy integral (exponentials always are), we will just do it:

```{math}
\begin{aligned}
\int\limits_0^{\infty}\, \lambda e^{-\lambda x} \, dx & = -\dfrac{\lambda}{\lambda} e^{-\lambda x} \bigg|_0^{\infty}\\
&= - \left( e^{-\infty} - e^{0} \right) \\
&= - \left( 0 - 1 \right) \\
&= 1
\end{aligned}
```

Normalisation check
: $\int\limits_0^{\infty}\, \lambda e^{-\lambda x} \, dx = 1$: the
Exponential PDF is valid.

---

(sec:pdf_expprob)=
### Probabilities

The [CDF](#eq:pdf_contcdf) is calculated as the integral of the PDF.

The Probability of $X \leq x$, given some value of the parameter $\lambda$, is:

```{math}
:label: eq:pdf_expcdf
P(X\leq x |\,\lambda) = \int\limits_{x=0}^{x}  \lambda e^{-\lambda x} \, dx = 1-e^{-\lambda x}

```

Example
: What is the probability that I will wait less than or equal to one hour to hitch a lift on a road with an average hourly rate of hitchhiker pickups of $\lambda= 1.3$ ?

  $P(X\leq 1 | \lambda) = 1-e^{-\lambda x} = 1-e^{-1.3} = 0.727$

  What would Poisson give us? The question is now "what is the probability that a count of at least one occurs in a time period of 1H?"

  $1 - P(K=0 | \lambda) = 1 - e^{-\lambda} \dfrac{\lambda^k}{k!}= 1- e^{-1.3}\dfrac{1.3^0}{0!} = 1- 0.273  = 0.727$

---

(sec:pdf_expev)=
### Expectation & Variance

The Exponential PDF has
: $$\label{eq:pdf_expexp}E[X] = \dfrac{1}{\lambda}$$

  $$\label{eq:pdf_expvar}V[X] = \dfrac{1}{\lambda^2}$$

The **Expectation** is an integral product of two terms in $x$, so we have to do it piecewise ([integration by parts](#sec:mma_intbypar)). This is an excellent exercise, try it if you have time ($u = x,\; du= dx\;, v = −e^{-\lambda x},\; dv = \lambda e^{-\lambda x}\,dx$).

The **Variance** is also calculated using integration by parts - I found I needed two rounds of by-parts integration for the variance.

---

(sec:pdf_exppy)=
### Python

```{code-cell} python

def ExploreExpon(mu):

  # import the exponential distribution
  from scipy.stats import expon

  # set up an instantiation of the distribution with our chosen parameter value mu (for lambda, the rate parameter)
  
  expo = expon(scale=1/mu) 

  # access the lower bound and upper bound of the support
  lb, ub = expo.support()

  print(f"Support is {lb} <= x <= {ub}")

  # calculate the cumulative probability of x<=1/mu
  cdf1 = expo.cdf(1/mu)
  print(f"P(X<={1/mu}) = {cdf1}")

  # extract the true mean and variance
  mean, var = expo.stats(moments='mv')
  print(f"Expectation = {mean}, Variance = {var}")

ExploreExpon(12.3)

```

---

(sec:pdf_explike)=
### Likelihoods

Suppose we observe $N$ waiting times $\mathbf{x} = (x_1, x_2, \ldots, x_N)$ drawn independently from Expon($\lambda$).

The Likelihood is the joint PDF evaluated at all the observed data points:

```{math}
:label: eq:pdf_explik
\mathcal{L}(\lambda; x) = \prod_{i=1}^{N} \lambda\, e^{-\lambda x_i}
= \lambda^N\, e^{-\lambda \sum_{i=1}^{N} x_i}
```

The Log Likelihood is:

```{math}
:label: eq:pdf_expll
\begin{aligned}
\ln\mathcal{L}(\lambda;\mathbf{x}) &= N\ln\lambda - \lambda\sum_{i=1}^{N} x_i \\
                                   &= N\ln\lambda - \lambda\, N\overline{x} \\
\end{aligned}
```

The Derivative of the Log Likelihood is:

```{math}
\dfrac{d}{d\lambda}\ln\mathcal{L} = \dfrac{N}{\lambda} - N\overline{x}
```

We find the MLE by setting $\dfrac{d}{d\lambda}\ln\mathcal{L}=0$ and solving for $\lambda$:

```{math}
\hat{\lambda}_{\mathsf{MLE}} = \dfrac{1}{\bar{x}}
```

The MLE of the rate $\lambda$ is the reciprocal of the sample mean waiting time — exactly what we would expect from $E[X]=1/\lambda$.

Example
: A hitch-hiker records the following waiting times (in hours): $1.2, 0.5, 0.8, 1.6, 0.4$. The MLE of the rate is:

  $\hat{\lambda}_{\mathsf{MLE}} = \dfrac{5}{1.2+0.5+0.8+1.6+0.4} = \dfrac{5}{4.5} \approx 1.11 \text{ per hour}$


---
(sec:pdf_norm)=
## The Normal Distribution

We have already looked at the Normal distribution in detail in [](#chapter:norm). Here is a brief summary for reference alongside the other continuous distributions in this chapter.

$X \sim$ Norm($\mu, \sigma$)

The parameters are $\mu$ (the mean, which sets the centre of the distribution) and $\sigma$ (the standard deviation, which sets the width).

---

(sec:pdf_normpk)=
### PDF & Kernel

The Normal PDF is:

```{math}
:label: eq:pdf_normpdf
f_X(x) = \dfrac{1}{\sigma\sqrt{2\pi}}\exp\!\left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right] \;\;;\;\; -\infty \leq x \leq \infty
```

The **Kernel** of the Normal PDF is:

```{math}
:label: eq:pdf_normkern
e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
```


A plot of the Normal PDF for chosen parameter values $\mu=0, \sigma=1$ is shown in [](#fig:pdf_norm) along with the Cumulative Distribution Function (CDF).

:::{figure}
:label: fig:pdf_norm

![image](../figures/Norm.png)

The Normal PDF is shown in blue with reference to the left-hand $y$-axis, and the CDF is shown in red with reference to the right-hand $y$-axis. The parameter values are $\mu=0$ (the mean) and $\sigma=1$. The CDF is the Cumulative probability, which must total 1 as an integral over all possible outcomes.
:::

The normalisation integral requires the **Gaussian integral trick** given in [Math Methods B](#sec:mmb_gastrick) for those with an interest.


Normalisation check
: $\displaystyle\int_{-\infty}^{\infty} \dfrac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/2\sigma^2}\,dx = 1$: the Normal PDF is valid.

---

(sec:pdf_normprob)=
### Probabilities

The Normal CDF has no closed form in elementary functions. It is expressed in terms of the **error function**, $\mathrm{erf}$:

```{math}
:label: eq:pdf_normcdf
F_X(x) = P(X \leq x) = \dfrac{1}{2}\left[1 + \mathrm{erf}\!\left(\dfrac{x-\mu}{\sigma\sqrt{2}}\right)\right]
```

where $\mathrm{erf}(y) = \dfrac{2}{\sqrt{\pi}}\int\limits_0^y e^{-t^2}\,dt$.

In practice we use `scipy.stats.norm.cdf()` to calculate Normal probilities. The probability within $n$ standard deviations of the mean follows the **68–95–99.7 rule**:

$P(\mu - \sigma \leq X \leq \mu+\sigma) \approx 0.683$

$P(\mu - 2\sigma \leq X \leq \mu+2\sigma) \approx 0.954$

$P(\mu - 3\sigma \leq X \leq \mu+3\sigma) \approx 0.997$

---

(sec:pdf_normev)=
### Expectation & Variance

The Normal PDF has:
: $$\label{eq:pdf_normexp}E[X] = \mu$$

  $$\label{eq:pdf_normvar}V[X] = \sigma^2$$

The **Expectation** $E[X] = \mu$ follows from the symmetry of $f_X(x)$ about $x=\mu$: the integrand of $\int\limits_{x\in S} x\,f_X(x)\,dx$ is an odd function[^odd] of $(x-\mu)$, so the integral evaluates to $\mu$ directly.

[^odd]: An odd function satisfies $f(-x) = -f(x)$: it is antisymmetric. The integral of an odd function over any symmetric interval $[-a, a]$ is exactly zero, because the positive and negative contributions cancel perfectly.

The **Variance** is $\sigma^2$ by construction — the PDF is parametrised so that $\sigma$ is exactly the standard deviation of the distribution.

---

(sec:pdf_normpy)=
### Python

```{code-cell} python

def ExploreNormal(mu, sigma):

  from scipy.stats import norm

  # set up an instantiation of the distribution with our chosen parameter values
  n = norm(loc=mu, scale=sigma)

  # access the support
  lb, ub = n.support()
  print(f"Support is {lb} <= x <= {ub}")

  # probability within one and two standard deviations of the mean
  p1 = n.cdf(mu + sigma) - n.cdf(mu - sigma)
  p2 = n.cdf(mu + 2*sigma) - n.cdf(mu - 2*sigma)
  print(f"P(mu - sigma <= X <= mu + sigma) = {p1:.4f}")
  print(f"P(mu - 2sigma <= X <= mu + 2sigma) = {p2:.4f}")

  # extract the true mean and variance
  mean, var = n.stats(moments='mv')
  print(f"Expectation = {mean}, Variance = {var}")

ExploreNormal(0, 1)

```

---

(sec:pdf_normlike)=
### Likelihoods

Suppose we observe $N$ measurements $\mathbf{x} = (x_1, x_2, \ldots, x_N)$ drawn independently from Norm($\mu, \sigma$).

The Likelihood is the joint PDF evaluated at the observed data:

```{math}
:label: eq:pdf_normlik

\begin{aligned}

\mathcal{L}(\mu,\sigma;x) &= \prod_{i=1}^{N}\dfrac{1}{\sigma\sqrt{2\pi}}\exp\!\left[-\dfrac{1}{2}\left(\dfrac{x_i-\mu}{\sigma}\right)^2\right]\\ \\

&= \left(\dfrac{1}{\sigma\sqrt{2\pi}}\right)^{\!N} \exp\!\left[-\dfrac{1}{2}\sum_{i=1}^{N}\left(\dfrac{x_i-\mu}{\sigma}\right)^2\right]

\end{aligned}
```

The Log Likelihood is:

```{math}
:label: eq:pdf_normll

\begin{aligned}

\ln\mathcal{L}(\mu,\sigma;x) &= - N\ln(\sigma) - \dfrac{N}{2}\ln(2\pi) -\dfrac{1}{2}\sum_{i=1}^{N}\left(\dfrac{x_i-\mu}{\sigma}\right)^2

\end{aligned}
```

The Negative Log Likelihood is:

```{math}
:label: eq:pdf_normnll

\begin{aligned}

\mathsf{NLL} = -\ln\mathcal{L}(\mu,\sigma;x) &= N\ln(\sigma) + \dfrac{N}{2}\ln(2\pi) +\dfrac{1}{2}\sum_{i=1}^{N}\left(\dfrac{x_i-\mu}{\sigma}\right)^2

\end{aligned}
```


:::{note}
We are assuming that $\sigma$ is a constant parameter value in [](#eq:pdf_normlik) and [](#eq:pdf_normll). When we look at the Method of Least Squares later [](#sec:lse_chi2), we will make a different assumption and get a different form.
:::

Setting the **Partial Derivatives** $\dfrac{ \partial}{\partial\mu}\ln\mathcal{L} = 0$ and $\dfrac{ \partial}{\partial\sigma}\ln\mathcal{L} = 0$ gives the MLEs:

```{math}
:label: eq:pdf_normmle
\hat{\mu} = \overline{x}, \qquad \hat{\sigma}^2 = \dfrac{1}{N}\sum_{i=1}^{N}(x_i - \overline{x})^2
```

<!--$\textcolor{red}{\mathsf{TODO}}$: give full working $\hat{\mu}_{\mathsf{MLE}}$ and $\hat{\sigma}_{\mathsf{MLE}}$--> 


Note that $\hat{\sigma}^2$ is the **biased** sample variance (divides by $N$, not $N-1$). See [](#chapter:ei) for the distinction between the MLE and the unbiased estimator.

---

<!--TestStatistics.pdf slides 20-26 -->

(sec:pdf_chi2)=
## The Chi Squared ($\chi^2$) Distribution

The Chi Squared ($\chi^2$) distribution arises naturally when working with sums of squared Normal RVs. Recall the Standard Normal RV is $Z = \dfrac{X-\mu}{\sigma}$ [](#eq:norm_z). It most often arises as a **derived distribution** rather than as a primary sampling model. Two key examples of $\chi^2$ RVs in this module are:

The variance of the unbiased sample variance estimate 
: $\dfrac{(N-1)\sigma_x^2}{\sigma^2} \sim \chi^2(N-1)$ [](#eq:ei_sev)

The Pearson goodness-of-fit statistic
: $\displaystyle\sum_i \dfrac{(o_i - e_i)^2}{e_i} \;\dot{\sim}\; \chi^2(\nu)$

  Where $o_i$ and $e_i$ are observed and expected counts, respectively. We will look at this in [](#sec:lse_bin).


:::{note}
I will use $Q$ to indicate the Chi Squared RV, because the usual choice of $X$ looks very much like the symbol for "Chi" $\chi$, which is confusing.
:::

The $\chi^2$ RV, $Q$, is the sum of $\nu$ such squared Standard Normal RVs:

```{math}
:label: eq:pdf_chi2
q = \sum\limits_{i=1}^{\nu} z_i^2
```

where $\nu$ is the **Number of Degrees of Freedom**. Since it is a sum of squares, $Q \sim \chi^2(\nu) \geq 0$ always.


The parameter is $\nu$: a positive integer giving the number of independent squared Standard Normal RVs in the sum.

---

(sec:pdf_chi2gam)=
### PDF & Kernel

<!--f(x;\nu) = \dfrac{x^{\nu/2-1}\, e^{-x/2}}{2^{\nu/2}\,\Gamma(\nu/2)} \;\;;\;\; x \geq 0-->

The $\chi^2$ PDF is:

```{math}
:label: eq:pdf_chi2pdf
f_Q(q;\nu) = \dfrac{2^{-\nu/2} }{\Gamma(\nu/2) } q^{(\nu/2) - 1}\, e^{-q/2}\; ; \;\;\; 0\leq q < \infty

```

The $\Gamma(\nu/2)$ in the denominator is the **Gamma Function** - see [Math Methods B](#sec:mmb_gamfun).

The **Support** of $Q$ is $0 \leq q \leq \infty$.

The **Kernel** of the $\chi^2$ PDF is:

```{math}
:label: eq:pdf_chi2kern
q^{(\nu/2)-1}\,e^{-q/2}
```

A plot of the $\chi^2$ PDF and CDF for a parameter value of $\nu=1$ is shown in [](#fig:pdf_chi2).

:::{figure}
:label: fig:pdf_chi2

![image](../figures/Chi2-StandardNormLongTail.png)

The $\chi^2$ PDF for $\nu=1$. 
:::

When we increase the number of degrees of freedom $\nu$ (the number of standard normal RVs in the sum) the $\chi^2$ PDF completely changes shape, because $Q \sim \chi^2(\nu)$ is the sum over $\nu$ squared-residuals.

:::{figure}
:label: fig:pdf_chi22

![image](../figures/Chi2-nus-1to9.png)

The $\chi^2$ PDF for several values of the degrees of freedom $\nu$. For small $\nu$ the distribution is strongly right-skewed[^skew]. As $\nu$ increases the distribution becomes more symmetric.
:::


Dividing RV by its Expectation, $E[Q] = \nu$, standardises it. As we increase $\nu$, the **Reduced** (standardised) $Q/\nu$ RV distribution approaches Norm($\mu= 1, \sigma= \sqrt{2/\nu}$). The Reduced $\chi^2$ PDFs for a range of different parameter values are shown in [](#fig:pdf_chi2red)


:::{figure}
:label: fig:pdf_chi2red

![image](../figures/Chi2red-nus-10to300.png)

The $\chi^2$ PDF for the "reduced" standardised RV $Q/\nu$ for several values of the degrees of freedom $\nu$. For small $\nu$ the distribution is strongly right-skewed[^skew]. As $\nu$ increases the distribution becomes more symmetric, approaching a Normal distribution as per the [CLT](#sec:norm_clt) — since it is a sum of $\nu$ independent terms.
:::

[^skew]: A right-skewed distribution has its expectation (mean) to the right of the peak (median), so a right-skewed distribution is one with a long tail to the right of the peak.
---

(sec:pdf_chi2prob)=
### Probabilities

The $\chi^2$ CDF has no simple closed form[^rigf] and is evaluated numerically, as per the Normal CDF. In practice we use `scipy.stats.chi2.cdf()`.


[^rigf]: It is related to the **regularised incomplete gamma function**
$F_Q(q;\nu) = P(Q \leq q) = \dfrac{\gamma(\nu/2,\, q/2)}{\Gamma(\nu/2)}$, where $\gamma(s,t) = \int_0^t u^{s-1}e^{-u}\,du$ is the lower incomplete gamma function. 



The $\chi^2$ distribution is frequently used to find **critical values**: for a significance level $\alpha$, the critical value $q_{\alpha}$ satisfies $P(Q > q_{\alpha}) = \alpha$.

---

(sec:pdf_chi2ev)=
### Expectation & Variance

The $Q\sim\chi^2(\nu)$ distribution has:
: $$\label{eq:pdf_expchi2} E[Q] = \nu$$

  $$\label{eq:pdf_varchi2} V[Q] = 2\nu$$

The Reduced $Q/\nu$ distribution has:
: $$\label{eq:pdf_expchi2red} E[Q/\nu] = 1$$

  $$\label{eq:pdf_varchi2red} V[Q/\nu] = \frac{2}{\nu}$$

Please see [Math Methods B: Chi Squared Expectation](#sec:mmb_chisq_expect) if you are interested in the proofs for these.


---

(sec:pdf_chi2py)=
### Python

```{code-cell} python

def ExploreChi2(nu):

  from scipy.stats import chi2

  # set up an instantiation of the distribution with our chosen degrees of freedom
  chisq = chi2(df=nu)

  # access the support
  lb, ub = chisq.support()
  print(f"Support is {lb} <= q <= {ub}")

  # extract the true mean and variance
  mean, var, skew, kurtosis = chisq.stats(moments='mvsk')
  print(f"Expectation = {mean}, Variance = {var}, Skewness = {skew:.2f}, Kurtosis = {kurtosis}")

  # critical value at 5% significance level (upper tail)
  alpha = 0.05
  crit = chisq.ppf(1 - alpha)
  print(f"Critical value at alpha={alpha}: Q > {crit:.3f}")

  # reduced chi2 
  red_chisq = chi2(df=nu, scale=1/nu)
  print(f"Reduced chisq:")

  # access the support
  lb, ub = red_chisq.support()
  print(f"Support is {lb} <= q/nu <= {ub}")

  # extract the true mean and variance
  mean, var, skew, kurtosis = red_chisq.stats(moments='mvsk')
  print(f"Expectation = {mean}, Variance = {var}, Skewness = {skew:.2f}, Kurtosis = {kurtosis}")



ExploreChi2(5)

```

---

<!--
### Likelihoods

The $\chi^2$ distribution most often arises as a **derived distribution** rather than as a primary sampling model. Two key examples in this course are:

- The rescaled sample variance: $\dfrac{(N-1)S^2}{\sigma^2} \sim \chi^2(N-1)$ (see [](#chapter:estimators))
- The Pearson goodness-of-fit statistic: $\displaystyle\sum_i \dfrac{(O_i - E_i)^2}{E_i} \;\dot{\sim}\; \chi^2(\nu)$

When $\chi^2(\nu)$ is used directly as a sampling model for positive-valued data, the Likelihood for $N$ IID observations $\mathbf{x}$ is:

```{math}
:label: eq:pdf_chi2lik
\mathcal{L}(\nu;\mathbf{x}) = \prod_{i=1}^{N} \dfrac{x_i^{\nu/2-1}\, e^{-x_i/2}}{2^{\nu/2}\,\Gamma(\nu/2)}
```

The Log Likelihood is:

```{math}
:label: eq:pdf_chi2ll
\ln\mathcal{L}(\nu;\mathbf{x}) = \left(\dfrac{\nu}{2}-1\right)\!\sum_i \ln x_i - \dfrac{1}{2}\sum_i x_i - \dfrac{N\nu}{2}\ln 2 - N\ln\Gamma\!\left(\dfrac{\nu}{2}\right)
```

The MLE of $\nu$ has no closed form and must be found numerically.
-->
---



(sec:pdf_sumkern)=
## Summary of Kernels

```{table} Uniform, Expon, Norm,  ChiSquared

|  Name | Kernel  | E[X] | V[X] | Support | Parameters  | `scipy.stats` |
| ---   |  ---     | ---  | ---  | ---     |  ---        |  ---          |
| Uniform($a,b$) | $1$ | $\frac{b+a}{2}$ | $\frac{(b-a)^2}{12}$ | $[a,b]$ | $a,b$ | `uniform(loc=a,scale=b-a)` |
| Expon($\lambda$) | $e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ | $[0,\infty)$ | $\lambda$ | `expon(scale=1/λ)` |
| Normal($\mu,\sigma$) | $e^{-x^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | $(-\infty,\infty)$ | $\mu,\sigma$ | `norm(loc=μ,scale=σ)` |
| $\chi^2(\nu)$ | $q^{(\nu/2)-1}e^{-q/2}$ | $\nu$ | $2\nu$ | $[0,\infty)$ | $\nu$ | `chi2(df=nu)` |


```

(sec:pdf_trans)=
## Transforms: Changing Variables

:::{seealso}
I have moved this content - 1D Monotonic Transform, the Gaussian integral trick, and the multivariate Jacobian — to [Math Methods B](#sec:mmb_trans).
:::

<!--
In real life we often cannot measure the RV we want directly. We measure $X$ and want the PDF for $Y = f(X)$. We saw in [](#sec:dd_varfun) that we can access the uncertainties on a function using Expectation Algebra or using Taylor Series expansion.

---

### 1D Monotonic Transform

Suppose $Y = f(X)$ is a **monotonically increasing** ([](#fig:dd_mono)) function of $X$ (no turning points), with inverse $X = g(Y)$.

Because $Y$ increases whenever $X$ increases:

```{math}
P(Y \leq y) = P(X \leq g(y)) = F_X(g(y))
```

Differentiating the CDF of $Y$ to get its PDF:

```{math}
f_Y(y) = \dfrac{d}{dy} F_Y(y) = \dfrac{d}{dy} F_X(g(y)) = \dfrac{dg}{dy}\, f_X(g(y)) = \dfrac{dx}{dy}\, f_X(x)
```

For a **monotonically decreasing** function the same argument applies with a sign flip, so in general:

```{math}
:label: eq:mmb_trans1d
f_Y(y) = \left|\dfrac{dx}{dy}\right| f_X(x)
```

where $x = g(y)$ is the inverse function evaluated at $y$.

:::{note}
If $f(X)$ is **not** monotonic, we can split the domain into intervals on which it is monotonic and sum the contributions. 
:::

Example
: Suppose $X \sim$ Expon($\lambda$) with PDF $f_X(x;\lambda) = \lambda e^{-\lambda x}$ 

  Define a new RV $Y = 2X$: "double the waiting time"

  $x = \dfrac{y}{2}$

  $\dfrac{dx}{dy} = \dfrac{1}{2}$

  $f_Y(y) = \dfrac{1}{2}\, \lambda\, e^{-\lambda \frac{y}{2}} = \dfrac{\lambda}{2}\, e^{-(\lambda/2)\, y}$

  $Y \sim$ Expon($\lambda/2$): doubling the RV (waiting time) is equivalent to halving the rate parameter $\lambda$.

---

(sec:mmb_gastrick)=
### Gaussian Integral Trick

Let $I = \int\limits_{-\infty}^{\infty} e^{-u^2}\,du$. We cannot evaluate $I$ directly, but we can evaluate $I^2$:

```{math}
\begin{aligned}
I^2 &= \left(\int_{-\infty}^{\infty} e^{-x^2}\,dx\right)\!\left(\int_{-\infty}^{\infty} e^{-y^2}\,dy\right)
     = \int_{-\infty}^{\infty}\!\int_{-\infty}^{\infty} e^{-(x^2+y^2)}\,dx\,dy
\end{aligned}
```

Switching to polar coordinates $x = r\cos\theta$, $y = r\sin\theta$, with area element $dx\,dy = r\,dr\,d\theta$:

```{math}
\begin{aligned}
I^2 &= \int_0^{2\pi}\!\int_0^{\infty} e^{-r^2}\, r\,dr\,d\theta
     = 2\pi \int_0^{\infty} r\,e^{-r^2}\,dr
     = 2\pi \left[-\tfrac{1}{2}e^{-r^2}\right]_0^{\infty}
     = \pi
\end{aligned}
```

So $I = \int\limits_{-\infty}^{\infty} e^{-u^2}\,du = \sqrt{\pi}$. 

Substituting $u = \dfrac{(x-\mu)}{\sigma\sqrt{2}}$ 

gives $\dfrac{du}{dx} = \dfrac{1}{\sigma\sqrt{2}}$

We can use this to show that the Normal PDF is valid:

 $\int\limits_{-\infty}^{\infty} \dfrac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)\,dx = \dfrac{1}{\sigma\sqrt{2\pi}} \cdot \sigma\sqrt{2} \cdot I = \dfrac{I}{\sqrt{\pi}} = 1$

 ---
 

### Multivariate Transform and the Jacobian

For a multivariate RV $X = (X_1, X_2, \ldots, X_Z)$ transformed to $Y = (Y_1, Y_2, \ldots, Y_Z)$ via some set of relationships $Y_{(i)} = f(X_{(1)}, \ldots, X_{(Z)})$, the generalisation of [](#eq:mmb_trans1d) involves the **Jacobian**:

```{math}
:label: eq:mmb_transjac
f_{Y}(y) = \dfrac{1}{|J|}\, f_{X}(x)
```

where $J$ is the **determinant** of the $Z\times Z$ matrix of partial derivatives:

```{math}
:label: eq:mmb_jac
J = \begin{vmatrix}
\dfrac{\partial y_1}{\partial x_1} & \cdots & \dfrac{\partial y_1}{\partial x_Z} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial y_Z}{\partial x_1} & \cdots & \dfrac{\partial y_Z}{\partial x_Z}
\end{vmatrix}
```

Example: Cartesian to Polar
: Points $(x,y)$ are uniformly distributed in a plane. We transform to polar coordinates $(r,\theta)$ via $r = \sqrt{x^2+y^2}$, $\theta = \arctan(y/x)$.

  The partial derivatives give:

  $\dfrac{\partial r}{\partial x} = \dfrac{x}{r}, \quad \dfrac{\partial r}{\partial y} = \dfrac{y}{r}, \quad \dfrac{\partial \theta}{\partial x} = -\dfrac{y}{r^2}, \quad \dfrac{\partial \theta}{\partial y} = \dfrac{x}{r^2}$

  The Jacobian determinant is:

  $J = \dfrac{x}{r}\cdot\dfrac{x}{r^2} - \dfrac{y}{r}\cdot\left(-\dfrac{y}{r^2}\right) = \dfrac{x^2+y^2}{r^3} = \dfrac{1}{r}$

  So $f(r,\theta) = r\, g(x,y)$: if $(x,y)$ are uniform (constant $g$), then $f(r,\theta) \propto r$, meaning the density in polar coordinates increases linearly with $r$. This makes geometric sense — concentric rings of equal width have area $\propto r$.
-->
---


