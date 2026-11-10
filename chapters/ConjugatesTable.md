---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: 'Python 3'
numbering:
  title: false
  headings: false
  equation: true
---

(chapter:conjugates)=
# Conjugate Priors

A **conjugate prior** is a prior distribution that, when combined with a given likelihood, produces a posterior of the same distributional family. This is a highly desirable property: the Bayesian update has a closed-form solution, and the posterior parameters have an intuitive interpretation in terms of the data.

## Notation conventions

Throughout this table:

- $N$ = number of observations
- $s = \sum x_i$ = sum of observations (the sufficient statistic for most distributions here)
- $\bar{x} = s/N$ = sample mean
- $x_{(N)} = \max(x_i)$ = sample maximum
- **Gamma** distributions use the **shape–rate** parameterisation: $\text{Gamma}(\alpha, \beta)$ has mean $\alpha/\beta$ and kernel $\lambda^{\alpha-1}e^{-\beta\lambda}$.
- Posterior parameters $\alpha', \beta', \ldots$ are defined in the Posterior Params column and carry over to the Predictive Params column.
- The likelihood kernel is the part of the likelihood that depends on the unknown parameter, with all constants suppressed.

---

## The Table

| Likelihood | Likelihood kernel | Conjugate prior | Prior kernel | Prior params | Posterior | Posterior params | Predictive posterior | Predictive params |
|---|---|---|---|---|---|---|---|---|
| **Bernoulli** | $\theta^x(1-\theta)^{1-x}$ | Beta | $\theta^{\alpha-1}(1-\theta)^{\beta-1}$ | $\alpha,\beta>0$ | Beta | $\alpha'=\alpha+s$ $\beta'=\beta+N-s$ | Beta-Bernoulli | $\alpha',\beta'$; $\;\text{P}(\tilde{x}=1)=\dfrac{\alpha'}{\alpha'+\beta'}$ |
| **Binomial** (fixed $n$) | $\theta^k(1-\theta)^{n-k}$ | Beta | $\theta^{\alpha-1}(1-\theta)^{\beta-1}$ | $\alpha,\beta>0$ | Beta | $\alpha'=\alpha+s$ $\beta'=\beta+Nn-s$ | Beta-Binomial | $n,\;\alpha',\;\beta'$ |
| **Negative Binomial** (fixed $r$) | $\theta^r(1-\theta)^x$ | Beta | $\theta^{\alpha-1}(1-\theta)^{\beta-1}$ | $\alpha,\beta>0$ | Beta | $\alpha'=\alpha+Nr$ $\beta'=\beta+s$ | Beta-Negative Binomial | $r,\;\alpha',\;\beta'$ |
| **Multinomial** (fixed $n$) | $\displaystyle\prod_k \theta_k^{x_k}$ | Dirichlet | $\displaystyle\prod_k \theta_k^{\alpha_k-1}$ | $\alpha_k>0\;\forall k$ | Dirichlet | $\alpha_k'=\alpha_k+s_k$† | Dirichlet-Multinomial | $n,\;(\alpha_1',\ldots,\alpha_K')$ |
| **Poisson** | $e^{-\lambda}\lambda^x$ | Gamma | $\lambda^{\alpha-1}e^{-\beta\lambda}$ | $\alpha,\beta>0$ | Gamma | $\alpha'=\alpha+s$ $\beta'=\beta+N$ | Negative Binomial | $r=\alpha',\;p=\dfrac{\beta'}{\beta'+1}$ |
| **Uniform(0,θ)** | $\theta^{-1}\mathbf{1}(x\leq\theta)$ | Pareto | $\theta^{-(\alpha+1)}\mathbf{1}(\theta\geq\beta)$ | $\alpha,\beta>0$ | Pareto | $\alpha'=\alpha+N$ $\beta'=\max(\beta,\,x_{(N)})$ | Piecewise‡ | — |
| **Exponential** | $\lambda e^{-\lambda x}$ | Gamma | $\lambda^{\alpha-1}e^{-\beta\lambda}$ | $\alpha,\beta>0$ | Gamma | $\alpha'=\alpha+N$ $\beta'=\beta+s$ | Lomax (Pareto II) | shape $\alpha'$, scale $\beta'$ $f(\tilde{x})=\dfrac{\alpha'}{\beta'}\!\left(1+\dfrac{\tilde{x}}{\beta'}\right)^{\!-(\alpha'+1)}$ |
| **Normal** (unknown $\mu$, known $\sigma^2$) | $\exp\!\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)$ | Normal | $\exp\!\left(-\dfrac{(\mu-\mu_0)^2}{2\sigma_0^2}\right)$ | $\mu_0\in\mathbb{R},\;\sigma_0^2>0$ | Normal | $\mu_N,\;\sigma_N^2$§ | Normal | mean $\mu_N$, variance $\sigma^2+\sigma_N^2$ |
| **Normal** (known $\mu$, unknown $\tau=1/\sigma^2$) | $\tau^{1/2}\exp\!\left(-\dfrac{\tau(x-\mu)^2}{2}\right)$ | Gamma | $\tau^{\alpha-1}e^{-\beta\tau}$ | $\alpha,\beta>0$ | Gamma | $\alpha'=\alpha+\dfrac{N}{2}$ $\beta'=\beta+\dfrac{1}{2}\sum(x_i-\mu)^2$ | Student's $t$ | $\nu=2\alpha'$, location $\mu$, scale $\sqrt{\beta'/\alpha'}$ |

---

## Notes

**† Multinomial**: $s_k = \sum_i x_{ik}$ is the total count of category $k$ across all $N$ observations. The Dirichlet posterior with parameters $(\alpha_1',\ldots,\alpha_K')$ has marginal means $\text{E}[\theta_k] = \alpha_k'/\sum_j \alpha_j'$, which are the natural posterior estimates of each category's probability.

**‡ Uniform predictive**: the predictive posterior is piecewise with no standard name.

$$
f(\tilde{x}\,|\,\text{data}) = \begin{cases} \dfrac{\alpha'}{(\alpha'+1)\,\beta'} & 0 \leq \tilde{x} \leq \beta' \\[8pt] \dfrac{\alpha'\,\beta'^{\,\alpha'}}{(\alpha'+1)}\,\tilde{x}^{-(\alpha'+1)} & \tilde{x} > \beta' \end{cases}
$$

The first piece is uniform (flat); the second is a Pareto-like power-law tail accounting for the possibility that the new observation exceeds the largest value seen so far.

**§ Normal (unknown mean) posterior parameters**:

$$
\frac{1}{\sigma_N^2} = \frac{1}{\sigma_0^2} + \frac{N}{\sigma^2}, \qquad \mu_N = \sigma_N^2\!\left(\frac{\mu_0}{\sigma_0^2} + \frac{N\bar{x}}{\sigma^2}\right)
$$

The posterior precision (reciprocal variance) is the sum of the prior precision and the data precision. The posterior mean is the precision-weighted average of the prior mean and the sample mean.

---

## Pattern recognition

Looking across the table, a few structural patterns emerge:

**Exponential family likelihoods** (Bernoulli, Binomial, Poisson, Exponential, Normal) all have conjugate priors in the same exponential family. The posterior update is always additive: the hyperparameters just accumulate the sufficient statistics of the data.

**The Beta prior** serves three different likelihoods (Bernoulli, Binomial, Negative Binomial) because all three share the same kernel structure $\theta^a(1-\theta)^b$. They differ only in what $a$ and $b$ count.

**The Gamma prior** serves three likelihoods (Poisson, Exponential, Normal precision), again because the kernel $\lambda^a e^{-b\lambda}$ matches all three.

**Predictive posteriors** are generally heavier-tailed than the likelihood, because they must account for uncertainty in the parameter as well as sampling variability. This is why the Normal-precision predictive is Student's $t$ rather than Normal: integrating out the uncertain precision inflates the tails.

**The Dirichlet–Multinomial** is the natural generalisation of the Beta–Binomial to $K>2$ categories, and the Dirichlet is the natural generalisation of the Beta to probability vectors that must sum to 1.
