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

(chapter:mcmc)=
# Predicting the Future

<!--
This chapter covers a powerful technique for working with uncertainty when analytical solutions are unavailable: **Markov Chain Monte Carlo** (sampling from complex posterior distributions). We then walk through the workshop Python file `PoissonGamma-v2.py`, which implements Metropolis-Hastings MCMC for the Poisson-Gamma model introduced in the [previous chapter](#chapter:bay).

Key Python tools used in this chapter:
* `scipy.stats`: `norm`, `gamma`, `poisson`, `bootstrap`
* `numpy`: `random.choice`
-->

---

---

(sec:bay_predpost)=
## Predictive Posterior



In the [previous chapter](#chapter:bay) we derived the conjugate models analytically for Bernoulli-Beta, Binomial-Beta, Poisson-Gamma, and Exponential-Gamma, and Normal-Normal. This allowed use to generate Posterior Probability distributions on the parameters, which gave us the Maximum A Posteriori (MAP) estimates and credible intervals.


We can continue analytically to find the **Predictive Posterior** distributions in some cases. These are not distributions on the parameter space but on the sample space: they are **predictive distributions of data not yet seen**.

The Predictive distribution is recovered by integrating out the parameter uncertainty, as illustrated in [](#fig:mcmc_pred).
<!--
```{math}
P(\tilde{y}|y) = \int f(\tilde{y}|\theta)\,P(\theta|y)\,d\theta = \text{NBinom}\!\left(\alpha + n\bar{y},\; \frac{\beta+n}{\beta+n+1}\right)
```
-->


:::{figure}
:label: fig:mcmc_pred
![](/figures/predD.png)
:::


The Predictive Posterior distributions and parameters for the most common data distributions are given in [](#tab:conjugate_table).


:::{table}
:label: tab:conjugate_table

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

:::

<!--Recall:

```{math}
:label: eq:poisson_gamma_recap
\text{Prior: } \pi(\theta) &\propto \theta^{\alpha-1} e^{-\beta\theta} \\
\text{Likelihood: } \mathcal{L}(y|\theta) &\propto \theta^{n\bar{y}} e^{-n\theta} \\
\text{Posterior: } P(\theta|y) &= \text{Gamma}(\alpha + n\bar{y},\; \beta + n)
```
-->




<!--
We also found the **Predictive Posterior** — integrating out $\theta$ to get the distribution over new data $\tilde{y}$:

$$P(\tilde{y}|y) = \int f(\tilde{y}|\theta)\,P(\theta|y)\,d\theta = \text{NBinom}\!\left(\alpha + n\bar{y},\; \frac{\beta+n}{\beta+n+1}\right)$$

This was possible because Poisson-Gamma is a **conjugate pair** with a known closed-form solution.
-->






So far, the Posterior gives us a distribution over the **parameter** $\theta$. The **Predictive Posterior** instead gives us a distribution over a **future observation** $\tilde{y}$, integrating out the parameter uncertainty:

```{math}
:label: eq:bay_pred

P(\tilde{y} \mid y) = \int_\Theta f(\tilde{y} \mid \theta)\; P(\theta \mid y)\, d\theta
```

For the Poisson-Gamma case, this integral can be computed analytically. The result is a **Negative Binomial** distribution:

```{math}
:label: eq:bay_nbinpred

P(\tilde{y} \mid y) = \text{NBinom}\!\left(\alpha + n\bar{y},\;\frac{\beta+n}{\beta+n+1}\right)
```

The Negative Binomial models the number of failures before a fixed number of successes. As a predictive distribution it has a key property: unlike the Posterior (which shrinks to a spike at $\hat{\theta}$ as $n \to \infty$), the Predictive Posterior maintains the same shape as the data distribution, because it integrates over parameter uncertainty.

The Poisson-Gamma chain in full:

$$\underbrace{\mathsf{Gamma}(\alpha, \beta)}_{\text{Prior}} \xrightarrow{\text{Poisson data}} \underbrace{\mathsf{Gamma}(\alpha+n\bar{y},\; \beta+n)}_{\text{Posterior}} \xrightarrow{\text{integrate out }\theta} \underbrace{\text{NBinom}\!\left(\alpha+n\bar{y},\; \frac{\beta+n}{\beta+n+1}\right)}_{\text{Predictive Posterior}}$$


:::{figure}
:label: fig:mcmc_bbanim
![](../figures/hnBB-a2-b8-panels3.gif)

Caption to be added 
:::

In general, the posterior integral $\int \mathcal{L}(y|\theta)\,\pi(\theta)\,d\theta$ cannot be solved analytically. We can approximate the posterior by **sampling** from it using a Markov Chain.


---

(sec:mcmc_mcmc)=
## Markov Chain Monte Carlo (MCMC)

<!--
## 

### The Idea
-->
Rather than integrating analytically, MCMC builds up an **empirical sampling distribution** that approximates the posterior. We:

1. Start at some value of the parameter $\theta$.
2. Take a random step to a new proposed value.
3. Decide whether to accept the new value or stay at the old one.
4. Repeat — the sequence of accepted values forms a **Markov Chain**.

The chain is **memoryless**: each decision depends only on the current value, not on how we got there.

The key insight is that we only ever need the **ratio** of posteriors between two proposed values — the normalising constant (evidence) cancels out:

```{math}
:label: eq:mh_ratio
p_a = \frac{P(\theta_1|y)}{P(\theta_0|y)} = \frac{\mathcal{L}(y|\theta_1)\,\pi(\theta_1)}{\mathcal{L}(y|\theta_0)\,\pi(\theta_0)}
```

### The Metropolis-Hastings Algorithm

One **Metropolis-Hastings** iteration:

1. **Start**: current value $\theta_0$ (initial guess).
2. **Propose**: add random noise to get a candidate $\theta_1 \sim \mathcal{N}(\theta_0, \sigma_\text{step})$.
3. **Compare**: compute the posterior ratio $p_a = P(\theta_1|y)/P(\theta_0|y)$.
   - If $p_a > 1$: **accept** $\theta_1$ (it's better).
   - If $p_a > U(0,1)$: **randomly accept** $\theta_1$ (it's worse but not by much — this is what allows the chain to explore).
   - Otherwise: **reject**, keep $\theta_0$.
4. **Repeat**: the winner becomes the starting point for the next iteration.

After many iterations we have a sample from the posterior, from which we can compute MAP estimates, credible intervals, and predictive posteriors.

### Python Implementation (from `PoissonGamma-v2.py`)

The sampler function takes the data, a prior configuration, and hyperparameters:

```python
from scipy.stats import norm, gamma, poisson

def sampler(data, prior_config, hp_config):

    # --- Starting value ---
    a0 = hp_config.guess
    if a0 == -1:
        a0 = np.random.choice(data, size=1)[0]
    if hp_config.non_neg:
        a0 = abs(a0)

    samples = []

    for i in range(hp_config.N):

        # --- Propose new value ---
        a1 = norm(a0, hp_config.stepsize).rvs()
        if hp_config.non_neg:
            a1 = abs(a1)

        # --- Log-likelihoods ---
        d0  = poisson(mu=a0)
        LL0 = np.sum(d0.logpmf(data))

        d1  = poisson(mu=a1)
        LL1 = np.sum(d1.logpmf(data))

        # --- Prior PDFs ---
        prior  = gamma(a=prior_config.shape, scale=prior_config.scale)
        prior0 = prior.pdf(a0)
        prior1 = prior.pdf(a1)

        # --- Posterior ratio ---
        post0 = LL0 * prior0
        post1 = LL1 * prior1
        p_a   = post1 / post0

        # --- Decision ---
        if p_a > 1:                          # accept
            val1 = a1
        elif p_a > np.random.rand():         # random accept
            val1 = a1
        else:                                # reject
            val1 = a0

        # --- Burn-in and thinning ---
        if i >= int(hp_config.burnin * hp_config.N):
            if i % hp_config.thin == 0:
                samples.append(val1)

        a0 = val1   # winner becomes next starting point

    return samples
```

Note that **log-likelihoods** are used (`logpmf`) rather than likelihoods to avoid numerical underflow with large datasets, and the ratio is computed in log space or as shown above.

### Sampler Hyperparameters

The key settings that control the MCMC run:

```python
s_N        = 5_000    # Number of samples (more is better, but slower)
s_guess    = -1       # Starting value (-1 = random draw from data range)
s_stepsize = 4.0      # Step size: std of the Normal proposal distribution
s_burnin   = 0.01     # Fraction of samples to discard as burn-in
s_thin     = 1        # Keep every s_thin-th sample (1 = keep all)
s_non_neg  = True     # Enforce non-negative parameter values
```

**Burn-in**: The first few samples in a chain may be unreliable because the chain starts at an arbitrary point. Discarding the initial `s_burnin` fraction removes this warm-up period.

**Thinning**: If `s_thin > 1`, only every $k$-th sample is kept. This reduces the autocorrelation between consecutive samples, at the cost of fewer samples.

---

## Gibbs Sampling

**Gibbs sampling** extends Metropolis-Hastings to multiple parameters. Instead of proposing all parameters simultaneously, it updates **one parameter at a time**, conditioning on the current values of all others.

For two parameters $a$ and $b$:

1. Start with initial guesses $(a_0, b_0)$.
2. Propose $a_1 \sim \mathcal{N}(a_0, \sigma_a)$.
3. Compare posteriors **given** $b_0$; accept or reject $a_1$ as in MH.
4. Propose $b_1 \sim \mathcal{N}(b_0, \sigma_b)$.
5. Compare posteriors **given** $a_1$; accept or reject $b_1$.
6. Repeat from step 2.

Gibbs sampling is correct but can be **very slow** when parameters are correlated, because the chain zigzags through the posterior rather than moving efficiently along the main axis of correlation.

---

## Differential Evolution

**Differential Evolution (DE)** addresses the inefficiency of Gibbs sampling by running **multiple chains simultaneously** and using the information across chains to generate proposals.

For each chain $i$:

1. **Start**: initial guess $\theta_i^{(0)}$ (one guess per chain, usually spread across the parameter space).
2. **Propose**: randomly select two other chains $j$ and $k$; the proposal is

   $$\theta_i^* = \theta_i^{(t)} + \gamma\,(\theta_j^{(t)} - \theta_k^{(t)}) + \epsilon$$

   where $\gamma$ is a tuning scale factor and $\epsilon$ is a small random noise (to avoid degeneracy).
3. **Accept/Reject**: as in MH.

DE has only **two tuning parameters** (the scale $\gamma$ and the noise $\epsilon$), regardless of the number of model parameters — a significant advantage over standard MH which needs a separate step size for each parameter. The multiple chains automatically adapt to the shape of the posterior.

---

## Probabilistic Programming Libraries

In practice, most MCMC analyses use dedicated libraries rather than hand-coded samplers. Two popular choices:

**PyMC** ([docs](https://www.pymc.io/welcome.html)) is the most widely used probabilistic programming library in Python. It uses Differential Evolution and other modern samplers. The documentation can be dense, but the [Bayesian Methods for Hackers](https://nbviewer.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC2.ipynb) notebook is an excellent introduction.

**emcee** ([docs](https://emcee.readthedocs.io/en/stable/tutorials/quickstart/)) is a simpler library popular in astronomy. It is easy to get started with, but does not support predictive posteriors out of the box.

---

## Workshop: `PoissonGamma-v2.py`

The workshop file `PoissonGamma-v2.py` implements the full Metropolis-Hastings MCMC pipeline for the Poisson-Gamma model. The key sections of the file are:

### Data and Prior Setup

```python
# Data: drawn from Poisson(data_shape)
data_shape = 4       # true Poisson rate mu
data_N     = 50      # number of data points

# Prior: Gamma(prior_shape, prior_scale)  [scale convention, not rate]
prior_shape = 5.     # Gamma alpha hyperparameter
prior_scale = 1.0    # Gamma scale hyperparameter (scale = 1/rate)
```

**Important**: `PoissonGamma-v2.py` uses the **scale** convention for the Gamma distribution (`scale = 1/rate`), consistent with `scipy.stats.gamma`. This differs from the lecture notes where $\beta$ is the **rate** (and `scipy` is called with `scale=1/beta`). Always check which convention a file uses.

### Running the Sampler

The sampler is called as:

```python
samples, sval = sampler(data, prior_config, hp1)
```

where `sval` is the starting value used (for diagnostics). The returned `samples` list contains the accepted parameter values, already filtered by burn-in and thinning.

### Diagnostics

The file produces three standard MCMC diagnostics:

**Trace plot**: the accepted parameter value at each step. A good trace should look like white noise — exploring freely around the true value. A trace that moves too slowly (step size too small) or that jumps erratically (step size too large) indicates poor mixing.

**Autocorrelation function (ACF)**: measures correlation between samples separated by different lags. Good MCMC samples should have low autocorrelation. High autocorrelation suggests the chain needs more thinning or a larger step size.

**Acceptance fraction**: the proportion of proposed steps that were accepted. A very high acceptance fraction (near 1) means the step size is too small (the sampler barely moves). A very low fraction means the step size is too large (most proposals are rejected). A target of roughly 20–50% is typical.

### Comparing MCMC to the Analytical Result

Since we know the analytical posterior for Poisson-Gamma is $\text{Gamma}(\alpha + n\bar{y},\; \beta + n)$, we can **validate** the MCMC output by overlaying:

* The **MCMC histogram** of sampled $\theta$ values
* The **analytical posterior** Gamma PDF
* The **predictive posterior** NBinom PMF vs the MCMC-based predictive

When properly tuned, the MCMC histogram should closely match the analytical posterior — confirming the sampler is working correctly.

---

<!--
## Summary

| Technique | When to Use | Key Idea |
|---|---|---|
| Non-parametric bootstrap | Uncertainty on any statistic; distribution unknown | Resample with replacement, compute statistic many times |
| Parametric bootstrap | Uncertainty on statistic; distribution known | Resample from fitted MLE distribution |
| Smoothed bootstrap | Inflated uncertainty (conservative) | Add noise to each bootstrap resample |
| Metropolis-Hastings | Single parameter, posterior known up to normalisation | Propose, accept/reject via posterior ratio |
| Gibbs sampling | Multiple parameters, can sample each conditional | MH one parameter at a time |
| Differential Evolution | Multiple correlated parameters | Multiple chains; proposals informed by inter-chain differences |

The bootstrap is a **frequentist** tool: it estimates sampling variability of a statistic under repeated experiments. MCMC is a **Bayesian** tool: it explores the posterior distribution of parameters given the data. Both are indispensable for modern data analysis.
00-->