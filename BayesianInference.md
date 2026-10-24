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

(chapter:bayesian)=
# Bayesian Inference

Bayesian Inference gives us a mathematically coherent framework for updating our beliefs about a parameter $\theta$ using data $y$. Rather than treating $\theta$ as a fixed but unknown constant (the Frequentist view), the Bayesian approach treats $\theta$ as a random variable with its own probability distribution, and updates that distribution as data arrive.

Key Python tools used in this chapter:
* `scipy.stats`: `beta`, `gamma`, `nbinom`, `poisson`
* `scipy.special`: `beta as betafunc`
* Workshop file: `Week10-BayesUpdating.py`

---

## Bayes' Theorem

Bayes' Theorem relates the **Posterior** probability of a parameter given data to the **Likelihood** of the data given the parameter, the **Prior** probability of the parameter, and the **Evidence**:

```{math}
:label: eq:bayes

P(\theta \mid y) = \frac{\mathcal{L}(y \mid \theta)\; \pi(\theta)}{\int \mathcal{L}(y \mid \theta)\, \pi(\theta)\, d\theta}
```

Each element has a precise name and role:

| Term | Symbol | Meaning |
|---|---|---|
| **Posterior** | $P(\theta \mid y)$ | Our updated belief about $\theta$ after seeing data $y$ |
| **Likelihood** | $\mathcal{L}(y \mid \theta)$ | The probability of observing data $y$ given $\theta$ |
| **Prior** | $\pi(\theta)$ | Our belief about $\theta$ *before* seeing data |
| **Evidence** | $\int \mathcal{L}\, \pi\, d\theta$ | A normalisation constant ensuring $\int P(\theta \mid y)\, d\theta = 1$ |

The Prior is a PDF (or PMF for discrete $\theta$), not a single number.

---

## The COVID Test Example

### Setup

A COVID test has the following known properties (from the manufacturer):

| Hypothesis | Negative | Positive |
|---|---|---|
| **Sick** ($\theta_+$) | $p = 0$ | $p = 1$ |
| **Healthy** ($\theta_-$) | $p = 0.95$ | $p = 0.05$ |

Data $y$ are **Bernoulli**: a binary outcome from a single test ($y=1$ positive, $y=0$ negative).

The prior probability of being sick at any given time is $\pi(\theta_+) = 1/1000$.

### Likelihoods

The Bernoulli PMF is $p_Y(y \mid \theta) = \theta^y (1-\theta)^{1-y}$, and for $N=1$ the Likelihood equals the PMF.

For the **sick** hypothesis ($\theta_+ = 1$):

$$\mathcal{L}(y_+ \mid \theta_+) = 1^1 \cdot 0^0 = 1 \qquad \mathcal{L}(y_- \mid \theta_+) = 0$$

For the **healthy** hypothesis ($\theta_- = 0.05$):

$$\mathcal{L}(y_+ \mid \theta_-) = 0.05^1 \cdot 0.95^0 = 0.05 \qquad \mathcal{L}(y_- \mid \theta_-) = 0.95$$

### Posterior Calculation

Applying Bayes' Theorem for a positive test result $y_+$:

```{math}
:label: eq:covid_posterior

P(\theta_+ \mid y_+) = \frac{\mathcal{L}(y_+ \mid \theta_+)\; \pi(\theta_+)}{\mathcal{L}(y_+ \mid \theta_+)\pi(\theta_+) + \mathcal{L}(y_+ \mid \theta_-)\pi(\theta_-)}
= \frac{1 \times 0.001}{1 \times 0.001 + 0.05 \times 0.999} \approx 0.0196
```

The full posterior table:

| $\theta$ | Likelihood $\mathcal{L}(y_+ \mid \theta)$ | Prior $\pi(\theta)$ | Numerator | Evidence | Posterior $P(\theta \mid y_+)$ |
|---|---|---|---|---|---|
| $\theta_+$ | 1 | 0.001 | 0.001 | 0.05095 | **0.0196** |
| $\theta_-$ | 0.05 | 0.999 | 0.04995 | 0.05095 | **0.9804** |

Despite a perfect sensitivity (sick people always test positive), the probability of actually being sick given a positive test is only $\approx 2\%$. This is because the disease is rare, so most positives come from the false positive rate applied to the large healthy population.

### Bayesian Updating

What we did above is called **Bayesian Updating**:

- **Before** the test, we held a Prior $\pi(\theta_+) = 0.001$
- **After** the test, we calculated a Posterior $P(\theta_+ \mid y_+) \approx 0.02$

The data have updated our belief. With more data (more tests), the posterior would narrow further.

### The Evidence as a Constant

The Evidence (denominator) is just a constant factor $c$ identical for all hypotheses:

$$P(\theta_+ \mid y_+) = \mathcal{L}(y_+ \mid \theta_+)\; \pi(\theta_+) \times c$$

We can compute all numerators first, then normalise by their sum:

- $\mathcal{L}(y_+ \mid \theta_+)\pi(\theta_+) = 0.001$
- $\mathcal{L}(y_+ \mid \theta_-)\pi(\theta_-) = 0.04995$
- Sum $= 0.05095$, so $c = 1/0.05095 \approx 19.627$

This shortcut is useful when computing posteriors for many hypotheses.

---

## From Discrete to Continuous Priors

In the example above, we assigned a single number $\pi(\theta_+) = 1/1000$ as our prior. This implies infinite precision — zero uncertainty about the prevalence rate. A more honest treatment uses a **prior PDF** over the parameter.

For a parameter that lives in $[0, 1]$ (like a probability or prevalence), the natural conjugate prior is the **Beta distribution**.

---

## The Beta Distribution

(bayesian:beta)=

The Beta distribution is a flexible PDF for parameters constrained to $(0, 1)$:

```{math}
:label: eq:beta_pdf

f_\theta(\theta) = \frac{1}{B(\alpha,\beta)}\, \theta^{\alpha-1} (1-\theta)^{\beta-1}
\qquad 0 \leq \theta \leq 1, \quad \alpha > 0,\; \beta > 0
```

where $B(\alpha, \beta)$ is the Beta function (a normalisation constant).

Its moments are:

```{math}
:label: eq:beta_moments

E[\theta] = \frac{\alpha}{\alpha+\beta} \qquad V[\theta] = \frac{E[\theta]\,(1-E[\theta])}{\alpha+\beta+1}
```

The two **shape hyper-parameters** $\alpha$ and $\beta$ allow it to take on a huge range of shapes: uniform, U-shaped, bell-shaped, skewed left or right. In Python:

```python
from scipy.stats import beta
from scipy.special import beta as betafunc   # careful: same name!

# PDF evaluated at theta=0.3, with shape params alpha=2, beta=5
beta.pdf(0.3, a=2, b=5)

# Normalisation constant
betafunc(2, 5)
```

---

## Bernoulli-Beta Conjugacy

(bayesian:conjugacy_bb)=

A key computational convenience arises when the Prior and the Posterior belong to the same distribution family. This is called **conjugacy**, and the prior is called the **conjugate prior** for that likelihood.

**Beta is conjugate for Bernoulli (and Binomial) likelihoods.**

Starting with a Beta prior and a Bernoulli likelihood:

```{math}
:label: eq:bb_posterior

\begin{align*}
P(\theta \mid y) &\propto \mathcal{L}(y \mid \theta)\; \pi(\theta) \\
&\propto \theta^y (1-\theta)^{1-y} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1} \\
&\propto \theta^{(\alpha+y)-1}(1-\theta)^{(\beta+1-y)-1}
\end{align*}
```

The Posterior is $\text{Beta}(\alpha', \beta')$ with updated hyper-parameters:

$$\alpha' = \alpha + y \qquad \beta' = \beta + (1-y)$$

A **positive** result ($y=1$) increments $\alpha$; a **negative** result ($y=0$) increments $\beta$. Each new data point nudges the posterior in a transparent, interpretable way.

For **Binomial** data with $n$ trials and $y$ successes:

$$P(\theta \mid y) \propto \text{Beta}(\alpha+y,\; \beta+n-y)$$

---

## Maximum A Posteriori Estimation

The **MAP estimate** $\hat{\theta}_{MAP}$ is the mode of the Posterior — the value of $\theta$ that maximises $P(\theta \mid y)$:

$$\hat{\theta}_{MAP} = \arg\max_\theta \; P(\theta \mid y) = \arg\max_\theta \; \mathcal{L}(y \mid \theta)\; \pi(\theta)$$

For the Beta distribution, $\text{mode} = (\alpha'-1)/(\alpha'+\beta'-2)$.

If the Prior is **uniform** (uninformative), then $\hat{\theta}_{MAP}$ coincides with the **MLE** $\hat{\theta}_{MLE}$.

The key advantage of MAP over MLE is that it incorporates prior beliefs — especially valuable with small datasets. In the limit of large $n$, MAP $\rightarrow$ MLE regardless of the prior chosen.

---

## Credible Intervals

A **Credible Interval** is a range of posterior values that contains the true parameter with a specified probability (e.g. 95%). This is the intuitive statement that many people incorrectly associate with Frequentist confidence intervals.

> A **95% Credible Interval** $[a, b]$ means: given the data and the prior, there is a 95% probability that $\theta \in [a, b]$.

In Python, use the PPF (percent point function) of the posterior distribution:

```python
from scipy.stats import gamma

ci = 0.95
lower = gamma.ppf(0.5*(1-ci), a_post, 0, 1/b_post)
upper = gamma.ppf(1 - 0.5*(1-ci), a_post, 0, 1/b_post)
print(f"MAP: {map_est:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")
```

> A **95% Confidence Interval** (Frequentist) means: if you repeated the experiment 100 times, 95 of the resulting intervals would contain the true value. The true value is fixed; the interval is random.

---

## Notes on Priors

**Why do we need a Prior?**
A Frequentist does not use a Prior and writes $P(\theta \mid y) = \mathcal{L}(y \mid \theta)$. The Bayesian counter-argument is that sampling distributions are also a form of subjective prior knowledge — we still have to choose a distribution family for the likelihood.

**Why use a conjugate Prior?**
Conjugate priors produce a posterior in closed form, making the analysis transparent. Beta is the conjugate prior for probability-valued parameters. Gamma is conjugate for rate/count parameters.

**Uninformative Priors**
If you have no prior knowledge, use a uniform (or weakly informative) prior. A uniform prior on the real line integrates to infinity — an **Improper Prior** — but this is acceptable as long as the resulting Posterior is proper (integrates to a finite value). See [Jeffreys Prior](https://en.wikipedia.org/wiki/Jeffreys_prior) for a principled way to choose uninformative priors. Note: a prior uniform in $X$ is not uniform in a transform of $X$.

**Avoid hard boundaries**
If your parameter must be physically positive but theoretically unrestricted, a uniform prior with a hard boundary at zero causes problems near the boundary. Prefer a soft constraint — a distribution that naturally decays near zero (e.g. a half-normal or exponential prior).

**Advice**: Always plot the prior distribution before proceeding. It is easy to accidentally choose one with reasonable moments but a pathological shape.

---

## Conjugate Priors: Summary Table

### Beta (for probability parameters, $0 \leq \theta \leq 1$)

```python
from scipy.stats import beta
# beta.pdf(theta, a=alpha, b=beta_param)
```

| Likelihood | Posterior |
|---|---|
| Bernoulli: $\theta^y(1-\theta)^{1-y}$ | $\text{Beta}(\alpha+y,\; \beta+1-y)$ |
| Binomial: $\binom{n}{y}\theta^y(1-\theta)^{n-y}$ | $\text{Beta}(\alpha+y,\; \beta+n-y)$ |

### Gamma (for rate/count parameters, $0 \leq \theta < \infty$)

```python
from scipy.stats import gamma
# gamma.pdf(theta, a=alpha, scale=1/beta_rate)
```

The Gamma PDF is:

```{math}
:label: eq:gamma_pdf

f(\theta) = \frac{\beta^\alpha}{\Gamma(\alpha)}\, \theta^{\alpha-1} e^{-\beta\theta}
```

with $E[\theta] = \alpha\beta$ and $V[\theta] = \alpha\beta^2$ (using the scale convention, where $\beta$ is the scale parameter). Note: in `scipy.stats.gamma`, pass `scale=1/beta_rate` when $\beta$ is a *rate*.

| Likelihood | Posterior |
|---|---|
| Poisson: $e^{-\theta}\theta^y/y!$ | $\text{Gamma}(\alpha+n\bar{y},\; \beta+n)$ |
| Exponential: $\theta e^{-\theta y}$ | $\text{Gamma}(\alpha+1,\; \beta+n\bar{y})$ |

---

## Poisson-Gamma Example

(bayesian:poisson_gamma)=

Suppose we observe Poisson-distributed counts (e.g. particle decays, bus arrivals) with unknown rate $\theta$. We place a Gamma prior on $\theta$.

**Choosing hyper-parameters**: If our best guess is $\theta = 15$, we want $E[\theta] = \alpha\beta = 15$ and $V[\theta] = \alpha\beta^2 = 15$ (matching Poisson's equal mean and variance). This gives $\beta = 1$, $\alpha = 15$. So we use $\pi(\theta) = \text{Gamma}(15, 1)$.

**Posterior update**: After $n$ data points with sample mean $\bar{y}$:

$$P(\theta \mid y) = \text{Gamma}(\alpha + n\bar{y},\; \beta + n)$$

The **MAP estimate** (mode of Gamma) is:

$$\hat{\theta}_{MAP} = \frac{\alpha' - 1}{\beta'} = \frac{(\alpha + n\bar{y}) - 1}{\beta + n}$$

As $n$ grows, the posterior narrows around the true value and the choice of prior matters less. Starting from a poor prior (e.g. $\text{Gamma}(1,1)$) gives a slower initial convergence, but the data eventually override the prior.

In Python (from `Week10-BayesUpdating.py`):

```python
from scipy.stats import gamma, poisson

# Prior hyper-parameters
a, b = 15, 1

# After seeing n data points with mean mean_y
a_post = a + n * mean_y
b_post = b + n

# MAP estimate
map_est = (a_post - 1) / b_post

# 95% credible interval
ci = 0.95
lower = gamma.ppf(0.5*(1-ci), a_post, 0, 1/b_post)
upper = gamma.ppf(1 - 0.5*(1-ci), a_post, 0, 1/b_post)
```

---

## Predictive Posterior

So far, the Posterior gives us a distribution over the **parameter** $\theta$. The **Predictive Posterior** instead gives us a distribution over a **future observation** $\tilde{y}$, integrating out the parameter uncertainty:

```{math}
:label: eq:predictive

P(\tilde{y} \mid y) = \int_\Theta f(\tilde{y} \mid \theta)\; P(\theta \mid y)\, d\theta
```

For the Poisson-Gamma case, this integral can be computed analytically. The result is a **Negative Binomial** distribution:

```{math}
:label: eq:nbinom_pred

P(\tilde{y} \mid y) = \text{NBinom}\!\left(\alpha + n\bar{y},\;\frac{\beta+n}{\beta+n+1}\right)
```

The Negative Binomial models the number of failures before a fixed number of successes. As a predictive distribution it has a key property: unlike the Posterior (which shrinks to a spike at $\hat{\theta}$ as $n \to \infty$), the Predictive Posterior maintains the same shape as the data distribution, because it integrates over parameter uncertainty.

In Python:

```python
from scipy.stats import nbinom

nbn_n = a + n * mean_y
nbn_p = (b + n) / (b + n + 1)
prediction = nbinom.pmf(xvals, n=nbn_n, p=nbn_p)
```

The Poisson-Gamma chain in full:

$$\underbrace{\text{Gamma}(\alpha, \beta)}_{\text{Prior}} \xrightarrow{\text{Poisson data}} \underbrace{\text{Gamma}(\alpha+n\bar{y},\; \beta+n)}_{\text{Posterior}} \xrightarrow{\text{integrate out }\theta} \underbrace{\text{NBinom}\!\left(\alpha+n\bar{y},\; \frac{\beta+n}{\beta+n+1}\right)}_{\text{Predictive Posterior}}$$

---

## Workshop: `Week10-BayesUpdating.py`

The workshop file `Week10-BayesUpdating.py` (Marimo notebook) implements the Poisson-Gamma Bayesian updating loop interactively. Key parameters to explore:

```python
ci = 0.95       # credible interval width
mu = 15         # true Poisson rate for fake data
a  = 2          # prior shape (try a=15 for a 'good' prior, a=1 for a bad one)
b  = 1          # prior rate/scale
n_for_plots = [1, 2, 3, 4, 5, 50]   # which data sizes to plot
```

For each value of $n$, the notebook plots:
1. **Upper panel**: Prior, data histogram, and Posterior on the same axes, with MAP and credible interval printed.
2. **Lower panel**: Predictive Posterior (NBinom) against the true Poisson PMF.

**Suggested experiments**:
- Try `a=15, b=1` (good prior) vs `a=2, b=1` (vague prior) vs `a=1, b=1` (very vague): observe how fast the posterior converges.
- Try `a=1, b=1` (very bad prior): notice how many data points are needed before the prior stops distorting results.
- Increase `ci` from 0.90 to 0.99 and observe how the credible interval widens.

---

## Summary

| Concept | Key equation | Python |
|---|---|---|
| Bayes' Theorem | $P(\theta\|y) \propto \mathcal{L}(y\|\theta)\,\pi(\theta)$ | — |
| Beta prior | $f(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}$ | `scipy.stats.beta(a, b)` |
| Bernoulli-Beta | $\alpha' = \alpha+y,\; \beta' = \beta+1-y$ | update $\alpha, \beta$ |
| MAP | mode of posterior | `(a_post-1)/b_post` for Gamma |
| 95% Credible Interval | posterior quantiles | `gamma.ppf(0.025, ...)`, `gamma.ppf(0.975, ...)` |
| Gamma prior | $f(\theta) \propto \theta^{\alpha-1}e^{-\beta\theta}$ | `scipy.stats.gamma(a, scale=1/b)` |
| Poisson-Gamma | $\alpha' = \alpha+n\bar{y},\; \beta' = \beta+n$ | update $\alpha, \beta$ |
| Predictive | $\text{NBinom}(\alpha', \beta'/(\beta'+1))$ | `scipy.stats.nbinom(n, p)` |

The power of Bayesian inference lies in its behaviour with small datasets: the prior preserves uncertainty where the data are sparse, and the posterior converges robustly as data accumulate. When an analytical posterior is not available, we use **MCMC** to sample from it numerically — covered in the [next chapter](#chapter:mcmc).
