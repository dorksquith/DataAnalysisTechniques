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

(chapter:bay)=
# Bayesian Inference

* `scipy.stats`: `beta`, `gamma`, `nbinom`, `poisson`
* `scipy.special`: `beta as betafunc`
<!--* Workshop file: `Week10-BayesUpdating.py`-->



Bayesian Inference gives us a framework for **updating our beliefs** about a parameter $\theta$ using data $y$. Rather than treating $\theta$ as a fixed but unknown constant (the Frequentist view), the Bayesian approach treats $\theta$ as a random variable with its own probability distribution, and updates that distribution as data arrive.



---

(sec:bay_baythm)=
## Bayes' Theorem

We already met Bayes' theorem in {ref}`sec:pe_baythm`, where we wrote it as {eq}`eq:pe_bayes`:

$$P(A | B) = \frac{P(B | A)\, P(A)}{P(B)}$$

The A and B in the above formula are events. We noted that Bayes' theorem becomes most interesting when we define the events as a hypothesis (or parameter value) $\theta$ and a dataset $y$:

$$P(\theta | y) = \frac{P(y | \theta)\, P(\theta)}{P(y)}$$

When using Bayes' theorem in this way, it is convenient to define each of the terms:

- **$P(y | \theta) \to \mathcal{L}(y | \theta)$**: the **Likelihood** of the data $y$ being described by a parameter value $\theta$. The data are now *fixed* (we've already observed them) and $\theta$ is what varies.
- **$P(\theta) \to \pi(\theta)$**: the **Prior** Probability, $\pi$, is the belief about $\theta$ before seeing the data, not a frequency from a repeatable experiment. The use of the Greek letter $\pi$ is a deliberate choice to keep it separate from sampling probabilities.
- **$P(y) \to \int \mathcal{L}(y | \theta)\, \pi(\theta)\, d\theta$**: the **Evidence**; $P(y)$ isn't an independent quantity — it's what we get by integrating the numerator over all possible $\theta$ values. Writing the integral out explicitly makes this obvious, and guarantees the right hand side integrates to 1.
- **$P(\theta | y)$**: the **Posterior** Probability of the parameter $\theta$, given the data.

Bayes' Theorem relates the **Posterior** probability of a parameter given data to the **Likelihood** of the data given the parameter, the **Prior** probability of the parameter, and the **Evidence**:

:::{figure} /figures/BayesPic.png
:name: fig:bay_bayespic
:width: 60%
Bayes' theorem: The Posterior is equal to the product of the Likelihood and Prior, divided by the Evidence.
:::

<!--
```{math}
:label: eq:bay_bayes

P(\theta \mid y) = \frac{\mathcal{L}(y \mid \theta)\; \pi(\theta)}{\int \mathcal{L}(y \mid \theta)\, \pi(\theta)\, d\theta}
```
-->

Each element has a precise name and role:

| Term | Symbol | Meaning |
|---|---|---|
| **Posterior** | $P(\theta \mid y)$ | Our updated belief about $\theta$ after seeing data $y$ |
| **Likelihood** | $\mathcal{L}(y \mid \theta)$ | The probability of observing data $y$ given $\theta$ |
| **Prior** | $\pi(\theta)$ | Our belief about $\theta$ before seeing data |
| **Evidence** | $\int \mathcal{L}(y \mid \theta)\, \pi(\theta)\, d\theta$ | A normalisation constant ensuring $\int P(\theta \mid y)\, d\theta = 1$ |

The Prior is a PDF (or PMF for discrete $\theta$), not a single number. In the discrete example below there are only two hypotheses, so the Prior reduces to two numbers: $\pi(\theta_s)$ and $\pi(\theta_h)$. This is a special case; in general $\theta$ ranges continuously over a whole parameter space and the Prior is a full probability distribution over that space.



---

(sec:bay_covid)=
## A Medical Screening Test

A screening test for a rare condition has the following known properties (from the manufacturer):

|| Positive, $y=1$  | Negative $y=0$ | 
|---|---|---|
| **Sick** ($\theta_s$) | $p = 1$ | $p = 0$ |
| **Healthy** ($\theta_h$) | $p = 0.05$ | $p = 0.95$ |

The data $y$ are **Bernoulli**: a binary outcome from a single test. Each hypothesis specifies a different value of the Bernoulli parameter $p$ — each lives in its own probability distribution:

::::{grid} 1 1 2 2

:::{figure} /figures/NewBernCovid_sick.png
:name: fig:bay_bernsick
:width: 100%
Bernoulli PMF under the sick hypothesis ($\theta_s$): $p = 1$, so a positive result is certain.
:::

:::{figure} /figures/NewBernCovid_healthy.png
:name: fig:bay_bernhealthy
:width: 100%
Bernoulli PMF under the healthy hypothesis ($\theta_h$): $p = 0.05$, a small false-positive rate.
:::

::::

We will now walk through each term of [](#fig:bay_bayespic) in turn.

---

(sec:bay_covidlike)=

**Likelihood.** The [Bernoulli PMF](#eq:pmf_bernpmf) is:

$$p_Y(y \mid \theta) = \theta^y (1-\theta)^{1-y}$$

Here we are using $p \rightarrow \theta$ for the Bernoulli parameter and $k \rightarrow y$ for the data. 

For $N=1$ the Likelihood equals the PMF. We get one Likelihood function per hypothesis:

$$\mathcal{L}(y=1 \mid \theta_s) = 1^1 \cdot 0^0 = 1 \qquad \mathcal{L}(y=0 \mid \theta_s) = 0$$

$$\mathcal{L}(y=1 \mid \theta_h) = 0.05^1 \cdot 0.95^0 = 0.05 \qquad \mathcal{L}(y=0 \mid \theta_h) = 0.95$$

Under $\theta_s$ a positive test is certain; under $\theta_h$ it occurs with probability 0.05 (the [false positive rate](#eq:hyp_fpr)).

---

**Prior.** The prevalence of this condition in the general population is roughly 1 in 1000. We set:

$$\pi(\theta_s) = \tfrac{1}{1000}, \qquad \pi(\theta_h) = \tfrac{999}{1000}$$

The prior distributions $\pi(\theta)$ is reduced to two numbers, because there are only two hypotheses. In general $\theta$ ranges over a continuous parameter space and the Prior is a full distribution. 

:::{warning}
Note that setting $\pi(\theta_s) = 1/1000$ without stating the uncertainty is highly improper! A continuous prior, like the Beta distribution we'll meet in [](#sec:bay_betpdf), would allow us to express the prior with uncertainty, as we always should. This is a toy example.
:::

---

(sec:bay_evcon)=

**Evidence.** The denominator of [](#fig:bay_bayespic) is the total probability of the data (a single outcome) under all hypotheses, weighted by their priors. With two discrete hypotheses it is a sum:

```{math}
:label: eq:bay_evidence

\begin{aligned}
\text{Evidence} &= \mathcal{L}(y=1 \mid \theta_s)\,\pi(\theta_s) + \mathcal{L}(y=1 \mid \theta_h)\,\pi(\theta_h) \\
&= 1 \times \tfrac{1}{1000} + 0.05 \times \tfrac{999}{1000} \\
&= 0.001 + 0.04995 \\
&= 0.05095
\end{aligned}
```

Note that we do not integrate over the data $y$: this is the evidence for a positive test result only (one of the two possible outcomes). The same procedure can be done for the other possible outcome, $y=0$.

Note also that the evidence doesn't depend on which $\theta$ you're evaluating — that's exactly why dividing by it makes the posteriors sum to 1. In the continuous case this sum becomes the integral in [](#fig:bay_bayespic), which is often impossible to compute analytically — a key motivation for the numerical methods in the next chapter. <!-- TODO: add link to MCMC chapter when complete -->

---

(sec:bay_covidpost)=

**Posterior.** We now have everything we need. For a positive test result $y=1$:

```{math}
:label: eq:bay_covidpost

\begin{aligned}
P(\theta_s \mid y=1) &= \frac{\mathcal{L}(y=1 \mid \theta_s)\,\pi(\theta_s)}{\text{Evidence}} \\[8pt]
&= \frac{0.001}{0.05095} \approx 0.0196
\end{aligned}
```

Despite a perfect sensitivity — sick people **always** test positive — the probability of actually being sick given a positive test is only $\approx 2\%$. The condition is so rare that most positive results come from the small false-positive rate applied to the large healthy population.

---

(sec:bay_bayupd)=
### Bayesian Updating

The full posterior table makes the update visible for both hypotheses:

| $\theta$ | Likelihood $\mathcal{L}(y=1\mid \theta)$ | Prior $\pi(\theta)$ | Numerator | Evidence | Posterior $P(\theta \mid y=1)$ |
|---|---|---|---|---|---|
| $\theta_s$ | 1 | 0.001 | 0.001 | 0.05095 | **0.0196** |
| $\theta_h$ | 0.05 | 0.999 | 0.04995 | 0.05095 | **0.9804** |

This is **Bayesian Updating**:

- **Before** the test, we held a Prior $\pi(\theta_s) = 0.001$
- **After** the test, we calculated a Posterior $P(\theta_s \mid y=1) \approx 0.02$

The data have updated our belief. With more data (more tests), the posterior would narrow further.

---

(sec:bay_priodist)=
## Prior Distributions

(sec:bay_betpdf)=
### The Beta PDF

(bayesian:beta)=

The **kernel** of a distribution is the part of its PDF or PMF that depends on the random variable — the normalising constant is dropped. Recognising kernel shapes is a powerful shortcut: when two distributions share a kernel shape, one is the natural conjugate prior for the other.

The [**Bernoulli**](#eq:pmf_berpmf) and [**Binomial**](#eq:pmf_binpmf) likelihoods both have the same kernel shape in $\theta$:

$$\text{Bernoulli:}\quad \theta^y(1-\theta)^{1-y} \qquad \text{Binomial:}\quad \theta^y(1-\theta)^{n-y}$$

Both are of the form $\theta^a(1-\theta)^b$. The **Beta kernel** is:

$$\theta^{\alpha-1}(1-\theta)^{\beta-1}$$

which has exactly the same form — with $a = \alpha-1$ and $b = \beta-1$. Beta is therefore the natural **Conjugate Prior** for both Bernoulli and Binomial likelihoods.

The Beta distribution places a probability distribution on the **parameter** $\theta$ itself — not on data. The random variable is $\theta$, which plays the role of the Bernoulli parameter $p$. We are expressing uncertainty about what $p$ might be, rather than about what a data point will be.

The Beta PDF is:

```{math}
:label: eq:bay_betapdf

f_\theta(\theta) = \frac{1}{B(\alpha,\beta)}\, \theta^{\alpha-1} (1-\theta)^{\beta-1}
\qquad 0 \leq \theta \leq 1, \quad \alpha > 0,\; \beta > 0
```

where $B(\alpha, \beta)$ is the Beta function (a normalising constant).

The Beta PDF has
: $$\label{eq:bay_betaexp}E[\theta] = \frac{\alpha}{\alpha+\beta}$$

  $$\label{eq:bay_betavar}V[\theta] = \frac{E[\theta]\,(1-E[\theta])}{\alpha+\beta+1}$$

The two **shape hyper-parameters** $\alpha$ and $\beta$ are a "chameleon" pair: larger $\alpha$ pulls mass toward $\theta=1$; larger $\beta$ pulls it toward $\theta=0$; large equal values give a symmetric bell.

::::{grid} 1 1 2 2

:::{figure} /figures/Beta1_1.png
:name: fig:bay_beta11
:width: 100%
$\mathsf{Beta}(1,1)$: uniform — no prior knowledge.
:::

:::{figure} /figures/Beta1_100.png
:name: fig:bay_beta1100
:width: 100%
$\mathsf{Beta}(1,100)$: concentrated near 0 — strong prior belief that $\theta$ is small.
:::

:::{figure} /figures/BetaPDF-100-1.png
:name: fig:bay_beta1001
:width: 100%
$\mathsf{Beta}(100,1)$: concentrated near 1 — strong prior belief that $\theta$ is large.
:::

:::{figure} /figures/Beta100_100.png
:name: fig:bay_beta100100
:width: 100%
$\mathsf{Beta}(100,100)$: symmetric bell at 0.5 — strong belief that $\theta \approx 0.5$.
:::

::::

**Bernoulli-Beta conjugacy**

(bayesian:conjugacy_bb)=

Because the Beta kernel matches the Bernoulli/Binomial kernel, the posterior is also Beta. Starting with a $\mathsf{Beta}(\alpha, \beta)$ prior and a Bernoulli likelihood:

```{math}
:label: eq:bay_bbpost

\begin{align*}
P(\theta \mid y) &\propto \mathcal{L}(y \mid \theta)\; \pi(\theta) \\
&\propto \theta^y (1-\theta)^{1-y} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1} \\
&\propto \theta^{(\alpha+y)-1}(1-\theta)^{(\beta+1-y)-1}
\end{align*}
```

The posterior is $\mathsf{Beta}(\alpha', \beta')$ with updated hyper-parameters:

$$\alpha' = \alpha + y \qquad \beta' = \beta + (1-y)$$

A **positive** result ($y=1$) increments $\alpha$; a **negative** result ($y=0$) increments $\beta$. For **Binomial** data with $n$ trials and $y$ successes:

$$P(\theta \mid y) \propto \mathsf{Beta}(\alpha+y,\; \beta+n-y)$$

**Worked example.** Start with no prior knowledge: $\pi(\theta) = \mathsf{Beta}(1,1)$ (uniform). Observe a single positive result, $y=1$:

$$\alpha' = 1 + 1 = 2 \qquad \beta' = 1 + (1-1) = 1$$

The posterior is $\mathsf{Beta}(2,1)$, with $E[\theta] = 2/3$. One success has shifted our belief upward from $0.5$. The rule is: **each success adds 1 to $\alpha$, each failure adds 1 to $\beta$**.

---

(sec:bay_gampdf)=
### The Gamma PDF

(bayesian:gamma)=

The **Poisson** and **Exponential** likelihoods have kernels in $\theta$:

$$\text{Poisson:}\quad e^{-\theta}\theta^y \qquad \text{Exponential:}\quad \theta e^{-\theta y}$$

Both are of the form $\theta^a e^{-b\theta}$. The **Gamma kernel** is $\theta^{\alpha-1} e^{-\beta\theta}$, which has the same form — making Gamma the natural conjugate prior for Poisson and Exponential likelihoods.

The Gamma PDF is:

```{math}
:label: eq:bay_gammapdf

f(\theta) = \frac{\beta^\alpha}{\Gamma(\alpha)}\, \theta^{\alpha-1} e^{-\beta\theta}
```

with $E[\theta] = \alpha\beta$ and $V[\theta] = \alpha\beta^2$ (using the scale convention, where $\beta$ is the scale parameter). Note: in `scipy.stats.gamma`, pass `scale=1/beta_rate` when $\beta$ is a *rate*.

**Poisson-Gamma conjugacy**

(bayesian:poisson_gamma)=

Suppose we observe Poisson-distributed counts (e.g. particle decays, bus arrivals) with unknown rate $\theta$. We place a Gamma prior on $\theta$.

**Choosing hyper-parameters**: If our best guess is $\theta = 15$, we want $E[\theta] = \alpha\beta = 15$ and $V[\theta] = \alpha\beta^2 = 15$ (matching Poisson's equal mean and variance). This gives $\beta = 1$, $\alpha = 15$. So we use $\pi(\theta) = \mathsf{Gamma}(15, 1)$.

**Posterior update**: After $n$ data points with sample mean $\bar{y}$:

$$P(\theta \mid y) = \mathsf{Gamma}(\alpha + n\bar{y},\; \beta + n)$$

The **MAP estimate** (mode of Gamma) is:

$$\hat{\theta}_{MAP} = \frac{\alpha' - 1}{\beta'} = \frac{(\alpha + n\bar{y}) - 1}{\beta + n}$$

As $n$ grows, the posterior narrows around the true value and the choice of prior matters less.

| Likelihood | Posterior |
|---|---|
| Bernoulli: $\theta^y(1-\theta)^{1-y}$ | $\mathsf{Beta}(\alpha+y,\; \beta+1-y)$ |
| Binomial: $\binom{n}{y}\theta^y(1-\theta)^{n-y}$ | $\mathsf{Beta}(\alpha+y,\; \beta+n-y)$ |
| Poisson: $e^{-\theta}\theta^y/y!$ | $\mathsf{Gamma}(\alpha+n\bar{y},\; \beta+n)$ |
| Exponential: $\theta e^{-\theta y}$ | $\mathsf{Gamma}(\alpha+1,\; \beta+n\bar{y})$ |

---

(sec:bay_map)=
## Maximum A Posteriori Estimation

The **MAP estimate** $\hat{\theta}_{MAP}$ is the mode of the Posterior — the value of $\theta$ that maximises $P(\theta \mid y)$:

$$\hat{\theta}_{MAP} = \arg\max_\theta \; P(\theta \mid y) = \arg\max_\theta \; \mathcal{L}(y \mid \theta)\; \pi(\theta)$$

For the Beta distribution, $\text{mode} = (\alpha'-1)/(\alpha'+\beta'-2)$.

If the Prior is **uniform** (uninformative), then $\hat{\theta}_{MAP}$ coincides with the **MLE** $\hat{\theta}_{MLE}$.

The key advantage of MAP over MLE is that it incorporates prior beliefs — especially valuable with small datasets. In the limit of large $n$, MAP $\rightarrow$ MLE regardless of the prior chosen.

---

(sec:bay_credint)=
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

(sec:bay_priors)=
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

(sec:bay_predpost)=
## Predictive Posterior

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

---

(sec:bay_sum)=
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
