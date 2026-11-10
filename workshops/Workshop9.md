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

(workshop-9)=
# Workshop 9: Bayesian Inference — Conjugate Priors

This workshop introduces **Bayesian parameter estimation** using a conjugate prior, which allows the posterior to be computed analytically. Material is drawn from [Bayesian Inference](#chapter:BayesianInference).

The dataset `DataW9.txt` contains hourly meteor counts recorded at an observatory over 40 hours. Load it as:

```python
import numpy as np
data = np.loadtxt('DataW9.txt', dtype=int)
```

We model the counts as Poisson-distributed with unknown rate $\lambda$ (meteors per hour). A **Gamma distribution** is the conjugate prior for the Poisson rate, meaning the posterior is also a Gamma distribution — no numerical integration required.

The Gamma distribution has PDF

$$
p(\lambda \mid a, b) = \frac{b^a}{\Gamma(a)} \lambda^{a-1} e^{-b\lambda}, \qquad \lambda > 0,
$$

where $a > 0$ is the shape and $b > 0$ is the rate parameter (note: `scipy.stats.gamma` uses `a` as shape and `scale = 1/b`).

---

## Task 1: True or False

State whether each of the following is **True** or **False**, and give a one-sentence justification.

1. In Bayesian inference, the posterior distribution represents our **updated belief** about a parameter after observing data.

2. A **conjugate prior** is a prior that has the same functional form as the likelihood.

3. With a Gamma(a, b) prior and Poisson data $\{x_i\}$, the posterior is Gamma$(a + \sum x_i,\; b + N)$.

4. As the dataset size $N \to \infty$, the posterior becomes insensitive to the choice of prior.

5. The **posterior predictive distribution** gives the probability of observing a new data point, averaged over uncertainty in the parameter.

6. A **vague** (weakly informative) Gamma prior with $a = 1, b = 0.01$ places most of its probability at very large $\lambda$.

---

## Task 2: Prior and Likelihood

a) Choose a weakly informative Gamma prior with $a_0 = 2$ and $b_0 = 0.5$. Using `scipy.stats.gamma`, plot the prior PDF for $\lambda \in [0, 20]$. Mark the prior mean ($a_0 / b_0$) with a dashed vertical line and label it.

b) Write a function `log_likelihood_poisson(lam, data)` returning $\ln\mathcal{L}(\lambda)$ for a Poisson model. Plot the (normalised) likelihood as a function of $\lambda$ over $[0, 20]$ on the same axes as the prior. You can normalise by dividing by `np.trapz(likelihood_vals, lam_vals)`.

c) What is the MLE $\hat\lambda$? (You may compute it analytically as the sample mean, or use `minimize_scalar` as in Workshop 7.)

d) Comment on the relative widths of the prior and likelihood curves. What does this tell you about the balance of information between prior and data?

---

## Task 3: Bayesian Update

a) Derive (or simply state from the notes) the updated posterior parameters after observing $N$ counts with sum $S = \sum x_i$:
   $$
   a_N = a_0 + S, \qquad b_N = b_0 + N.
   $$
   Compute $a_N$ and $b_N$ for the meteor dataset. Print the **posterior mean** $a_N / b_N$ and **posterior mode** $(a_N - 1) / b_N$.

b) Plot the prior, (normalised) likelihood, and posterior on the same axes. Use different line styles and colours. What do you observe about how the posterior relates to the prior and likelihood?

c) Compute and print a 95% **credible interval** for $\lambda$ using `scipy.stats.gamma.ppf(0.025, a=a_N, scale=1/b_N)` for the lower bound and `.ppf(0.975, ...)` for the upper.

d) Repeat the analysis with a **stronger** prior: $a_0 = 10, b_0 = 2$. How do the posterior mean and credible interval change compared to the weakly informative prior? What does this show about the influence of the prior?

---

## Task 4: Posterior Predictive Distribution

The **posterior predictive PMF** gives the probability of observing a new count $x_\text{new}$, averaged over the posterior uncertainty in $\lambda$. For Poisson data with a Gamma posterior, this is a **Negative Binomial** distribution:

$$
p(x_\text{new} \mid \text{data}) = \binom{x_\text{new} + a_N - 1}{x_\text{new}} \left(\frac{b_N}{1 + b_N}\right)^{a_N} \left(\frac{1}{1 + b_N}\right)^{x_\text{new}}.
$$

In `scipy.stats`, this is `nbinom(n=a_N, p=b_N/(1+b_N))`.

a) Compute and plot the posterior predictive PMF for $x_\text{new} = 0, 1, \ldots, 20$. Overplot the empirical frequency histogram of the observed data (normalised to a probability mass).

b) Compute the **Poisson predictive** using only the MLE: `scipy.stats.poisson(lam_hat).pmf(x_vals)`. Overplot this on the same axes.

c) The Negative Binomial is wider than the Poisson with the same mean. Compute and compare the means and variances of:
   - The posterior predictive (NegBinom)
   - The Poisson predictive (using $\hat\lambda$)
   
   Why is the Negative Binomial wider?

d) For a new 2-hour observation block, what is the probability of observing **10 or more** meteors in a single hour? Use the posterior predictive PMF (NegBinom). Compare to the Poisson predictive answer.

