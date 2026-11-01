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

(workshop-10)=
# Workshop 10: Markov Chain Monte Carlo

This workshop introduces the **Metropolis–Hastings (MH) algorithm** for sampling from a posterior distribution that cannot be computed analytically. Material is drawn from [MCMC](#chapter:MCMC).

Workshop 9 showed that for a Poisson likelihood with a Gamma prior, the posterior is available in closed form. Workshop 10 uses the same **meteor count dataset** (`DataW9.txt`) but treats the problem as if the posterior were unknown — using MH to sample from it. This lets you verify your MCMC results against the exact answer.

```python
import numpy as np
data = np.loadtxt('DataW9.txt', dtype=int)
```

---

## Task 1: True or False

State whether each of the following is **True** or **False**, and give a one-sentence justification.

1. The Metropolis–Hastings algorithm requires the posterior to be normalised (i.e., the denominator in Bayes' theorem must be computed).

2. A Gaussian proposal distribution with step size $\sigma_\text{prop}$ that is too small leads to poor mixing, even if the chain eventually converges.

3. The **burn-in** period is discarded because the chain has not yet reached the target (stationary) distribution.

4. The acceptance rate in MH sampling should ideally be 100% — the more often proposals are accepted, the better the sampler.

5. MCMC samples from the posterior can be used directly to estimate the posterior predictive distribution, without integrating over the posterior analytically.

6. For a symmetric Gaussian proposal, the MH acceptance probability simplifies to $\min(1, \exp(\ln\pi(\theta') - \ln\pi(\theta)))$, where $\pi$ is the unnormalised posterior.

---

## Task 2: Warm-Up — MH on a Gaussian Target

Before tackling the real problem, build intuition with a known target. Suppose the target distribution is a standard Gaussian: $\pi(\theta) = \mathcal{N}(\theta; \mu=3, \sigma=1)$.

a) Write a function `log_target_gaussian(theta, mu=3, sigma=1)` returning $\ln\pi(\theta)$. You may use `scipy.stats.norm.logpdf` or write it explicitly.

b) Implement a Metropolis–Hastings sampler:

   ```python
   def metropolis_hastings(log_target, start, n_steps, prop_std, rng):
       chain = np.zeros(n_steps)
       chain[0] = start
       n_accept = 0
       for i in range(1, n_steps):
           current = chain[i-1]
           proposal = current + rng.normal(0, prop_std)
           log_alpha = log_target(proposal) - log_target(current)
           if np.log(rng.uniform()) < log_alpha:
               chain[i] = proposal
               n_accept += 1
           else:
               chain[i] = current
       return chain, n_accept / (n_steps - 1)
   ```

c) Run the sampler with `start=0`, `n_steps=10000`, `prop_std=1.0`, and a fixed `rng = np.random.default_rng(42)`. Print the acceptance rate. Plot the **trace** (chain values vs step number). Shade the burn-in period (first 1000 steps) in grey.

d) Discard the burn-in and plot a histogram of the remaining samples. Overplot the true Gaussian target PDF. Compute the sample mean and standard deviation and compare to the true values ($\mu=3$, $\sigma=1$).

e) Repeat with `prop_std=0.05` and `prop_std=10.0`. Plot all three traces on separate panels. Comment on the mixing quality and acceptance rate for each.

---

## Task 3: MH for the Poisson–Gamma Posterior

Now apply MH to the meteor count problem. The unnormalised log-posterior for $\lambda > 0$ is

$$
\ln\pi(\lambda \mid \text{data}) = \ln\mathcal{L}(\lambda) + \ln p(\lambda),
$$

where the likelihood is Poisson and the prior is Gamma$(a_0, b_0)$ with $a_0 = 2, b_0 = 0.5$.

a) Write a function `log_posterior(lam, data, a0, b0)` that returns $\ln\pi(\lambda)$. Use `scipy.stats.gamma.logpdf` for the prior term and the Poisson log-likelihood (without the factorial constant, since it cancels in the ratio).

b) Run the MH sampler with:
   - `start = data.mean()` (a sensible starting point)
   - `n_steps = 20000`
   - `prop_std = 0.5`
   - Reject proposals where $\lambda \leq 0$ (set log-target to $-\infty$ there)

   Print the acceptance rate. Plot the trace and shade the burn-in (first 2000 steps).

c) Discard the burn-in. Plot a histogram of the MCMC samples and overplot the **exact** Gamma posterior from Workshop 9 (use the same $a_N, b_N$). Do they agree?

d) From the MCMC samples, compute:
   - The posterior mean and standard deviation
   - A 95% credible interval (using `np.percentile`)
   
   Compare to the exact values from Workshop 9.

---

## Task 4: Posterior Predictive from MCMC Samples

One of the strengths of MCMC is that the posterior predictive can be estimated by simulation, without analytical integration.

a) From your MCMC samples $\{\lambda^{(s)}\}$ (after burn-in), draw one Poisson sample for each: $x_\text{new}^{(s)} \sim \text{Poisson}(\lambda^{(s)})$. The resulting array `x_pred` is a sample from the posterior predictive.

   ```python
   rng2 = np.random.default_rng(99)
   x_pred = rng2.poisson(lam=mcmc_samples)
   ```

b) Plot the histogram of `x_pred` as a normalised bar chart. On the same axes, plot the exact Negative Binomial predictive PMF from Workshop 9. Do they agree?

c) Estimate $P(x_\text{new} \geq 10)$ from the MCMC-based predictive by computing `np.mean(x_pred >= 10)`. Compare to the NegBinom answer from Workshop 9 Task 4d.

d) Write a short paragraph (4–5 sentences) summarising what you have done across Workshops 9 and 10: what was the model, what was the analytical result, how did MCMC approximate it, and in what situations would MCMC be **essential** rather than just a convenience?

