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

(workshop-10-solutions)=
# Workshop 10 Solutions: Markov Chain Monte Carlo

---

## Task 1: True or False

1. **False.** MH only requires the posterior up to a normalising constant; the log-ratio of unnormalised posteriors cancels the normalisation.

2. **True.** A tiny step size gives high acceptance but the chain moves very slowly through the posterior, taking many steps to explore it (poor mixing).

3. **True.** The chain starts at an arbitrary point and needs time to reach the stationary distribution; samples before this are biased toward the starting point.

4. **False.** An acceptance rate of 100% means proposals are always accepted, which typically happens when steps are tiny — leading to a slow random walk with poor mixing. For a 1D Gaussian target, an acceptance rate of ~44% is theoretically optimal; for high-dimensional problems, ~23% is often the target.

5. **True.** Drawing one sample from the sampling distribution for each MCMC sample and pooling the results gives samples from the posterior predictive, without any integration.

6. **True.** For a symmetric proposal $q(\theta'\mid\theta) = q(\theta\mid\theta')$, the proposal ratio cancels and the acceptance probability reduces to $\min(1, \pi(\theta')/\pi(\theta))$, which in log space is $\min(0, \ln\pi(\theta') - \ln\pi(\theta))$.

---

## Task 2: Warm-Up — MH on a Gaussian Target

### 2a–2b — Target and sampler

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as sp_norm, gamma as sp_gamma
from scipy.stats import nbinom

def log_target_gaussian(theta, mu=3.0, sigma=1.0):
    return sp_norm.logpdf(theta, loc=mu, scale=sigma)

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

### 2c — Trace plot

```{code-cell} python
rng = np.random.default_rng(42)
chain, acc_rate = metropolis_hastings(log_target_gaussian, start=0, n_steps=10000,
                                      prop_std=1.0, rng=rng)
print(f"Acceptance rate: {acc_rate:.3f}")

burnin = 1000
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(chain, lw=0.5, color='steelblue')
ax.axvspan(0, burnin, color='grey', alpha=0.3, label='Burn-in')
ax.set_xlabel('Step')
ax.set_ylabel(r'$\theta$')
ax.set_title('MCMC trace (Gaussian target, σ_prop=1.0)')
ax.legend()
plt.tight_layout()
plt.show()
```

### 2d — Posterior histogram

```{code-cell} python
samples = chain[burnin:]
theta_grid = np.linspace(0, 6, 300)

fig, ax = plt.subplots()
ax.hist(samples, bins=50, density=True, alpha=0.6, label='MCMC samples')
ax.plot(theta_grid, sp_norm.pdf(theta_grid, loc=3, scale=1), 'r-', lw=2, label='True target N(3,1)')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()

print(f"Sample mean = {samples.mean():.4f}  (true: 3.0)")
print(f"Sample std  = {samples.std():.4f}  (true: 1.0)")
```

### 2e — Step size comparison

```{code-cell} python
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, std, title in zip(axes, [0.05, 1.0, 10.0],
                          ['σ_prop=0.05 (too small)', 'σ_prop=1.0 (good)', 'σ_prop=10.0 (too large)']):
    rng_i = np.random.default_rng(42)
    ch, ar = metropolis_hastings(log_target_gaussian, start=0, n_steps=10000,
                                  prop_std=std, rng=rng_i)
    ax.plot(ch, lw=0.5)
    ax.set_ylabel(r'$\theta$')
    ax.set_title(f'{title}  — acceptance rate: {ar:.2f}')
axes[-1].set_xlabel('Step')
plt.tight_layout()
plt.show()
```

Small $\sigma_\text{prop}$: chain moves in tiny steps, very slow mixing despite high acceptance. Large $\sigma_\text{prop}$: most proposals are rejected, chain gets stuck. $\sigma_\text{prop} \approx 1$ gives the best balance.

---

## Task 3: MH for the Poisson–Gamma Posterior

### 3a — Log-posterior

```{code-cell} python
data = np.loadtxt('DataW9.txt', dtype=int)
N, S = len(data), data.sum()
a0, b0 = 2.0, 0.5

def log_posterior(lam, data, a0, b0):
    if lam <= 0:
        return -np.inf
    ll = np.sum(data * np.log(lam) - lam)   # Poisson log-lik (no factorial)
    lp = sp_gamma.logpdf(lam, a=a0, scale=1/b0)
    return ll + lp
```

### 3b — MH run

```{code-cell} python
rng3 = np.random.default_rng(42)
log_post_fn = lambda lam: log_posterior(lam, data, a0, b0)
chain3, acc3 = metropolis_hastings(log_post_fn, start=data.mean(), n_steps=20000,
                                    prop_std=0.5, rng=rng3)
print(f"Acceptance rate: {acc3:.3f}")

burnin3 = 2000
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(chain3, lw=0.5, color='steelblue')
ax.axvspan(0, burnin3, color='grey', alpha=0.3, label='Burn-in')
ax.set_xlabel('Step')
ax.set_ylabel(r'$\lambda$')
ax.set_title('MCMC trace (Poisson–Gamma posterior)')
ax.legend()
plt.tight_layout()
plt.show()
```

### 3c — MCMC vs exact posterior

```{code-cell} python
mcmc_samples = chain3[burnin3:]
a_N = a0 + S
b_N = b0 + N
lam_grid = np.linspace(3, 9, 400)
exact_pdf = sp_gamma.pdf(lam_grid, a=a_N, scale=1/b_N)

fig, ax = plt.subplots()
ax.hist(mcmc_samples, bins=60, density=True, alpha=0.6, label='MCMC samples')
ax.plot(lam_grid, exact_pdf, 'r-', lw=2, label=f'Exact Gamma({a_N}, {b_N:.1f})')
ax.set_xlabel(r'$\lambda$ (meteors/hour)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()
```

### 3d — Summary statistics

```{code-cell} python
print(f"MCMC mean  = {mcmc_samples.mean():.4f}  (exact: {a_N/b_N:.4f})")
print(f"MCMC std   = {mcmc_samples.std():.4f}  (exact: {np.sqrt(a_N)/b_N:.4f})")
print(f"MCMC 95% CI: [{np.percentile(mcmc_samples,2.5):.3f}, {np.percentile(mcmc_samples,97.5):.3f}]")
ci_lo = sp_gamma.ppf(0.025, a=a_N, scale=1/b_N)
ci_hi = sp_gamma.ppf(0.975, a=a_N, scale=1/b_N)
print(f"Exact 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
```

---

## Task 4: Posterior Predictive from MCMC

### 4a–4b — Predictive comparison

```{code-cell} python
rng2 = np.random.default_rng(99)
x_pred = rng2.poisson(lam=mcmc_samples)

x_vals = np.arange(0, 21)
p_nb = b_N / (1 + b_N)
exact_pmf = nbinom.pmf(x_vals, n=a_N, p=p_nb)

pred_hist = np.bincount(x_pred, minlength=21)[:21] / len(x_pred)

fig, ax = plt.subplots()
ax.bar(x_vals - 0.2, pred_hist, width=0.35, alpha=0.7, label='MCMC predictive')
ax.plot(x_vals + 0.2, exact_pmf, 'rs', ms=8, label='NegBinom predictive (exact)')
ax.set_xlabel('Counts per hour')
ax.set_ylabel('Probability')
ax.legend()
plt.tight_layout()
plt.show()
```

### 4c — P(x ≥ 10)

```{code-cell} python
p_mcmc = np.mean(x_pred >= 10)
p_exact = nbinom.sf(9, n=a_N, p=p_nb)
print(f"P(x ≥ 10) from MCMC:   {p_mcmc:.4f}")
print(f"P(x ≥ 10) NegBinom:    {p_exact:.4f}")
```

### 4d — Summary paragraph

We modelled hourly meteor counts as Poisson($\lambda$) with a Gamma(2, 0.5) prior on the unknown rate $\lambda$. Because the Gamma is the conjugate prior for the Poisson, the posterior is exactly Gamma($a_N$, $b_N$), which was computed in closed form in Workshop 9. In this workshop, we used the Metropolis–Hastings algorithm to draw samples from the same posterior without exploiting conjugacy, obtaining results that agree closely with the exact answer. The posterior predictive — the distribution over a future count — was estimated by simulating one Poisson draw per MCMC sample, reproducing the analytical Negative Binomial result. MCMC would be **essential** (rather than merely convenient) when the likelihood and prior are not conjugate — for example, if the prior encoded expert constraints that do not match the likelihood family, or when the model has multiple parameters whose joint posterior has no closed form.

