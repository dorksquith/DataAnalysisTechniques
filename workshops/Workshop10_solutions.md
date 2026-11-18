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

(workshop-9-solutions)=
# Workshop 9 Solutions: Bayesian Inference — Conjugate Priors

---

## Task 1: True or False

1. **True.** The posterior $p(\theta \mid \text{data}) \propto p(\text{data}\mid\theta)\, p(\theta)$ encodes everything we know about $\theta$ after seeing the data.

2. **False.** A conjugate prior is one that, when combined with the likelihood, yields a posterior in the **same family** as the prior (not the same family as the likelihood).

3. **True.** With $N$ Poisson observations and sum $S$, the Gamma$(a_0, b_0)$ prior updates to Gamma$(a_0 + S,\; b_0 + N)$.

4. **True.** As $N \to \infty$, the likelihood dominates the prior, and the posterior concentrates around the MLE regardless of the prior choice (provided the prior assigns non-zero probability to the true value).

5. **True.** The posterior predictive $p(x_\text{new} \mid \text{data}) = \int p(x_\text{new}\mid\lambda)\,p(\lambda\mid\text{data})\,\mathrm{d}\lambda$ averages the sampling distribution over the posterior.

6. **True.** Gamma$(1, 0.01)$ has mean $1/0.01 = 100$, placing most probability at large $\lambda$ — not vague in the conventional sense. A vague prior for a typical count rate should have a mean comparable to expected values.

---

## Task 2: Prior and Likelihood

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as sp_gamma, poisson as sp_poisson
from scipy.optimize import minimize_scalar

data = np.loadtxt('DataW9.txt', dtype=int)
N = len(data)
S = data.sum()

a0, b0 = 2.0, 0.5
lam_vals = np.linspace(0.01, 20, 500)
```

### 2a — Prior

```{code-cell} python
prior_pdf = sp_gamma.pdf(lam_vals, a=a0, scale=1/b0)
prior_mean = a0 / b0

fig, ax = plt.subplots()
ax.plot(lam_vals, prior_pdf, 'b-', label=f'Prior Gamma({a0}, {b0})')
ax.axvline(prior_mean, color='b', linestyle='--', alpha=0.6, label=f'Prior mean = {prior_mean:.1f}')
ax.set_xlabel(r'$\lambda$ (meteors/hour)')
ax.set_ylabel('Probability density')
ax.legend()
plt.tight_layout()
plt.show()
```

### 2b — Likelihood

```{code-cell} python
def log_likelihood_poisson(lam, data):
    return np.sum(data * np.log(lam) - lam)   # constants in lam dropped

like_vals = np.exp(log_likelihood_poisson(lam_vals, data) - log_likelihood_poisson(lam_vals, data).max())
like_norm = like_vals / np.trapz(like_vals, lam_vals)

fig, ax = plt.subplots()
ax.plot(lam_vals, prior_pdf / prior_pdf.max(), 'b-', label='Prior (scaled)')
ax.plot(lam_vals, like_norm / like_norm.max(), 'g--', label='Likelihood (scaled)')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('Scaled density')
ax.legend()
plt.tight_layout()
plt.show()
```

### 2c & 2d

```{code-cell} python
lam_hat = data.mean()
print(f"MLE λ̂ = {lam_hat:.4f}")
```

The likelihood is typically much narrower than the prior for $N=40$ observations, showing the data strongly constrain $\lambda$.

---

## Task 3: Bayesian Update

### 3a — Posterior parameters

```{code-cell} python
a_N = a0 + S
b_N = b0 + N
post_mean = a_N / b_N
post_mode = (a_N - 1) / b_N
print(f"a_N = {a_N},  b_N = {b_N}")
print(f"Posterior mean = {post_mean:.4f}")
print(f"Posterior mode = {post_mode:.4f}")
print(f"MLE            = {lam_hat:.4f}")
```

### 3b — Three curves

```{code-cell} python
post_pdf = sp_gamma.pdf(lam_vals, a=a_N, scale=1/b_N)

fig, ax = plt.subplots()
ax.plot(lam_vals, prior_pdf, 'b-', label='Prior', alpha=0.7)
ax.plot(lam_vals, like_norm, 'g--', label='Likelihood (normalised)', alpha=0.7)
ax.plot(lam_vals, post_pdf, 'r-', lw=2, label='Posterior')
ax.set_xlabel(r'$\lambda$ (meteors/hour)')
ax.set_ylabel('Probability density')
ax.legend()
plt.tight_layout()
plt.show()
```

The posterior lies between the prior and the likelihood, but is pulled toward the likelihood because $N=40$ data points carry much more information than the weak prior.

### 3c — Credible interval

```{code-cell} python
ci_lo = sp_gamma.ppf(0.025, a=a_N, scale=1/b_N)
ci_hi = sp_gamma.ppf(0.975, a=a_N, scale=1/b_N)
print(f"95% credible interval: [{ci_lo:.3f}, {ci_hi:.3f}]")
```

### 3d — Stronger prior

```{code-cell} python
a0_strong, b0_strong = 10.0, 2.0
a_N_s = a0_strong + S
b_N_s = b0_strong + N
ci_lo_s = sp_gamma.ppf(0.025, a=a_N_s, scale=1/b_N_s)
ci_hi_s = sp_gamma.ppf(0.975, a=a_N_s, scale=1/b_N_s)
print(f"Strong prior — posterior mean: {a_N_s/b_N_s:.4f},  95% CI: [{ci_lo_s:.3f}, {ci_hi_s:.3f}]")
print(f"Weak prior  — posterior mean: {post_mean:.4f},  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
```

The strong prior pulls the posterior mean toward its own mean ($10/2 = 5$). With more data, this influence would shrink further.

---

## Task 4: Posterior Predictive Distribution

### 4a — NegBinom predictive vs data

```{code-cell} python
from scipy.stats import nbinom, poisson as sp_poisson2

x_vals = np.arange(0, 21)
p_nb = b_N / (1 + b_N)
pred_pmf = nbinom.pmf(x_vals, n=a_N, p=p_nb)

# Empirical counts normalised
counts_hist = np.bincount(data, minlength=21)[:21] / N

fig, ax = plt.subplots()
ax.bar(x_vals - 0.2, counts_hist, width=0.35, label='Observed (normalised)', alpha=0.7)
ax.bar(x_vals + 0.2, pred_pmf, width=0.35, label='NegBinom predictive', alpha=0.7)
ax.set_xlabel('Counts per hour')
ax.set_ylabel('Probability')
ax.legend()
plt.tight_layout()
plt.show()
```

### 4b — Poisson predictive

```{code-cell} python
pred_pois = sp_poisson2.pmf(x_vals, mu=lam_hat)

fig, ax = plt.subplots()
ax.plot(x_vals, pred_pmf, 'rs-', ms=6, label='NegBinom predictive')
ax.plot(x_vals, pred_pois, 'b^--', ms=6, label=f'Poisson($\\hat\\lambda={lam_hat:.2f}$) predictive')
ax.set_xlabel('Counts per hour')
ax.set_ylabel('Probability')
ax.legend()
plt.tight_layout()
plt.show()
```

### 4c — Means and variances

```{code-cell} python
nb_mean = a_N / b_N                     # = posterior mean
nb_var  = a_N * (1 + b_N) / b_N**2     # NegBinom variance

pois_mean = lam_hat
pois_var  = lam_hat

print(f"NegBinom: mean={nb_mean:.4f}, var={nb_var:.4f}")
print(f"Poisson:  mean={pois_mean:.4f}, var={pois_var:.4f}")
```

The NegBinom is wider because it accounts for uncertainty in $\lambda$ itself (parameter uncertainty), on top of the intrinsic Poisson sampling variability.

### 4d — P(x ≥ 10)

```{code-cell} python
prob_nb   = nbinom.sf(9, n=a_N, p=p_nb)          # P(X >= 10) = 1 - CDF(9)
prob_pois = sp_poisson2.sf(9, mu=lam_hat)

print(f"P(x ≥ 10) NegBinom:  {prob_nb:.4f}")
print(f"P(x ≥ 10) Poisson:   {prob_pois:.4f}")
```

The NegBinom gives a slightly higher probability, reflecting the extra uncertainty in $\lambda$. For large $N$, the two converge.

