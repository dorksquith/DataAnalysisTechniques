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
# Workshop 10 Solutions: Bayesian Inference — Conjugate Priors

---

## Task 1: True or False

1. **True.** The posterior $P(\theta | \text{data}) \propto P(\text{data}|\theta)\, P(\theta)$ encodes everything we know about $\theta$ after seeing the data.

2. **False.** A conjugate prior is one that, when combined with the likelihood, gives a posterior in the **same family** as the prior (not the same family as the likelihood).

3. **True.** With $N$ Poisson observations and sum $S = \sum y_i$, the $\mathsf{Gamma}(\alpha_0, \beta_0)$ prior updates to $\mathsf{Gamma}(\alpha_0 + S,\; \beta_0 + N)$.

4. **True.** As $N \to \infty$, the likelihood dominates the prior, and the posterior concentrates around the MLE whatever the prior (provided the prior gives non-zero probability to the true value).

5. **True.** The posterior predictive $P(\ynew | \text{data}) = \int P(\ynew|\theta)\,P(\theta|\text{data})\,d\theta$ averages the sampling distribution over the posterior.

6. **True.** $\mathsf{Gamma}(1, 0.01)$ is an Exponential distribution with mean $1/0.01 = 100$. It is nearly flat over a huge range, which is why it is a popular "vague" prior. But being flat over a huge range means that most of its probability lies at large $\theta$: for example $P(\theta > 20) = e^{-0.01\times 20} \approx 0.82$, even though the data suggest $\theta \approx 5$. A flat prior over a wide range is not the same as a prior with no information.

---

## Task 2: Prior and Likelihood

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson, nbinom
from matplotlib.colors import to_rgb

data = np.loadtxt('DataW10.txt', dtype=int)
N = len(data)
S = data.sum()

alpha0, beta0 = 2.0, 0.5
theta_vals = np.linspace(0.01, 20, 500)
```

### 2a — Prior

```{code-cell} python
prior_pdf  = gamma.pdf(theta_vals, a=alpha0, scale=1/beta0)
prior_mean = alpha0 / beta0

fig, ax = plt.subplots()
ax.plot(theta_vals, prior_pdf, color='blue', linestyle=':', linewidth=2,
        label=rf'Prior Gamma({alpha0:g}, {beta0:g})')
ax.axvline(prior_mean, color='k', linestyle='--', label=f'Prior mean = {prior_mean:.1f}')
ax.set_xlabel(r'$\theta$ (meteors/hour)')
ax.set_ylabel('Probability density')
ax.legend()
plt.tight_layout()
plt.show()
```

### 2b — Likelihood

```{code-cell} python
def log_likelihood_poisson(theta, data):
    # sum over i of (y_i ln(theta) - theta), with the ln(y_i!) constants dropped;
    # written with sum(data) and len(data) so it works for an array of theta values
    return np.sum(data) * np.log(theta) - len(data) * theta

ll_vals    = log_likelihood_poisson(theta_vals, data)
like_vals  = np.exp(ll_vals - ll_vals.max())        # subtract the max to avoid underflow
like_norm  = like_vals / np.trapezoid(like_vals, theta_vals)

fig, ax = plt.subplots()
ax.plot(theta_vals, prior_pdf, color='blue', linestyle=':', linewidth=2, label='Prior')
ax.plot(theta_vals, like_norm, color='green', linestyle='--', linewidth=2, label='Likelihood (normalised)')
ax.set_xlabel(r'$\theta$ (meteors/hour)')
ax.set_ylabel('Probability density')
ax.legend()
plt.tight_layout()
plt.show()
```

### 2c & 2d

```{code-cell} python
theta_hat = data.mean()
print(f"MLE theta_hat = {theta_hat:.4f}")
```

The likelihood is much narrower than the prior for $N=40$ observations: the data constrain $\theta$ far more strongly than the prior does.

---

## Task 3: Bayesian Update

### 3a — Posterior parameters

```{code-cell} python
alpha_N = alpha0 + S
beta_N  = beta0 + N
post_mean = alpha_N / beta_N
post_mode = (alpha_N - 1) / beta_N
print(f"alpha_N = {alpha_N},  beta_N = {beta_N}")
print(f"Posterior mean = {post_mean:.4f}")
print(f"Posterior mode = {post_mode:.4f}")
print(f"MLE            = {theta_hat:.4f}")
```

### 3b — Prior, likelihood and posterior

```{code-cell} python
post_pdf = gamma.pdf(theta_vals, a=alpha_N, scale=1/beta_N)

fig, ax = plt.subplots()
ax.plot(theta_vals, prior_pdf, color='blue', linestyle=':', linewidth=2, label='Prior')
ax.plot(theta_vals, like_norm, color='green', linestyle='--', linewidth=2, label='Likelihood (normalised)')
ax.plot(theta_vals, post_pdf, color='mediumvioletred', linewidth=2, label='Posterior')
ax.set_xlabel(r'$\theta$ (meteors/hour)')
ax.set_ylabel('Probability density')
ax.legend()
plt.tight_layout()
plt.show()
```

The posterior lies between the prior and the likelihood, but almost on top of the likelihood, because $N=40$ data points carry much more information than the weak prior.

### 3c — Credible interval

```{code-cell} python
ci_lo = gamma.ppf(0.025, a=alpha_N, scale=1/beta_N)
ci_hi = gamma.ppf(0.975, a=alpha_N, scale=1/beta_N)
print(f"95% equal-tailed credible interval: [{ci_lo:.3f}, {ci_hi:.3f}]")
```

The posterior here is nearly symmetric, so the 95% HPD interval would be almost the same.

### 3d — Strong prior

```{code-cell} python
alpha0_s, beta0_s = 100.0, 10.0          # prior mean 10, prior sd 1
alpha_N_s = alpha0_s + S
beta_N_s  = beta0_s + N
ci_lo_s = gamma.ppf(0.025, a=alpha_N_s, scale=1/beta_N_s)
ci_hi_s = gamma.ppf(0.975, a=alpha_N_s, scale=1/beta_N_s)
print(f"Strong prior: posterior mean = {alpha_N_s/beta_N_s:.4f},  95% CI: [{ci_lo_s:.3f}, {ci_hi_s:.3f}]")
print(f"Weak prior:   posterior mean = {post_mean:.4f},  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

post_pdf_s = gamma.pdf(theta_vals, a=alpha_N_s, scale=1/beta_N_s)
fig, ax = plt.subplots()
ax.plot(theta_vals, gamma.pdf(theta_vals, a=alpha0_s, scale=1/beta0_s), color='blue',
        linestyle=':', linewidth=2, label='Strong prior Gamma(100, 10)')
ax.plot(theta_vals, like_norm, color='green', linestyle='--', linewidth=2, label='Likelihood (normalised)')
ax.plot(theta_vals, post_pdf_s, color='mediumvioletred', linewidth=2, label='Posterior (strong prior)')
ax.plot(theta_vals, post_pdf, color='mediumvioletred', linewidth=1, alpha=0.5, label='Posterior (weak prior)')
ax.set_xlabel(r'$\theta$ (meteors/hour)')
ax.set_ylabel('Probability density')
ax.legend()
plt.tight_layout()
plt.show()
```

The strong prior is centred at $\theta = 10$, well away from the data ($\bar{y} \approx 5.2$), and is narrow enough to compete with 40 data points. It pulls the posterior mean up from about 5.2 to about 6.2, and the posterior now sits between the prior and the likelihood. A prior this confident should only be used if we really have that much information before seeing the data. With more data, its influence would shrink.

---

## Task 4: Posterior Predictive Distribution

### 4a — Negative Binomial predictive vs data

```{code-cell} python
y_vals = np.arange(0, 21)
p_nb   = beta_N / (1 + beta_N)
pred_pmf = nbinom.pmf(y_vals, n=alpha_N, p=p_nb)

# Observed counts, normalised to a probability mass
counts_hist = np.bincount(data, minlength=21)[:21] / N

fig, ax = plt.subplots()
ax.bar(y_vals - 0.2, counts_hist, width=0.35, color='green', alpha=0.4, label='Observed (normalised)')
ax.bar(y_vals + 0.2, pred_pmf, width=0.35, facecolor=(*to_rgb('orange'), 0.2),
       edgecolor='mediumvioletred', linestyle=':', linewidth=1.5, label='Posterior predictive (NegBinom)')
ax.set_xlabel(r'Counts per hour $\widetilde{y}$')
ax.set_ylabel('Probability')
ax.legend()
plt.tight_layout()
plt.show()
```

### 4b — Poisson predictive

```{code-cell} python
pred_pois = poisson.pmf(y_vals, mu=theta_hat)

fig, ax = plt.subplots()
ax.plot(y_vals, pred_pmf, 'o-', color='mediumvioletred', label='Posterior predictive (NegBinom)')
ax.plot(y_vals, pred_pois, '^--', color='grey', label=rf'Poisson($\hat\theta={theta_hat:.2f}$) predictive')
ax.set_xlabel(r'Counts per hour $\widetilde{y}$')
ax.set_ylabel('Probability')
ax.legend()
plt.tight_layout()
plt.show()
```

### 4c — Means and variances

```{code-cell} python
nb_mean = alpha_N / beta_N                       # = posterior mean
nb_var  = alpha_N * (1 + beta_N) / beta_N**2     # NegBinom variance

pois_mean = theta_hat
pois_var  = theta_hat

print(f"NegBinom: mean={nb_mean:.4f}, var={nb_var:.4f}")
print(f"Poisson:  mean={pois_mean:.4f}, var={pois_var:.4f}")
```

The Negative Binomial is wider because it includes the uncertainty in $\theta$ itself (parameter uncertainty), on top of the Poisson variation in the counts.

### 4d — P($\widetilde{y}$ ≥ 10)

```{code-cell} python
prob_nb   = nbinom.sf(9, n=alpha_N, p=p_nb)          # P(y >= 10) = 1 - CDF(9)
prob_pois = poisson.sf(9, mu=theta_hat)

print(f"P(y >= 10) NegBinom: {prob_nb:.4f}")
print(f"P(y >= 10) Poisson:  {prob_pois:.4f}")
```

The Negative Binomial gives a slightly higher probability, reflecting the extra uncertainty in $\theta$. For large $N$, the two converge.
