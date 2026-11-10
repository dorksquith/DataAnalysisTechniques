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

(workshop-7-solutions)=
# Workshop 7 Solutions: Numerical MLE with `scipy.optimize`

---

## Task 1: True or False

1. **True.** `minimize_scalar` operates on a scalar (single-variable) function; for two or more parameters use `minimize`.

2. **False.** Doubling the dataset sharpens the NLL (increases its curvature at the minimum), so the uncertainty decreases. The location of the minimum is approximately unchanged.

3. **True.** The inverse of the Hessian of the NLL at the minimum is the estimated covariance matrix; its diagonal entries are the variances of the MLEs.

4. **False.** L-BFGS-B returns a *limited-memory approximation* to the inverse Hessian (`hess_inv.todense()`), not the exact inverse. It is often a good estimate but can underestimate uncertainties in poorly conditioned problems.

5. **True.** Differentiating $-\ln\mathcal{L}(\lambda) = N\lambda - N\bar x\ln\lambda$ with respect to $\lambda$ and setting to zero gives $\hat\lambda = \bar x$.

6. **False.** For the Gaussian, the MLEs of $\mu$ and $\sigma$ are independent: $\hat\mu = \bar x$ and $\hat\sigma^2 = \frac{1}{N}\sum(x_i-\bar x)^2$, so they can be found separately.

---

## Task 2: Poisson MLE

### 2a — NLL function

```{code-cell} python
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt

counts = np.loadtxt('DataW7.txt', dtype=int)

def nll_poisson(lam, data):
    return np.sum(lam - data * np.log(lam) + gammaln(data + 1))
```

### 2b — `minimize_scalar`

```{code-cell} python
result = minimize_scalar(
    fun=lambda lam: nll_poisson(lam, counts),
    bounds=(0.5 * counts.mean(), 2 * counts.mean()),
    method='bounded'
)
lam_hat = result.x
print(f"MLE:         λ̂ = {lam_hat:.4f}")
print(f"Sample mean: x̄ = {counts.mean():.4f}")
```

The MLE and sample mean agree to many decimal places, confirming the analytic result.

### 2c — NLL plot

```{code-cell} python
lam_vals = np.linspace(0.5 * counts.mean(), 1.5 * counts.mean(), 500)
nll_vals = [nll_poisson(l, counts) for l in lam_vals]

fig, ax = plt.subplots()
ax.plot(lam_vals, nll_vals - min(nll_vals), color='steelblue')
ax.axvline(lam_hat, linestyle='--', color='tomato', label=rf'$\hat\lambda = {lam_hat:.3f}$')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'NLL $- $ NLL$_{\min}$')
ax.set_title('Poisson NLL (shifted to zero at minimum)')
ax.legend()
plt.tight_layout()
plt.show()
```

### 2d — Uncertainty from NLL offset

```{code-cell} python
nll_min = nll_poisson(lam_hat, counts)
target = nll_min + 0.5

lam_lo = brentq(lambda l: nll_poisson(l, counts) - target, lam_vals[0], lam_hat)
lam_hi = brentq(lambda l: nll_poisson(l, counts) - target, lam_hat, lam_vals[-1])

delta_lo = lam_hat - lam_lo
delta_hi = lam_hi - lam_hat
print(f"λ̂ = {lam_hat:.3f}  +{delta_hi:.3f} / -{delta_lo:.3f}")
```

The interval should be approximately symmetric and of order $1/\sqrt{N}$.

---

## Task 3: Gaussian MLE

### 3a — NLL function

```{code-cell} python
from scipy.stats import norm as sp_norm
from scipy.optimize import minimize

energies = np.loadtxt('DataW7b.txt')

def nll_gaussian(params, data):
    mu, sigma = params
    return -np.sum(sp_norm.logpdf(data, loc=mu, scale=sigma))
```

### 3b — `minimize`

```{code-cell} python
x0 = [energies.mean(), energies.std()]
result2 = minimize(
    fun=nll_gaussian,
    x0=x0,
    args=(energies,),
    method='L-BFGS-B',
    bounds=[(None, None), (0.01, None)]
)
mu_hat, sig_hat = result2.x
print(f"μ̂ = {mu_hat:.4f},  σ̂ = {sig_hat:.4f}")
```

### 3c — Standard errors from Hessian

```{code-cell} python
hess_inv = result2.hess_inv.todense()
se_mu  = np.sqrt(hess_inv[0, 0])
se_sig = np.sqrt(hess_inv[1, 1])
print(f"μ̂ = {mu_hat:.4f} ± {se_mu:.4f}")
print(f"σ̂ = {sig_hat:.4f} ± {se_sig:.4f}")
```

The standard error on $\mu$ should be roughly $\sigma/\sqrt{N}$, and on $\sigma$ roughly $\sigma/\sqrt{2N}$.

### 3d — 2D NLL contour

```{code-cell} python
n_grid = 40
mu_grid = np.linspace(mu_hat - 3*se_mu, mu_hat + 3*se_mu, n_grid)
sig_grid = np.linspace(sig_hat - 3*se_sig, sig_hat + 3*se_sig, n_grid)
MU, SIG = np.meshgrid(mu_grid, sig_grid)

NLL_grid = np.array([
    [nll_gaussian([m, s], energies) for m in mu_grid]
    for s in sig_grid
])
NLL_shift = NLL_grid - result2.fun

fig, ax = plt.subplots(figsize=(6, 5))
cf = ax.contourf(MU, SIG, NLL_shift, levels=30, cmap='viridis_r')
plt.colorbar(cf, ax=ax, label=r'NLL $-$ NLL$_{\min}$')
ax.contour(MU, SIG, NLL_shift, levels=[0.5, 2.0], colors=['white', 'yellow'], linewidths=1.5)
ax.plot(mu_hat, sig_hat, 'w*', ms=12, label='MLE')
ax.set_xlabel(r'$\mu$ (GeV)')
ax.set_ylabel(r'$\sigma$ (GeV)')
ax.set_title('NLL surface')
ax.legend()
plt.tight_layout()
plt.show()
```

The white contour ($+0.5$) is the approximate 68% CI region; the yellow contour ($+2.0$) is the approximate 95% region.

---

## Task 4: Model Comparison with the LLR

### 4a — Log-likelihoods

```{code-cell} python
sigma_known = 0.72
mu_A, mu_B = 3.15, 3.50

lnL_A = np.sum(sp_norm.logpdf(energies, loc=mu_A, scale=sigma_known))
lnL_B = np.sum(sp_norm.logpdf(energies, loc=mu_B, scale=sigma_known))
print(f"ln L_A = {lnL_A:.3f}")
print(f"ln L_B = {lnL_B:.3f}")
print("Model A is preferred" if lnL_A > lnL_B else "Model B is preferred")
```

### 4b — LLR

```{code-cell} python
Lambda = -2 * (lnL_B - lnL_A)
print(f"LLR Λ = -2(ln L_B - ln L_A) = {Lambda:.3f}")
```

A positive $\Lambda$ means Model A fits better.

### 4c — ΔAIC

```{code-cell} python
k_A = k_B = 0   # no free parameters
AIC_A = 2*k_A - 2*lnL_A
AIC_B = 2*k_B - 2*lnL_B
dAIC = AIC_B - AIC_A
print(f"ΔAIC = AIC_B - AIC_A = {dAIC:.3f}")
print(f"LLR  = {Lambda:.3f}  (should equal ΔAIC when k_A = k_B = 0)")
```

When both models have $k=0$ free parameters, $\Delta\text{AIC} = -2(\ln\mathcal{L}_B - \ln\mathcal{L}_A) = \Lambda$ exactly.

### 4d — Summary

Model A ($\mu = 3.15$ GeV) is preferred. The LLR value indicates how decisively — values above ~4 suggest strong preference, while values below ~1 suggest weak evidence. Given the dataset size, the conclusion should be clear; a $\Delta\text{AIC} > 4$ is conventionally taken as "considerable" evidence in favour of the lower-AIC model.

