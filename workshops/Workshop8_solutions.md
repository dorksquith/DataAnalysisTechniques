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

(workshop-8-solutions)=
# Workshop 8 Solutions: Linear Regression and Model Comparison

---

## Task 1: True or False

1. **True.** `curve_fit` minimises $\sum_i[(y_i - f(x_i))/\sigma_i]^2$ when `sigma` and `absolute_sigma=True` are supplied.

2. **True.** The diagonal of `pcov` gives the estimated variances; take `np.sqrt(np.diag(pcov))` for standard errors.

3. **False.** For Poisson data the correct goodness-of-fit measure uses the Poisson log-likelihood (or the Pearson $\chi^2$ with $M_i$ in the denominator, not $\sigma_i^2$); the unweighted $\chi^2$ ignores the Poisson variance structure.

4. **False.** They are related but not numerically equal; the Pearson $\chi^2$ is an approximation to the LLR that works well in the large-count limit.

5. **False.** AIC penalises extra parameters ($+2k$), so a more complex model can have a *higher* AIC if the improvement in log-likelihood is small.

6. **False.** A $\Delta\text{AIC}$ of 2 is conventionally the threshold for *weak/positive* evidence; values above 4–6 are considered substantial.

---

## Task 2: Linear Regression

### 2a — Data plot

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x, y, sigma = np.loadtxt('DataW8_regression.txt', skiprows=1, unpack=True)

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma, fmt='o', color='steelblue', capsize=4, label='Data')
ax.set_xlabel('Position $x$ (cm)')
ax.set_ylabel('Force $y$ (N)')
ax.set_title('Force vs Position')
ax.legend()
plt.tight_layout()
plt.show()
```

### 2b — Linear fit

```{code-cell} python
def linear(x, m, c):
    return m * x + c

popt, pcov = curve_fit(linear, x, y, sigma=sigma, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
m_hat, c_hat = popt
print(f"Slope:     m̂ = {m_hat:.4f} ± {perr[0]:.4f}")
print(f"Intercept: ĉ = {c_hat:.4f} ± {perr[1]:.4f}")
```

### 2c — Best-fit line

```{code-cell} python
x_fit = np.linspace(x.min() - 0.3, x.max() + 0.3, 200)
y_fit = linear(x_fit, *popt)

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma, fmt='o', color='steelblue', capsize=4, label='Data')
ax.plot(x_fit, y_fit, 'r-', label=rf'$y = {m_hat:.2f}x + {c_hat:.2f}$')
ax.set_xlabel('Position $x$ (cm)')
ax.set_ylabel('Force $y$ (N)')
ax.legend()
plt.tight_layout()
plt.show()
```

### 2d — Residuals

```{code-cell} python
y_pred = linear(x, *popt)
residuals = (y - y_pred) / sigma

fig, ax = plt.subplots()
ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax.scatter(x, residuals, color='steelblue')
ax.set_xlabel('Position $x$ (cm)')
ax.set_ylabel('Weighted residual $(y - \hat y)/\sigma$')
ax.set_title('Residuals')
plt.tight_layout()
plt.show()
```

Residuals should scatter randomly around zero with no clear trend if the linear model is appropriate.

### 2e — Reduced chi-squared

```{code-cell} python
k_lin = 2
chi2_lin = np.sum(residuals**2)
nu_lin = len(x) - k_lin
chi2_nu_lin = chi2_lin / nu_lin
print(f"χ²_ν (linear) = {chi2_nu_lin:.3f}  (N={len(x)}, k={k_lin}, dof={nu_lin})")
```

A value near 1 indicates a good fit. Values much greater than 1 suggest underfit; much less than 1 may indicate over-estimated uncertainties.

---

## Task 3: Model Comparison for Binned Counts

### 3a — Data and model plot

```{code-cell} python
from scipy.special import gammaln

bc, D_float, M1, M2 = np.loadtxt('DataW8_models.txt', skiprows=1, unpack=True)
D = D_float.astype(int)

fig, ax = plt.subplots()
ax.step(bc, D, where='mid', color='k', label='Data $D$')
ax.plot(bc, M1, 'b-o', ms=4, label='Model 1')
ax.plot(bc, M2, 'r-s', ms=4, label='Model 2')
ax.set_xlabel('Bin centre')
ax.set_ylabel('Counts')
ax.set_title('Binned counts: data vs models')
ax.legend()
plt.tight_layout()
plt.show()
```

### 3b — Poisson log-likelihoods

```{code-cell} python
def poisson_log_likelihood(D, M):
    return np.sum(D * np.log(M) - M - gammaln(D + 1))

lnL1 = poisson_log_likelihood(D, M1)
lnL2 = poisson_log_likelihood(D, M2)
print(f"ln L₁ = {lnL1:.3f}")
print(f"ln L₂ = {lnL2:.3f}")
print("Model 1 preferred" if lnL1 > lnL2 else "Model 2 preferred")
```

### 3c — LLR and ΔAIC

```{code-cell} python
Lambda = -2 * (lnL2 - lnL1)
k1 = k2 = 0
dAIC = (2*k2 - 2*lnL2) - (2*k1 - 2*lnL1)
print(f"LLR  Λ  = {Lambda:.3f}")
print(f"ΔAIC    = {dAIC:.3f}  (equals LLR since k₁=k₂=0)")
```

A positive $\Lambda$ means Model 1 fits better. Since $k_1 = k_2 = 0$, $\Delta\text{AIC} = \Lambda$ exactly.

### 3d — Pearson chi-squared

```{code-cell} python
chi2_1 = np.sum((D - M1)**2 / M1)
chi2_2 = np.sum((D - M2)**2 / M2)
n_bins = len(D)
print(f"χ² (Model 1) = {chi2_1:.2f}  (expected ≈ {n_bins})")
print(f"χ² (Model 2) = {chi2_2:.2f}  (expected ≈ {n_bins})")
```

### 3e — Summary

Model 1 has a higher log-likelihood, lower AIC, and lower Pearson $\chi^2$ than Model 2, meaning it is the preferred description of the data. The LLR quantifies the strength of the evidence: a value above ~4 is substantial. The Pearson $\chi^2$ check tells us whether either model is plausible in absolute terms — if $\chi^2 \approx n_\text{bins}$, the model is a reasonable fit; if $\chi^2 \gg n_\text{bins}$, neither model is adequate. In this case, Model 1 is preferred and likely provides an acceptable fit.

---

## Task 4: Quadratic Extension

### 4a — Quadratic fit

```{code-cell} python
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

popt_q, pcov_q = curve_fit(quadratic, x, y, sigma=sigma, absolute_sigma=True)
perr_q = np.sqrt(np.diag(pcov_q))
a_hat, b_hat, c_hat_q = popt_q
print(f"a = {a_hat:.4f} ± {perr_q[0]:.4f}")
print(f"b = {b_hat:.4f} ± {perr_q[1]:.4f}")
print(f"c = {c_hat_q:.4f} ± {perr_q[2]:.4f}")
```

### 4b — Reduced chi-squared comparison

```{code-cell} python
y_pred_q = quadratic(x, *popt_q)
residuals_q = (y - y_pred_q) / sigma
k_quad = 3
chi2_quad = np.sum(residuals_q**2)
chi2_nu_quad = chi2_quad / (len(x) - k_quad)
print(f"χ²_ν (linear)    = {chi2_nu_lin:.3f}")
print(f"χ²_ν (quadratic) = {chi2_nu_quad:.3f}")
```

### 4c — ΔAIC

```{code-cell} python
# For Gaussian likelihood with known sigma: -2 ln L ∝ chi2_weighted + const
# AIC = 2k + chi2_weighted (up to a dataset-constant)
AIC_lin  = 2*k_lin  + chi2_lin
AIC_quad = 2*k_quad + chi2_quad
dAIC_lq = AIC_quad - AIC_lin
print(f"ΔAIC = AIC_quad - AIC_lin = {dAIC_lq:.3f}")
```

### 4d — Interpretation

If $|\Delta\text{AIC}| < 2$, neither model is decisively preferred. If $\Delta\text{AIC} > 0$ (linear lower), the quadratic's improvement in fit did not justify its extra parameter. Visual inspection of the residuals from the linear fit is the clearest guide: if they are random with no curvature, the linear model is sufficient.

