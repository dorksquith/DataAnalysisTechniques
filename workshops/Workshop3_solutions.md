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

(workshop-3-solutions)=
# Workshop 3 Solutions: Covariance & Correlations

---

## Task 1: Covariance of Two Datasets

### 1a — Mean, variance, and standard deviation

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

X1 = np.array([18_841, 20_449, 20_987, 21_854, 22_778,
               24_075, 25_253, 26_155, 27_227, 27_092])

X2 = np.array([11, 16.3333, 23.75, 29.8333, 34.3333,
               42.25, 47.5, 53.25, 57.0833, 58.1667])

print("X1  mean: {:.2f},  var: {:.2f},  sigma: {:.2f}".format(
      np.mean(X1), np.var(X1), np.std(X1)))
print("X2  mean: {:.4f},  var: {:.4f},  sigma: {:.4f}".format(
      np.mean(X2), np.var(X2), np.std(X2)))
```

### 1b — Covariance matrix

Always plot the data first.

```{code-cell} python
fig, ax = plt.subplots()
ax.scatter(X1, X2, color='skyblue', marker='*', label='Data')
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.legend()
plt.show()

cov_mat = np.cov(X1, X2)
print("\nCovariance matrix:\n", cov_mat)
```

Note: `np.cov` uses the unbiased $1/(N-1)$ normalisation, whereas `np.var` defaults to $1/N$.
The diagonal of the covariance matrix therefore matches `np.var(Xi, ddof=1)`, not `np.var(Xi)`.

```{code-cell} python
print(f"\ncov_mat[0,0]       = {cov_mat[0,0]:.4f}")
print(f"np.var(X1, ddof=1) = {np.var(X1, ddof=1):.4f}  ← matches")
print(f"np.var(X1)         = {np.var(X1):.4f}  ← does not match")

cc_mat = np.corrcoef(X1, X2)
print("\nCorrelation matrix:\n", cc_mat)
print(f"\nLinear correlation coefficient: {cc_mat[0,1]:.4f}")
```

### 1c — Interpretation

The correlation coefficient is very close to $+1$, indicating a strong positive linear relationship between X1 and X2.

However, correlation does not imply causation. These data are a classic example from [Spurious Correlations](https://www.tylervigen.com/spurious/correlation/):

- **X1**: number of US bachelor's degrees awarded in mathematics and statistics per year
- **X2**: Google searches for "reddit" (relative to a baseline), over the same years

Both variables happen to trend upward over the same period for entirely unrelated reasons. The high correlation is coincidental.

---

## Task 2: Linear Correlations

### 2a–c — Energy vs speed

```{code-cell} python
from scipy.stats import uniform

uniform_dist = uniform(-160, 320)   # Uniform(-160, 160)
speeds   = uniform_dist.rvs(size=100, random_state=42)
energies = 0.5 * speeds**2

fig, ax = plt.subplots()
ax.scatter(speeds, energies, alpha=0.6)
ax.set_xlabel("Speed $v$")
ax.set_ylabel("Kinetic energy $E = \\frac{1}{2}v^2$")
ax.set_xlim(-160, 160)
plt.show()

cov_mat_ev = np.cov(energies, speeds)
cc_mat_ev  = np.corrcoef(energies, speeds)
print("Covariance matrix:\n", cov_mat_ev)
print(f"\nLinear correlation coefficient ρ(E, v) = {cc_mat_ev[0,1]:.4f}")
```

### 2d — Interpretation

The linear correlation coefficient $\rho(E, v)$ is close to zero, despite the obvious parabolic structure in the scatter plot.

This is expected: the relationship $E \propto v^2$ is **quadratic**, not linear, and the Pearson correlation coefficient only captures linear associations.
Because the speed distribution is symmetric about zero — positive and negative speeds are equally likely — the covariance

$$\mathsf{cov}(E, v) = E\!\left[\tfrac{1}{2}v^3\right] - E\!\left[\tfrac{1}{2}v^2\right]E[v]$$

has $E[v] = 0$ and $E[v^3] = 0$ (odd moments of a symmetric distribution vanish), so the true covariance is exactly zero.
The small non-zero value seen numerically is a finite-sample fluctuation.

The lesson: a near-zero linear correlation does **not** imply independence.

---

## Optional Math Task

**Claim.** For $f(\mathbf{x}) = \alpha_1 x_{(1)} + \alpha_2 x_{(2)} + \alpha_3 x_{(3)}$, the two formulae for $V[f]$ are identical.

**Method 1 — Expectation algebra:**

$$V[f] = \sum_{i=1}^{3}\sum_{j=1}^{3} \alpha_i\,\alpha_j\,\mathsf{cov}[x_{(i)}, x_{(j)}].$$

**Method 2 — Taylor expansion:**

$$V[f] = \sum_{i=1}^{3}\sum_{j=1}^{3}
         \frac{\partial f}{\partial x_{(i)}}\bigg|_{\mathbf{x}=\boldsymbol\mu}
         \frac{\partial f}{\partial x_{(j)}}\bigg|_{\mathbf{x}=\boldsymbol\mu}
         \mathsf{cov}[x_{(i)}, x_{(j)}].$$

Because $f$ is linear, its partial derivatives are exactly the coefficients:

$$\frac{\partial f}{\partial x_{(i)}} = \alpha_i \quad \text{(constant — evaluating at } \mathbf{x}=\boldsymbol\mu \text{ changes nothing)}.$$

Substituting into Method 2 immediately recovers Method 1. The two expressions are therefore identical for any linear function.

The Taylor-expansion approach is more general: for a nonlinear $f$ the partial derivatives are not constant, so they must be evaluated at the means before substituting.  For a linear $f$ this step is trivial, which is why the two methods agree exactly.
