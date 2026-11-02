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

(workshop-4-solutions)=
# Workshop 4 Solutions: Z Values, P Values, Oh My!

---

## Task 1: True or False

1. **False.** A p-value of 7% is *greater* than $\alpha = 5\%$, so we fail to reject the null hypothesis — we do not "confirm" it. Failing to reject is not the same as confirming; it simply means the data are not sufficiently surprising under $H_0$.

2. **True.** A p-value of 1% is less than $\alpha = 2\%$, so we reject the null hypothesis at this significance level.

3. **False.** The p-value is the probability of obtaining a result as extreme as (or more extreme than) the observed data, *assuming the null hypothesis is true*. It is **not** the probability that your conclusions are wrong, nor the probability that $H_0$ is true.

4. **True.** The False Positive rate (also called the Type I error rate) is $P(\text{reject } H_0 \mid H_0 \text{ true}) = \alpha$, by construction of the test.

5. **False.** The False Positive rate and True Positive rate are conditional on *different* hypotheses — FP on $H_0$ true, TP on $H_0$ false — so they need not sum to anything in particular.

6. **True.** The False Positive rate and True Negative rate are both conditioned on $H_0$ being true: $P(\text{reject} \mid H_0) + P(\text{fail to reject} \mid H_0) = 1$.

7. **True.** A hypothesis test is defined by: a null hypothesis, a significance level $\alpha$, and a critical region. Without specifying the critical region (one- or two-sided, which test statistic) the test is not fully defined.

8. **False.** Looking at the data before designing the test introduces bias even for an honest person. The human brain is pattern-seeking; unconsciously we are more likely to set a critical region that our data will cross. This is a form of p-hacking sometimes called "optional stopping" or "researcher degrees of freedom." Pre-registration of the analysis plan (before unblinding) is the standard remedy.

---

## Task 2: CDF, Z Value, P Value

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu, sigma = 1, 2
dist = norm(loc=mu, scale=sigma)
```

### 2a — CDF plot

```{code-cell} python
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)

fig, ax = plt.subplots()
ax.plot(x, dist.cdf(x), color='steelblue')
ax.set_xlabel("$x$")
ax.set_ylabel("CDF")
ax.set_title(rf"CDF of $\mathcal{{N}}(\mu={mu},\,\sigma={sigma})$")
ax.axhline(0.5, linestyle=':', color='grey', linewidth=0.8)
plt.tight_layout()
plt.show()
```

### 2b — Critical values for a two-sided 5% test

For a two-sided test with $\alpha = 5\%$, we place $2.5\%$ in each tail:

$$x_{\rm lo} = F^{-1}(0.025), \qquad x_{\rm hi} = F^{-1}(0.975).$$

```{code-cell} python
x_lo = dist.ppf(0.025)
x_hi = dist.ppf(0.975)
print(f"Critical values:  x_lo = {x_lo:.3f},  x_hi = {x_hi:.3f}")
print(f"(These are μ ± 1.96σ = {mu - 1.96*sigma:.3f}, {mu + 1.96*sigma:.3f})")
```

The critical region is $x < x_{\rm lo}$ or $x > x_{\rm hi}$.

### 2c — P value for $x = -4.5$

The p-value is the probability, under $H_0$, of observing a result at least as extreme as $x = -4.5$.
For a two-sided test:

$$p = 2\,P(X \le -4.5) = 2\,F(-4.5).$$

```{code-cell} python
x_obs = -4.5
p_val = 2 * dist.cdf(x_obs)
print(f"p-value = {p_val:.4f}  ({100*p_val:.2f}%)")
print("Reject H0 at 5% level:", p_val < 0.05)
```

### 2d — Z value

The Z value expresses the measurement in units of standard deviations from the mean:

$$Z = \frac{x - \mu}{\sigma} = \frac{-4.5 - 1}{2} = -2.75\,\sigma.$$

```{code-cell} python
Z = (x_obs - mu) / sigma
print(f"Z = {Z:.2f} sigma")
```

### 2e — Probability within $5\sigma$

```{code-cell} python
prob_5sigma = dist.cdf(mu + 5*sigma) - dist.cdf(mu - 5*sigma)
print(f"P(|X - μ| < 5σ) = {prob_5sigma:.7f}")
print(f"Probability outside 5σ: {1 - prob_5sigma:.2e}")
```

Roughly $1 - 5.7\times10^{-7}$ of all measurements fall within $5\sigma$ of the mean.

### 2f — Interpretation of a measurement outside $5\sigma$

A result outside the $5\sigma$ region occurs with probability $\approx 5.7\times 10^{-7}$ under $H_0$.
This is overwhelmingly unlikely as a random fluctuation, so we would strongly reject $H_0$.

In particle physics, $5\sigma$ is the conventional discovery threshold: it provides strong protection against claiming a discovery that is actually a statistical fluctuation.
In practice, finding a result far outside $5\sigma$ may also prompt investigation of systematic effects — detector miscalibration, analysis errors, or data quality issues — before concluding that the null hypothesis is wrong.

---

## Task 3: Optional Extension — Bivariate Gaussian

### 3a — Constraints on the covariance matrix

A valid covariance matrix must be **symmetric positive semi-definite**. For a $2\times 2$ matrix

$$\Sigma = \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix},$$

the constraints are: $\sigma_1^2 > 0$, $\sigma_2^2 > 0$, and the off-diagonal element must satisfy $-1 < \rho < 1$ (i.e. $|\mathsf{cov}(X_1, X_2)| < \sigma_1\sigma_2$).  Equivalently, the determinant must be non-negative: $\sigma_1^2\sigma_2^2 - [\mathsf{cov}(X_1, X_2)]^2 \ge 0$.

### 3b — Contours at $1\sigma, 2\sigma, 3\sigma$

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, multivariate_normal

mu2   = [0, 0]
cov2  = [[1, 0.7], [0.7, 1]]
rv    = multivariate_normal(mean=mu2, cov=cov2)

x_grid = np.linspace(-4, 4, 200)
y_grid = np.linspace(-4, 4, 200)
X, Y  = np.meshgrid(x_grid, y_grid)
pos   = np.dstack((X, Y))
Z     = rv.pdf(pos)

# The probability enclosed within k-sigma for a 2D Gaussian is 1 - exp(-k^2/2),
# which corresponds to the chi2(2) quantile at that probability level.
# The PDF contour at that level: f(x) = f_max * exp(-k^2/2)
f_max = rv.pdf(mu2)
levels = [f_max * np.exp(-k**2 / 2) for k in [1, 2, 3]]

fig, ax = plt.subplots(figsize=(5, 5))
cs = ax.contour(X, Y, Z, levels=sorted(levels), colors=['C0', 'C1', 'C2'])
ax.clabel(cs, fmt={v: f'{k}σ' for v, k in zip(sorted(levels), [3, 2, 1])}, inline=True)
ax.set_aspect('equal')
ax.set_xlabel("$X_1$")
ax.set_ylabel("$X_2$")
ax.set_title("Bivariate Gaussian — 1, 2, 3σ contours")
plt.tight_layout()
plt.show()
```

Note: for a **2D** Gaussian the probability enclosed within the $k\sigma$ ellipse is $1 - e^{-k^2/2}$, not the 1D values (68%, 95%, 99.7%). At $1\sigma$ this gives approximately 39%; at $2\sigma$ approximately 86%.

### 3c — 3D surface plot

```{code-cell} python
fig = plt.figure(figsize=(7, 5))
ax3 = fig.add_subplot(111, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, linewidth=0)
ax3.set_xlabel("$X_1$")
ax3.set_ylabel("$X_2$")
ax3.set_zlabel("PDF")
ax3.set_title("Bivariate Gaussian PDF")
plt.tight_layout()
plt.show()
```
