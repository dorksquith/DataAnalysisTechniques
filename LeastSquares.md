---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(chapter:LeastSquares)=
# Least Squares

Least squares (LS) is a widely used method for fitting a model to data.  The idea is to find the parameter values that minimise the sum of squared differences between the data and the model — the *residuals*.  As we will see, when the measurement uncertainties are Gaussian, minimising the sum of squared residuals is identical to [maximum likelihood estimation](chapter:MaxLike).

(sec:ls-chisquared)=
## The chi-squared statistic

Given $N$ measurements $y_i$ with uncertainties $\sigma_i$, and a model prediction $\mu_i(\theta)$ depending on parameters $\theta$, we define

```{math}
:label: eq:chisq
\chi^2 = \sum_{i=1}^{N} \frac{(y_i - \mu_i)^2}{\sigma_i^2}.
```

This is sometimes called the *weighted sum of squared residuals* or Pearson's chi-squared.

**Connection to the Gaussian log-likelihood.** If each $y_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$ then the log-likelihood is

$$
\ln \mathcal{L}_\theta = -\frac{N}{2}\ln(2\pi) - \sum_i \ln\sigma_i - \frac{1}{2}\chi^2.
$$

Maximising $\ln \mathcal{L}_\theta$ over $\theta$ is therefore *identical* to minimising $\chi^2$.  If all uncertainties are equal (the *homoscedastic* case) the $\sigma_i$ terms are constant, so $\hat\theta_{\rm LS} = \hat\theta_{\rm MLE}$.

> **Note:** If we also optimise over the $\sigma_i$ individually the MLE gives $\hat\sigma_i = 0$ and $\chi^2 = 0$, which is trivially perfect but useless.  Least squares is interesting when the $\sigma_i$ are fixed by the measurement process.

(sec:ls-binned)=
## Chi-squared with binned data

When data are binned (e.g.\ a histogram), the observed count $n_i$ in each bin follows a Poisson distribution.  Since $\mathbb{V}[K] = K$ for a Poisson RV, we use $\sigma_i^2 = n_i$ and the chi-squared becomes

```{math}
:label: eq:chisq-binned
\chi^2 = \sum_{\rm bins} \frac{(n_i - \hat{n}_i)^2}{n_i},
```

where $\hat{n}_i$ is the model prediction for bin $i$.

(sec:ls-straightline)=
## Fitting a straight line

The workhorse application is fitting $y = mx + c$ to data $(x_i, y_i)$ with uncertainties $\sigma_i$.  We minimise

```{math}
:label: eq:chisq-line
\chi^2 = \sum_{i=1}^{N} \frac{(y_i - mx_i - c)^2}{\sigma_i^2}.
```

Setting the partial derivatives to zero gives two simultaneous equations:

$$
\frac{\partial \chi^2}{\partial m} = -2\sum_i \frac{x_i(y_i - mx_i - c)}{\sigma_i^2} = 0,
\qquad
\frac{\partial \chi^2}{\partial c} = -2\sum_i \frac{y_i - mx_i - c}{\sigma_i^2} = 0.
$$

(sec:ls-shorthand)=
### Shorthand notation

Define the sums

```{math}
:label: eq:ls-sums
s_1 = \sum_i \frac{1}{\sigma_i^2}, \quad
s_x = \sum_i \frac{x_i}{\sigma_i^2}, \quad
s_{xx} = \sum_i \frac{x_i^2}{\sigma_i^2}, \quad
s_y = \sum_i \frac{y_i}{\sigma_i^2}, \quad
s_{xy} = \sum_i \frac{x_i y_i}{\sigma_i^2}.
```

The two simultaneous equations become

$$
s_{xx}\,\hat m + s_x\,\hat c = s_{xy}, \qquad s_x\,\hat m + s_1\,\hat c = s_y.
$$

(sec:ls-solution)=
### Solution by Cramer's rule

Define $\Delta = s_{xx}s_1 - s_x^2$.  Solving by Cramer's rule gives

```{math}
:label: eq:ls-mhat
\hat m = \frac{s_{xy}s_1 - s_x s_y}{\Delta},
```

```{math}
:label: eq:ls-chat
\hat c = \frac{s_{xx}s_y - s_x s_{xy}}{\Delta}.
```

In the **homoscedastic** case ($\sigma_i = \sigma$ for all $i$), the common $1/\sigma^2$ factors cancel throughout and the sums simplify: $s_1 = N$, $s_x = \sum x_i$, etc.

(sec:ls-uncertainties)=
## Parameter uncertainties

To find uncertainties we Taylor-expand $\chi^2$ about the minimum $(\hat m,\hat c)$.  Since $\partial\chi^2/\partial\theta\vert_{\hat\theta}=0$ the first-order term vanishes and

```{math}
:label: eq:chisq-taylor
\chi^2(m,c) = \chi^2(\hat m,\hat c) +
\begin{pmatrix} m-\hat m \\ c-\hat c \end{pmatrix}^{\!\top}
\mathbf{H}
\begin{pmatrix} m-\hat m \\ c-\hat c \end{pmatrix},
```

where the **Hessian** is

```{math}
:label: eq:ls-hessian
\mathbf{H} = \begin{pmatrix} s_{xx} & s_x \\ s_x & s_1 \end{pmatrix}.
```

The **covariance matrix** of the estimates is $\boldsymbol{\Sigma} = \mathbf{H}^{-1}$, giving

```{math}
:label: eq:ls-cov
\sigma^2_{\hat m} = \frac{s_1}{\Delta}, \qquad
\sigma^2_{\hat c} = \frac{s_{xx}}{\Delta}, \qquad
\operatorname{cov}(\hat m,\hat c) = -\frac{s_x}{\Delta}.
```

The *linear correlation coefficient* between $\hat m$ and $\hat c$ is

$$
\rho(\hat m,\hat c) = \frac{\operatorname{cov}(\hat m,\hat c)}{\sigma_{\hat m}\,\sigma_{\hat c}}.
$$

A negative correlation is typical: if the slope is steeper than expected, the intercept tends to be lower to compensate.

(sec:ls-matrix)=
## Matrix generalization

The straight-line formulae extend to any model that is **linear in its parameters**, including polynomials and nonlinear bases such as $y = \theta_1 x^2 + \theta_2 x^3 + \theta_3 x^{-1}$.  Write

$$
y = \sum_j \theta_j f_{x,j} \quad \Longleftrightarrow \quad \mathbf{y} = \mathbf{f}_x \,\boldsymbol\theta,
$$

where $\mathbf{f}_x$ is a $[Z \times N]$ matrix ($Z$ = number of parameters, $N$ = number of measurements).  The chi-squared is

```{math}
:label: eq:chisq-matrix
\chi^2_{\rm rel} = (\mathbf{y} - \mathbf{f}_x\boldsymbol\theta)^{\top} \mathbf{H} (\mathbf{y} - \mathbf{f}_x\boldsymbol\theta).
```

Minimising over $\boldsymbol\theta$ yields the **normal equations**

$$
(\mathbf{f}_x^{\top}\mathbf{H}\mathbf{f}_x)\,\boldsymbol{\hat\theta} = (\mathbf{f}_x^{\top}\mathbf{H})\,\mathbf{y},
$$

so $\boldsymbol{\hat\theta} = \mathbf{B}\mathbf{y}$ with $\mathbf{B} = (\mathbf{f}_x^{\top}\mathbf{H}\mathbf{f}_x)^{-1}\mathbf{f}_x^{\top}\mathbf{H}$, and the covariance matrix is

```{math}
:label: eq:ls-cov-matrix
\boldsymbol{\Sigma} = \mathbf{B}\mathbf{H}\mathbf{B}^{\top} = (\mathbf{f}_x^{\top}\mathbf{H}\mathbf{f}_x)^{-1}.
```

(sec:ls-gauss-markov)=
## Ordinary Least Squares and the Gauss–Markov theorem

**Ordinary Least Squares (OLS)** refers to functions that are linear in their parameters (the basis functions $f_x$ need not be linear in $x$).  **Nonlinear Least Squares (NLS)** is for functions that are nonlinear in their parameters.

The **Gauss–Markov theorem** states: if the measurement uncertainties are homoscedastic and serially uncorrelated then the OLS estimator is the **Minimum Variance Unbiased Estimator (MVUE)** — it achieves the lowest possible variance among all linear unbiased estimators.  If the uncertainties are additionally Gaussian, OLS and MLE are identical.

If the data are **heteroscedastic** (different $\sigma_i$), we must either transform the data or use *Weighted Least Squares*, which keeps the $1/\sigma_i^2$ weights non-trivially.

(sec:ls-scipy)=
## Numerical implementation with scipy

`scipy.optimize.least_squares` fits a model by minimising the sum of squares of a user-supplied *residuals* function.

### Uncertainties from the Jacobian

At the minimum, the Hessian is approximated by $\mathbf{H} \approx \mathbf{J}^\top\mathbf{J}$, where $\mathbf{J}$ is the Jacobian of the residuals.  `least_squares` returns `res.jac` directly.  The covariance is

```{math}
:label: eq:ls-cov-jac
\boldsymbol{\Sigma} = (\mathbf{J}^\top\mathbf{J})^{-1} \cdot \frac{\chi^2}{\nu},
```

where $\nu = N - Z$ is the number of degrees of freedom and $\chi^2/\nu$ is the *reduced chi-squared* (which estimates $\sigma^2$ in the homoscedastic case).

```{code-cell} python
import numpy as np
from scipy.optimize import least_squares
from numpy.random import default_rng

# --- generate fake data y = mx + c with Gaussian noise ---
m_true = -3.0
c_true =  2.0
x_vals = np.linspace(0, 10, 10)
noise  = 0.1

rng = default_rng(123)
y_vals = m_true * x_vals + c_true + noise * rng.standard_normal(x_vals.size)

# --- residuals function: p = [c, m] ---
def residuals(p, x, y):
    return p[1] * x + p[0] - y

N_params = 2
N_dof = len(y_vals) - N_params

# --- fit ---
p0 = np.array([1.0, 1.0])
res_ls = least_squares(residuals, p0, args=(x_vals, y_vals))

# --- uncertainties via Jacobian ---
sumsq_res    = np.sum(res_ls.fun**2)
reduced_chisq = sumsq_res / N_dof          # estimates sigma^2

J    = res_ls.jac
H    = J.T @ J
Hinv = np.linalg.inv(H)
cov  = Hinv * reduced_chisq               # covariance matrix

var   = np.diagonal(cov)
sigma = var**0.5
covmc = cov[0, 1]
rhomc = covmc / (sigma[0] * sigma[1])

print(f"Intercept  c: {res_ls.x[0]:.4f}  ±  {sigma[0]:.4f}")
print(f"Slope      m: {res_ls.x[1]:.4f}  ±  {sigma[1]:.4f}")
print(f"cov(c,m):  {covmc:.4f}")
print(f"rho(c,m):  {rhomc:.4f}")
```

### By-hand calculation (homoscedastic case)

When all $\sigma_i$ are equal, $s_1 = N$ and the sums simplify.  The uncertainty on $y$ is estimated from the residuals.

```{code-cell} python
def byhand_straightline(x_vals, y_vals):
    N   = len(y_vals)
    s1  = N
    sxx = np.sum(x_vals**2)
    sx  = np.sum(x_vals)
    sy  = np.sum(y_vals)
    sxy = np.sum(x_vals * y_vals)

    delta = N * sxx - sx**2

    m_est = (sxy * N  - sx * sy)  / delta
    c_est = (sxx * sy - sxy * sx) / delta

    resids    = m_est * x_vals + c_est - y_vals
    sigma_y2  = np.sum(resids**2) / (N - 2)

    sigma_m = np.sqrt(sigma_y2 * N    / delta)
    sigma_c = np.sqrt(sigma_y2 * sxx  / delta)
    covmc   = sigma_y2 * (-sx / delta)
    rhomc   = covmc / (sigma_c * sigma_m)

    print(f"Intercept  c: {c_est:.4f}  ±  {sigma_c:.4f}")
    print(f"Slope      m: {m_est:.4f}  ±  {sigma_m:.4f}")
    print(f"cov(c,m):  {covmc:.4f}")
    print(f"rho(c,m):  {rhomc:.4f}")

byhand_straightline(x_vals, y_vals)
```

> **Important:** The covariance matrix returned by `scipy.optimize.least_squares` is *not automatically scaled*.  You must multiply `(J^T J)^{-1}` by the reduced chi-squared `sumsq_res/N_dof` to get the true covariance.

(sec:ls-outliers)=
## Extreme outliers and robust least squares

Because the residuals appear **squared** in $\chi^2$, a single outlier can dominate the fit and pull the estimated parameters far from their true values.

**Robust Least Squares** replaces the squared residuals with a loss function that is less sensitive to large deviations.  The *Least Absolute Deviations* (LAD) method uses $|y_i - \mu_i|$ instead of $(y_i - \mu_i)^2$.  `scipy.optimize.least_squares` provides the `loss` argument with several built-in options:

| `loss`     | Effect |
|-----------|--------|
| `'linear'` | Standard LS (default) |
| `'soft_l1'`| Smooth approximation to LAD |
| `'huber'`  | Quadratic near zero, linear in the tails |
| `'cauchy'` | Very robust, maximally down-weights outliers |

The `f_scale` parameter sets the transition scale between the near-quadratic and near-linear regimes.

```{code-cell} python
from scipy.optimize import least_squares
from numpy.random import default_rng
import numpy as np

# quadratic model y = a*x + b*x^2 with two outliers
a, b = 0.5, 2.0
x_vals = np.linspace(-10, 10, 15)
noise  = 0.1

rng = default_rng(123)
y = a * x_vals + b * x_vals**2
y_noise = noise * rng.standard_normal(x_vals.size)
outliers = rng.integers(0, x_vals.size, 2)
y_noise[outliers] *= 10_000          # extreme outliers
y_vals = y + y_noise

def residuals_quad(p, x, y):
    return p[0] * x + p[1] * x**2 - y

p0 = np.array([1.0, 1.0])

# standard LS
res_std  = least_squares(residuals_quad, p0, args=(x_vals, y_vals))
# robust soft_l1
res_rob  = least_squares(residuals_quad, p0, args=(x_vals, y_vals),
                         loss='soft_l1', f_scale=noise)

print("True:       a={:.2f}, b={:.2f}".format(a, b))
print("Standard LS: a={:.2f}, b={:.2f}".format(*res_std.x))
print("soft_l1 LS:  a={:.2f}, b={:.2f}".format(*res_rob.x))
```

(sec:ls-nonlinear)=
## Nonlinear models

When the model is **nonlinear in its parameters** — for example $y = a + b\,e^{cx}$ — the normal equations no longer have a closed-form solution.  Two common iterative strategies are:

**Simplex (Nelder–Mead).** Explores the parameter space by maintaining a simplex of $Z+1$ points and iteratively reflecting, expanding or contracting it toward the minimum.  Does not require gradient information.

**Gradient Search.** Uses the gradient $\nabla_\theta\chi^2$ to step downhill:

$$
\boldsymbol\theta_{j+1} = \boldsymbol\theta_j - \gamma\,\nabla_\theta\chi^2,
$$

where $\gamma$ is the step size.  Choosing $\gamma$ too small makes convergence slow; too large and the algorithm can overshoot or oscillate.

`scipy.optimize.least_squares` handles nonlinear models transparently through the same interface:

```{code-cell} python
from scipy.optimize import least_squares
from numpy.random import default_rng
import numpy as np

# nonlinear model y = a + b * exp(c*x)
a_true, b_true, c_true = 0.5, 2.0, -0.5
x_vals = np.linspace(0, 10, 15)
noise  = 0.1

rng = default_rng(123)
y_vals = a_true + b_true * np.exp(c_true * x_vals) \
         + noise * rng.standard_normal(x_vals.size)

def residuals_exp(p, x, y):
    return p[0] + p[1] * np.exp(p[2] * x) - y

p0 = np.array([1.0, 1.0, 1.0])

# soft_l1 is a good choice for nonlinear models too
res = least_squares(residuals_exp, p0, args=(x_vals, y_vals),
                    loss='soft_l1', f_scale=noise)

print("True:    a={:.2f}, b={:.2f}, c={:.2f}".format(a_true, b_true, c_true))
print("Fitted:  a={:.2f}, b={:.2f}, c={:.2f}".format(*res.x))
```

> **Pitfalls of nonlinear fitting:** the algorithm can get stuck in local minima, or miss shallow turning points if $\gamma$ is chosen poorly.  Always try multiple starting points $\boldsymbol\theta_0$ and inspect the residuals visually.

For models whose uncertainties are non-Gaussian, a **Generalized Linear Model (GLM)** is more appropriate — see `statsmodels.genmod`.

(sec:ls-summary)=
## Summary

The key results from this chapter are:

Least squares minimises $\chi^2 = \sum_i (y_i - \mu_i)^2/\sigma_i^2$, which is equivalent to Gaussian MLE.

For a straight line $y = mx + c$, the analytical estimators are $\hat m = (s_{xy}s_1 - s_x s_y)/\Delta$ and $\hat c = (s_{xx}s_y - s_x s_{xy})/\Delta$ with $\Delta = s_{xx}s_1 - s_x^2$.

Parameter uncertainties come from the covariance matrix $\boldsymbol\Sigma = \mathbf{H}^{-1}$, giving $\sigma^2_{\hat m} = s_1/\Delta$, $\sigma^2_{\hat c} = s_{xx}/\Delta$, $\operatorname{cov}(\hat m,\hat c) = -s_x/\Delta$.

In `scipy.optimize.least_squares` the covariance is reconstructed as $(\mathbf{J}^\top\mathbf{J})^{-1} \times \chi^2/\nu$.

If data are heteroscedastic, or if outliers are present, standard OLS is no longer optimal and robust methods or weighted LS should be used.
