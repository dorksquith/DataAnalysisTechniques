---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: 'Python 3'
---

(chapter:MaxLike)=
# Maximum Likelihood Estimation
<!--LectureSlides2025/MaxLike.tex-->
<!--PythonWorkshopsAndExamples2025/Week8-MathHelpers-and-MLE.py-->
<!--PythonWorkshopsAndExamples2025/MaximumLikelihood.py-->

<!--
Derive and compute MLE for Binomial, Exponential and Gaussian
Understand Bias, Variance and MSE of an MLE
Know how to estimate uncertainties on an MLE
Understand the Score, Fisher Information and Minimum Variance Bound
Use scipy to compute MLE numerically
-->

Given a set of data $\{x_1, x_2, \ldots, x_N\}$, we want to estimate the unknown parameter(s) $\theta$ of the distribution from which the data were drawn. **Maximum Likelihood Estimation** (MLE) answers: *which value of $\theta$ makes the observed data most probable?*

(sec:likelihood)=
## The Likelihood Function

Suppose the data are independent and identically distributed (iid), each drawn from a PDF (or PMF) $f_X(x;\theta)$. The joint probability of observing all $N$ values is the product of the individual probabilities:

```{math}
:label: eq:likelihood
\mathcal{L}_\theta = \prod_{i=1}^{N} f_X(x_i;\,\theta)
```

This is the **Likelihood** $\mathcal{L}_\theta$: a function of the parameter $\theta$, not of $x$.

The **Maximum Likelihood Estimate** $\hat{\theta}$ is the value of $\theta$ that maximises $\mathcal{L}_\theta$:

```{math}
:label: eq:mle-def
\hat{\theta} = \arg\max_\theta \mathcal{L}_\theta
```

### The Log-Likelihood

Products of many small numbers quickly underflow to zero numerically. The **log-likelihood** avoids this: because $\ln$ is monotonically increasing, maximising $\ln \mathcal{L}_\theta$ gives the same $\hat{\theta}$ as maximising $\mathcal{L}_\theta$:

```{math}
:label: eq:loglike
\ln \mathcal{L}_\theta = \sum_{i=1}^{N} \ln f_X(x_i;\,\theta)
```

In practice we almost always work with $\ln \mathcal{L}_\theta$. Numerical optimisers typically minimise, so we minimise the **negative log-likelihood** (NLL):

```{math}
:label: eq:nll
-\ln \mathcal{L}_\theta = -\sum_{i=1}^{N} \ln f_X(x_i;\,\theta)
```

(sec:mle-analytical)=
## Analytical MLE

To find $\hat{\theta}$ analytically we differentiate the log-likelihood and set the result to zero:

```{math}
:label: eq:score-zero
\frac{\partial \ln \mathcal{L}_\theta}{\partial \theta}\bigg|_{\hat{\theta}} = 0
```

We must also check that the second derivative is negative (confirming a maximum, not a minimum or saddle point).

(sec:mle-binomial)=
### Binomial: estimating $p$

Let $K \sim \mathrm{Binom}(n, p)$ and suppose we observe $k$ successes in $n$ trials. The PMF is

$$f_K(k;\,p) = \binom{n}{k} p^k (1-p)^{n-k}$$

so the log-likelihood (up to constants that don't depend on $p$) is

$$\ln \mathcal{L}_p = k \ln p + (n-k)\ln(1-p) + \text{const}$$

Setting $\partial \ln \mathcal{L}_p / \partial p = 0$:

$$\frac{k}{p} - \frac{n-k}{1-p} = 0 \implies \hat{p} = \frac{k}{n}$$

```{math}
:label: eq:mle-binom
\boxed{\hat{p} = \frac{k}{n}}
```

This is simply the observed fraction of successes — exactly what intuition suggests.

**Example.** A coin is tossed $n=50$ times and $k=27$ heads are observed. Then $\hat{p} = 27/50 = 0.54$.

(sec:mle-expon)=
### Exponential: estimating $\lambda$

Let $T \sim \mathrm{Expon}(\lambda)$ with PDF $f_T(t;\lambda)=\lambda e^{-\lambda t}$. For $N$ iid observations $\{t_i\}$:

$$\ln \mathcal{L}_\lambda = N\ln\lambda - \lambda\sum_{i=1}^N t_i$$

Setting $\partial \ln \mathcal{L}_\lambda / \partial \lambda = 0$:

$$\frac{N}{\lambda} - \sum t_i = 0 \implies \hat{\lambda} = \frac{N}{\sum t_i} = \frac{1}{\bar{t}}$$

```{math}
:label: eq:mle-expon
\boxed{\hat{\lambda} = \frac{1}{\bar{t}}}
```

The MLE rate is the reciprocal of the sample mean waiting time.

**Example (hitch-hiker).** Taxi waiting times (minutes): $t = \{15, 18, 11\}$. Then $\bar{t} = 44/3$, so $\hat{\lambda} = 3/44 \approx 0.068\;\mathrm{min}^{-1}$.

(sec:mle-gauss)=
### Gaussian: estimating $\mu$ and $\sigma$

Let $Y \sim N(\mu, \sigma^2)$ with PDF $f_Y(y;\mu,\sigma)= \frac{1}{\sigma\sqrt{2\pi}} e^{-(y-\mu)^2/(2\sigma^2)}$. For $N$ iid observations:

$$\ln \mathcal{L}_{\mu,\sigma} = -N\ln\sigma - N\ln\sqrt{2\pi} - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \mu)^2$$

**MLE for $\mu$**: setting $\partial \ln \mathcal{L}/\partial \mu = 0$:

$$\frac{1}{\sigma^2}\sum_{i=1}^N(y_i - \mu) = 0 \implies \hat{\mu} = \bar{y}$$

```{math}
:label: eq:mle-mu
\boxed{\hat{\mu} = \bar{y}} \qquad \text{(unbiased)}
```

**MLE for $\sigma$**: setting $\partial \ln \mathcal{L}/\partial \sigma = 0$:

$$-\frac{N}{\sigma} + \frac{1}{\sigma^3}\sum_{i=1}^N(y_i - \hat{\mu})^2 = 0 \implies \hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^N(y_i - \bar{y})^2$$

```{math}
:label: eq:mle-sigma
\boxed{\hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^N(y_i - \bar{y})^2} \qquad \text{(biased: divides by } N \text{, not } N{-}1\text{)}
```

The factor $N$ (rather than $N-1$) arises because we used the MLE $\hat{\mu}=\bar{y}$ rather than the true $\mu$. Recall from [](#sec:var-est) that the **unbiased** sample variance divides by $N-1$.

(sec:mle-mse)=
## Bias, Variance, and MSE of MLE

Recall from [](#chapter:EstimatorsIntro) the **Mean Squared Error**:

```{math}
:label: eq:mse-mle
\text{MSE}[\hat{\theta}] = V[\hat{\theta}] + b^2
```

For the biased Gaussian variance estimator $\hat{\sigma}^2_N = \frac{1}{N}\sum(y_i-\bar{y})^2$ (with $\hat{\sigma}^2_N = \frac{N-1}{N}\hat{V}$ where $\hat{V}$ is the unbiased estimator):

$$V[\hat{\sigma}^2_N] = \frac{(N-1)^2}{N^2}\,V[\hat{V}] = \frac{2(N-1)}{N^2}\,\sigma^4$$

The bias is $b = E[\hat{\sigma}^2_N] - \sigma^2 = -\sigma^2/N$, so $b^2 = \sigma^4/N^2$.

For $N=10$ the MSE of the biased estimator is $\frac{2\cdot9}{100}\sigma^4 + \frac{1}{100}\sigma^4 = \frac{19}{100}\sigma^4$, while the unbiased estimator has $\text{MSE} = V[\hat{V}] = \frac{2}{9}\sigma^4$.

Since $\frac{19}{100} < \frac{2}{9}$, **the biased variance estimator actually has a smaller MSE than the unbiased one for finite $N$**. The biased estimator is sometimes the better tool.

(sec:mle-uncertainties)=
## Uncertainties on MLE

We have estimated $\hat{\theta}$, but what is the uncertainty $\sigma_{\hat{\theta}}$? There are three approaches:

**Monte Carlo (Brute Force).** Simulate the experiment $Z$ times. Each simulation gives an estimate $\hat{\theta}^{(j)}$. The standard deviation of $\{\hat{\theta}^{(j)}\}$ is $\sigma_{\hat{\theta}}$.

**Information / Minimum Variance Bound.** Use the Fisher Information to compute a lower bound on the variance (see [](#sec:mle-information)).

**Graphical Method.** Read off $\sigma_{\hat{\theta}}$ from the shape of the log-likelihood curve (see [](#sec:mle-graphical)).

(sec:mle-score)=
## The Score

At the maximum, the first derivative of the log-likelihood is zero. This derivative is called the **Score**:

```{math}
:label: eq:score
s = \frac{\partial \ln \mathcal{L}_\theta}{\partial \theta} = \frac{1}{\mathcal{L}_\theta}\frac{\partial \mathcal{L}_\theta}{\partial \theta}
```

For a vector of parameters $\theta = [\theta_1, \ldots, \theta_Z]$, the score is the gradient $s = \nabla \ln \mathcal{L}$.

The score can be written in terms of the PDF. Since $\ln \mathcal{L}_\theta = \sum_i \ln f(x_i;\theta)$:

$$s = \sum_i \frac{\partial \ln f}{\partial \theta} = \sum_i \frac{1}{f}\frac{\partial f}{\partial \theta} \xrightarrow[N\to\infty]{\text{LLN}} N\,\frac{\partial f_X/\partial \theta}{f_X}$$

```{math}
:label: eq:score-pdf
\boxed{s = \frac{N}{f_X}\frac{\partial f_X}{\partial \theta}}
```

### Expected Value of the Score

The expected value of the score is zero:

```{math}
:label: eq:escore
E[s] = 0
```

**Proof.** Using $s = N\,(\partial f_X/\partial\theta)/f_X$:

$$E[s] = \int s\,f_X\,dx = \int N\frac{\partial f_X}{\partial\theta}\,dx = N\frac{\partial}{\partial\theta}\int f_X\,dx = N\frac{\partial}{\partial\theta}\,1 = 0$$

(swapping $\partial/\partial\theta$ and $\int$ is valid under regularity conditions).

(sec:mle-information)=
## The Fisher Information

The **Variance of the Score** is known as the **Fisher Information** $\mathcal{J}_\theta$:

```{math}
:label: eq:information
\mathcal{J}_\theta = V[s] = E[s^2] = -N\,E\!\left[\frac{\partial^2 \ln \mathcal{L}_\theta}{\partial \theta^2}\right] = -N\,E\!\left[\frac{\partial^2 \ln f_X}{\partial \theta^2}\right]
```

(The second equality uses $E[s]=0$ so $V[s]=E[s^2]-E^2[s]=E[s^2]$; the third uses the identity $-NE[\partial s/\partial\theta] = E[s^2]$ derived by differentiating $E[s]=0$.)

A large $\mathcal{J}_\theta$ means the log-likelihood is sharply peaked: the data carry a lot of information about $\theta$, and we can discriminate close values of $\theta$ well.

The **Expected Information** for a single measurement is $\mathcal{J}_s = -E[\partial^2 \ln f_X/\partial\theta^2]$, so $\mathcal{J}_\theta = N\,\mathcal{J}_s$. The **Observed Information** is the curvature at the estimate:

$$\mathcal{J}_s^{\text{obs}} = -\frac{d^2 \ln \mathcal{L}_\theta}{d\theta^2}\bigg|_{\hat{\theta}}$$

**Example (Normal PDF for $\mu$).** From [](#sec:mle-gauss):

$$\frac{\partial^2 \ln \mathcal{L}}{\partial \mu^2} = -\frac{N}{\sigma^2} \implies \mathcal{J}_\mu = \frac{N}{\sigma^2} = \frac{N}{V[Y]} = \frac{1}{V[\bar{Y}]}$$

### Hessian Matrix for Multiple Parameters

When $\theta = [\theta_1, \ldots, \theta_Z]$, the Information generalises to the **Information Matrix**, which is the negative of the Hessian $H_\theta$ of the log-likelihood:

$$H_\theta = \begin{bmatrix}
\dfrac{\partial^2 \ln \mathcal{L}}{\partial\theta_1^2} & \cdots & \dfrac{\partial^2 \ln \mathcal{L}}{\partial\theta_1\partial\theta_Z}\\[6pt]
\vdots & \ddots & \vdots\\[6pt]
\dfrac{\partial^2 \ln \mathcal{L}}{\partial\theta_Z\partial\theta_1} & \cdots & \dfrac{\partial^2 \ln \mathcal{L}}{\partial\theta_Z^2}
\end{bmatrix}$$

The Information Matrix is $\mathcal{J}_\theta = -N\,E[H_\theta]$.

(sec:mvb)=
## The Minimum Variance Bound

The **Rao-Cramér-Fréchet (RCF) Inequality** states that for any **unbiased** estimator $\hat{\theta}$:

```{math}
:label: eq:rcf
V[\hat{\theta}] \geq \frac{1}{\mathcal{J}_\theta}
```

The right-hand side $1/\mathcal{J}_\theta$ is the **Minimum Variance Bound** (MVB). If an estimator achieves $V[\hat{\theta}] = 1/\mathcal{J}_\theta$ it is the **Minimum Variance Unbiased Estimator** (MVUE), the best possible unbiased estimator.

**Example.** For the Gaussian mean, $\mathcal{J}_\mu = N/\sigma^2$, so MVB $= \sigma^2/N = V[\bar{Y}]$. The sample mean $\hat{\mu}=\bar{y}$ attains this bound — it is the MVUE.

For a **biased** estimator with bias $b = E[\hat{\theta}]-\theta$, the full RCF inequality is:

```{math}
:label: eq:rcf-biased
V[\hat{\theta}] \geq \left(1 + \frac{db}{d\theta}\right)^2 / \mathcal{J}_\theta
```

and the corresponding lower bound on the MSE is:

```{math}
:label: eq:mse-bound
\text{MSE} \geq \left(1 + \frac{db}{d\theta}\right)^2 / \mathcal{J}_\theta + b^2
```

(sec:mle-graphical)=
## The Graphical Method

The shape of the log-likelihood near its maximum encodes the uncertainty on $\hat{\theta}$. To see this, Taylor-expand $\ln \mathcal{L}_\theta$ about $\hat{\theta}$:

$$\ln \mathcal{L}_\theta = \ln \mathcal{L}_{\hat{\theta}} + \underbrace{\frac{d\ln\mathcal{L}}{d\theta}\bigg|_{\hat{\theta}}}_{=\,0\text{ (score)}} (\theta - \hat{\theta}) + \frac{1}{2}\frac{d^2\ln\mathcal{L}}{d\theta^2}\bigg|_{\hat{\theta}}(\theta - \hat{\theta})^2 + \cdots$$

The first derivative is zero at the MLE by definition. The second derivative evaluated at $\hat{\theta}$ equals its expected value (both equal $-1/V[\hat{\theta}]$ at the maximum), so:

```{math}
:label: eq:loglike-taylor
\ln \mathcal{L}_\theta \approx \ln \mathcal{L}_{\hat{\theta}} - \frac{(\theta-\hat{\theta})^2}{2\,V[\hat{\theta}]}
```

This is a Gaussian shape! The log-likelihood is approximately parabolic near its peak. The uncertainty $\sigma_{\hat{\theta}} = \sqrt{V[\hat{\theta}]}$ can be read off as the distance from the peak to where $\ln \mathcal{L}$ falls by $1/2$:

```{math}
:label: eq:loglike-drop
\ln \mathcal{L}_{\hat{\theta}} - \ln \mathcal{L}_{\hat{\theta}\pm\sigma_{\hat{\theta}}} = \frac{1}{2}
```

In practice: plot $\ln \mathcal{L}_\theta$ vs $\theta$, find the peak $\hat{\theta}$, and read off the values of $\theta$ where the curve drops by $1/2$ from the peak. The half-width is $\sigma_{\hat{\theta}}$.

(sec:mle-numerical)=
## Numerical MLE with `scipy`

When the log-likelihood has no closed-form derivative, or when there are many parameters, we use numerical optimisation. The standard approach is to minimise the NLL.

### Using `sympy` for derivatives

`sympy` can compute analytical derivatives of the log-likelihood, which is helpful for checking analytical results or for simple cases:

```{code-cell} python
from sympy import symbols, log, factorial, diff

def binom_mle_sympy(n=50, k=27):
    """Derive the Binomial log-likelihood score with sympy."""
    n_sym, k_sym, p = symbols('n k p')

    # log-likelihood (up to constant)
    ln_L = log(factorial(n_sym)/(factorial(k_sym)*factorial(n_sym-k_sym))) \
           + k_sym*log(p) + (n_sym-k_sym)*log(1-p)

    # score: first derivative w.r.t. p
    score = diff(ln_L, p)
    print("Score  d(lnL)/dp =", score)

    # information: minus second derivative
    info  = -diff(ln_L, p, 2)
    print("Info  -d²(lnL)/dp² =", info)

binom_mle_sympy()
```

### Using `scipy.optimize` for numerical minimisation

`scipy` provides several routines for minimising the NLL. For continuous distributions `scipy.stats.fit()` (which uses MLE by default) is the most convenient:

```{code-cell} python
import numpy as np
from scipy.stats import binom, expon
import scipy.optimize as opt

def binomial_mle_numerical(n=50, k=27):
    """Numerical MLE for Binomial p using minimize, fsolve, and fit."""
    true_p = k / n
    rng = np.random.default_rng(42)
    data = binom.rvs(n, true_p, size=1000, random_state=rng)

    # negative log-likelihood (using scipy's built-in nnlf)
    def nll(p, n, data):
        return binom.nnlf((n, p, 0), data)

    p_guess = 0.5

    # scipy.optimize.minimize with SLSQP (handles bounds)
    res_min = opt.minimize(nll, p_guess, args=(n, data),
                           bounds=[(0, 1)], method='SLSQP')
    print(f"minimize (SLSQP): p_hat = {res_min.x[0]:.4f}")

    # scipy.optimize.fsolve (root-finder on NLL)
    res_fs = opt.fsolve(nll, p_guess, args=(n, data), full_output=True)
    print(f"fsolve:           p_hat = {res_fs[0][0]:.4f}")

    # scipy.stats.fit (highest-level interface)
    fit_res = opt.minimize(nll, p_guess, args=(n, data), bounds=[(0, 1)])
    print(f"True p: {true_p:.4f}")

binomial_mle_numerical()
```

```{code-cell} python
def expon_mle_numerical():
    """Numerical MLE for Exponential rate using scipy.stats.expon.fit."""
    data = np.array([15, 18, 11])          # taxi waiting times (minutes)

    # scipy.stats fit: floc=0 fixes the location parameter at 0
    _, scale_hat = expon.fit(data, floc=0)
    print(f"MLE mean waiting time: {scale_hat:.3f} min")
    print(f"MLE rate: {1/scale_hat:.4f} min⁻¹")
    print(f"Analytical: rate = 3/44 = {3/44:.4f} min⁻¹")

expon_mle_numerical()
```

### Plotting the Likelihood, Log-Likelihood, and NLL

Plotting all three functions helps build intuition. The likelihood is a sharply peaked product; the log-likelihood is a smooth parabola-like curve; the NLL is the same parabola flipped:

```{code-cell} python
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_likelihood_suite(mu_true=5.0, sigma=1.0, N=100, seed=42):
    """Plot L, lnL, and -lnL as a function of mu."""
    rng = np.random.default_rng(seed)
    data = norm.rvs(loc=mu_true, scale=sigma, size=N, random_state=rng)
    mus  = np.linspace(mu_true - 1.5, mu_true + 1.5, 200)

    L     = np.array([np.prod(norm.pdf(data, m, sigma)) for m in mus])
    lnL   = np.array([np.sum(np.log(norm.pdf(data, m, sigma))) for m in mus])
    nll   = -lnL

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, y, label in zip(axes,
                            [L, lnL, nll],
                            [r'$\mathcal{L}_\theta$',
                             r'$\ln\,\mathcal{L}_\theta$',
                             r'$-\ln\,\mathcal{L}_\theta$']):
        ax.plot(mus, y, 'r--')
        ax.axvline(mu_true, color='k', ls=':', label=rf'$\mu={mu_true}$')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(label)
        ax.legend()
    fig.tight_layout()
    plt.show()

plot_likelihood_suite()
```

(sec:mle-summary)=
## Summary

| Distribution | Parameter | MLE | Biased? |
|---|---|---|---|
| Binomial$(n,p)$ | $p$ | $\hat{p} = k/n$ | No |
| Exponential$(\lambda)$ | $\lambda$ | $\hat{\lambda} = 1/\bar{t}$ | No |
| Normal$(\mu,\sigma^2)$ | $\mu$ | $\hat{\mu} = \bar{y}$ | No |
| Normal$(\mu,\sigma^2)$ | $\sigma^2$ | $\hat{\sigma}^2 = \frac{1}{N}\sum(y_i-\bar{y})^2$ | Yes (use $N-1$ for unbiased) |

Key results:

- The Score $s = \partial\ln\mathcal{L}/\partial\theta$ is zero at the MLE, and $E[s]=0$ always.
- The Fisher Information $\mathcal{J}_\theta = V[s] = -NE[\partial^2\ln f_X/\partial\theta^2]$.
- The Rao-Cramér-Fréchet inequality: $V[\hat{\theta}] \geq 1/\mathcal{J}_\theta$ for any unbiased estimator.
- The log-likelihood is approximately parabolic near $\hat{\theta}$; the width gives $\sigma_{\hat{\theta}}$.
- Numerically: minimise the NLL using `scipy.optimize.minimize`, `fsolve`, or `scipy.stats.fit`.
