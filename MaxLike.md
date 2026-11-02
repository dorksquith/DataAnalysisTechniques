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
## The Likelihood Function (Recap)

Suppose the data are independent and identically distributed (iid), each drawn from a PDF (or PMF) $f_X(x;\theta)$. The joint probability of observing all $N$ values is the product of the individual probabilities:

```{math}
:label: eq:likelihood
\mathcal{L}_\theta = \prod_{i=1}^{N} f_X(x_i;\,\theta)
```

This is the **Likelihood** $\mathcal{L}_\theta$: a function of the parameter $\theta$, not of $x$.

The **Maximum Likelihood Estimate** $\hat{\theta}$ is the value of $\theta$ that maximises $\mathcal{L}_\theta$:

```{math}
:label: eq:mle-def
\hat{\theta}_{\mathsf{MLE}} = \arg\max_\theta \mathcal{L}_\theta
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
\mathsf{NLL}_{\theta} \equiv -\ln \mathcal{L}_\theta = -\sum_{i=1}^{N} \ln f_X(x_i;\,\theta)
```

The **Maximum Likelihood Estimate** $\hat{\theta}$ is the value of $\theta$ that **minimises** $\mathsf{NLL}_{\theta}$:

```{math}
:label: eq:nll-mle-def
\hat{\theta}_{\mathsf{MLE}} = \arg\min_\theta \mathsf{NLL}_{\theta}
```

(sec:mle-uncertainties)=
## Uncertainties on MLE

There are three approaches to calculating the uncertainty on a Maximum Likelihood parameter estimate:

Monte Carlo (Brute Force)
: Simulate the experiment $Z$ times. Each simulation gives an estimate $\hat{\theta}^{(j)}$. The standard deviation of $\{\hat{\theta}^{(j)}\}$ is $\sigma_{\hat{\theta}}$.

Information / Minimum Variance Bound
: Use the Fisher Information to compute a lower bound on the variance (see [](#sec:mle-information)).

Graphical Method
: Read off $\sigma_{\hat{\theta}}$ from the shape of the log-likelihood curve (see [](#sec:mle-graphical)).


(sec:mle-analytical)=
## Analytical MLE

To find $\hat{\theta}$ analytically we differentiate the log-likelihood and set the result to zero:

```{math}
:label: eq:score-zero
\frac{\partial \ln \mathcal{L}_\theta}{\partial \theta}\bigg|_{\hat{\theta}} = 0
```

We must also check that the second derivative is negative (confirming a maximum, not a minimum or saddle point).

(sec:mle-gauss)=
### Example: estimating Norm $\mu$ and $\sigma$

Let $X \sim$ Norm($\mu, \sigma$) with PDF $f_X(x;\mu,\sigma)= \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)}$. 

For $N$ IID observations:

$$\ln \mathcal{L}_{\mu,\sigma} = -N\ln\sigma - N\ln\sqrt{2\pi} - \frac{1}{2\sigma^2}\sum_{i=1}^N (x_i - \mu)^2$$

**MLE for $\mu$**: setting $\dfrac{ \partial \ln \mathcal{L}}{\partial \mu} = 0$:

$$\sum_{i=1}^N(x_i - \mu) = 0\;\; \mathsf{when}\;\;\sum_{i=1}^N x_i = N\mu$$

```{math}
:label: eq:mle-mu
\hat{\mu}_{_\mathsf{MLE}} = \bar{x}
```

**MLE for $\sigma$**: setting $\dfrac{ \partial \ln \mathcal{L}}{\partial \sigma} = 0$:

$$-\frac{N}{\sigma} + \frac{1}{\sigma^3}\sum_{i=1}^N(x_i - \hat{\mu})^2 = 0 \implies \hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^N(x_i - \bar{x})^2$$

```{math}
:label: eq:mle-sigma
\hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^N(x_i - \bar{x})^2
```

Note: the factor $N$ (rather than $N-1$) arises in [](#eq:mle-sigma) because we used the MLE $\hat{\mu}=\bar{x}$ rather than the true $\mu$. Recall from [](#sec:var-est) that the **unbiased** sample variance divides by $N-1$.

<!--
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

-->


(sec:mle-score)=
### The Score

At the maximum, the first derivative of the log-likelihood is zero. This derivative is called the **Score**:

```{math}
:label: eq:score
s = \frac{\partial \ln \mathcal{L}_\theta}{\partial \theta} = \frac{1}{\mathcal{L}_\theta}\frac{\partial \mathcal{L}_\theta}{\partial \theta}
```

Note that we have used the [chain rule](#sec:mma-chain-rule) in [](#eq:score).

For a vector of parameters $\theta = [\theta_1, \ldots, \theta_Z]$, the score is the gradient $s = \nabla \ln \mathcal{L}$, where $\nabla$ is the [gradient operator](#sec:mma-grad).

The score can helpfully be written in terms of the PDF, $f_X(x;\theta)$, which we will shorthand as $f$ in [](#eq:score-pdf): 

<!--Since $\ln \mathcal{L}_\theta = \sum_i \ln f_X(x_i;\theta)$:-->

```{math}
:label: eq:score-pdf
\begin{aligned}
s &= \sum_i \frac{\partial }{\partial \theta} \ln f_X \\
  &= \sum_i \frac{1}{f_X}\frac{\partial f_X}{\partial \theta}  \\
  &= \frac{N}{f_X}\frac{\partial f_X}{\partial \theta}  \\
\end{aligned}
```

Note that for the last line in [](#eq:score-pdf) we have used the [LLN](#eq:lln).


### Expected Value of the Score

The expected value of the score is zero:

```{math}
:label: eq:escore
E[s] = 0
```

**Proof:**

```{math}
:label: eq:escore-proof
\begin{aligned}
E[s] &= \int s\, f_X\, dx\\ \\
     &= \int \frac{N}{f_X}\frac{\partial f_X}{\partial\theta} f_X\, dx\\ \\
     &= \int N\frac{\partial f_X}{\partial\theta}\, dx \\ \\
     &= N\frac{\partial}{\partial\theta}\int f_X\, dx \\ \\
     &= N\frac{\partial}{\partial\theta}\,1 = 0
\end{aligned}
```

### Derivative of the Expected Value of the Score

Since $E[s] = 0$, its derivative with respect to $\theta$ is also zero. Differentiating both sides and unpacking the result gives a useful identity:

```{math}
:label: eq:descore-proof

\begin{aligned}
\frac{\partial}{\partial\theta}E[s]
= \frac{\partial}{\partial\theta}\int s\, f_X\, dx
&= \int \frac{\partial}{\partial\theta}\bigl[s\, f_X\bigr]\, dx\\ \\

&= \int \left[\frac{\partial s}{\partial\theta}f_X + s\frac{\partial f_X}{\partial\theta}\right]dx \\ \\

&= \int \left[\frac{\partial s}{\partial\theta}f_X + \frac{s^2}{N}f_X\right]dx\\ \\

&= E\!\left[\frac{\partial s}{\partial\theta}\right] + \frac{1}{N}E[s^2] = 0

\end{aligned}
```

Rearranging:

```{math}
:label: eq:descore-result
-N\,E\!\left[\frac{\partial s}{\partial\theta}\right] = E[s^2]
```

(sec:mle-information)=
## The Fisher Information

The **Variance of the Score** is known as the **Fisher Information** $\mathcal{J}_\theta$. Using the two results above:

```{math}
:label: eq:vscore-proof

\begin{aligned}
V[s] &= E[s^2] - E^2[s] \\
     &= E[s^2] \\
     &= -N\,E\!\left[\frac{\partial s}{\partial\theta}\right]\\
     &= -N\,E\!\left[\frac{\partial^2 \ln\mathcal{L}_\theta}{\partial\theta^2}\right]
\end{aligned}

```

```{math}
:label: eq:information
\mathcal{J}_\theta \equiv V[s] = -N\,E\!\left[\frac{\partial^2 \ln \mathcal{L}_\theta}{\partial \theta^2}\right] = -N\,E\!\left[\frac{\partial^2 \ln f_X}{\partial \theta^2}\right]
```

A large $\mathcal{J}_\theta$ means the log-likelihood is sharply peaked: the data carry a lot of information about $\theta$, and we can discriminate close values of $\theta$ well.


The **Expected Information**for a single measurement is $\mathcal{J}_s = \mathcal{J}_\theta/ N$. 


The **Observed Information** is the curvature at the estimate

  $$\mathcal{J}_s^{\text{obs}} = -\frac{d^2 \ln \mathcal{L}_\theta}{d\theta^2}\bigg|_{\hat{\theta}}$$


**Example (Normal PDF for $\mu$).** From [](#sec:mle-gauss):

$$ \mathcal{J}_\mu = -\frac{\partial^2 \ln \mathcal{L}}{\partial \mu^2}  = \frac{N}{\sigma^2} = \frac{N}{V[X]} = \frac{1}{V[\bar{X}]}$$



### Hessian Matrix for Multiple Parameters

The Hessian Matrix for a function is the matrix of second order partial derivatives (see[](#sec:mma_hessian)). For multiple parameters $\theta = [\theta_1, \ldots, \theta_Z]$, the Hessian Matrix for the Log Likelihood is:

```{math}
:label: eq:hessian_ll
H_\theta = \begin{bmatrix}
\dfrac{\partial^2 \ln \mathcal{L}}{\partial\theta_1^2} & \cdots & \dfrac{\partial^2 \ln \mathcal{L}}{\partial\theta_1\partial\theta_Z}\\[6pt]
\vdots & \ddots & \vdots\\[6pt]
\dfrac{\partial^2 \ln \mathcal{L}}{\partial\theta_Z\partial\theta_1} & \cdots & \dfrac{\partial^2 \ln \mathcal{L}}{\partial\theta_Z^2}
\end{bmatrix}
```

The **Information Matrix** is the negative of the Hessian $H_\theta$ of the log-likelihood:

```{math}
:label: eq:info_matrix
\mathcal{J}_\theta = -N\,E[H_\theta]
```

**Example: Normal PDF for $\theta = \{\mu, \sigma\}$.**

Starting from the log-likelihood

$$\ln\mathcal{L} = -N\ln\sigma - \tfrac{N}{2}\ln(2\pi) - \frac{1}{2\sigma^2}\sum_{i=1}^N(y_i-\mu)^2$$

the first partial derivatives are:

$$\frac{\partial \ln\mathcal{L}}{\partial\mu} = \frac{1}{\sigma^2}\sum_{i=1}^N(y_i-\mu), \qquad \frac{\partial \ln\mathcal{L}}{\partial\sigma} = -\frac{N}{\sigma} + \frac{1}{\sigma^3}\sum_{i=1}^N(y_i-\mu)^2$$

The second partial derivatives are:

$$\frac{\partial^2\ln\mathcal{L}}{\partial\mu^2} = -\frac{N}{\sigma^2}, \qquad \frac{\partial^2\ln\mathcal{L}}{\partial\sigma^2} = \frac{N}{\sigma^2} - \frac{3}{\sigma^4}\sum_{i=1}^N(y_i-\mu)^2, \qquad \frac{\partial^2\ln\mathcal{L}}{\partial\mu\,\partial\sigma} = -\frac{2}{\sigma^3}\sum_{i=1}^N(y_i-\mu)$$

Assembling the Hessian:

$$H_\theta = \begin{bmatrix} -N/\sigma^2 & -2\sigma^{-3}\displaystyle\sum(y_i-\mu) \\[6pt] -2\sigma^{-3}\displaystyle\sum(y_i-\mu) & N/\sigma^2 - 3\sigma^{-4}\displaystyle\sum(y_i-\mu)^2 \end{bmatrix}$$

Taking expectations using $E[y_i-\mu]=0$ and $E[(y_i-\mu)^2]=\sigma^2$:

$$E[H_\theta] = \begin{bmatrix} -N/\sigma^2 & 0 \\ 0 & -2N/\sigma^2 \end{bmatrix}$$

The Information Matrix is therefore:

$$\mathcal{J}_\theta = -N\,E[H_\theta] = \begin{bmatrix} N/\sigma^2 & 0 \\ 0 & 2N/\sigma^2 \end{bmatrix}$$

The off-diagonal zeros mean $\hat{\mu}$ and $\hat{\sigma}$ are uncorrelated estimators — information about $\mu$ and information about $\sigma$ are completely separable for the Normal distribution.

(sec:mvb)=
## The Minimum Variance Bound

The **Rao-Cramér-Fréchet (RCF) Inequality** states that for any **unbiased** estimator $\hat{\theta}$:

```{math}
:label: eq:rcf
V[\hat{\theta}] \geq \frac{1}{\mathcal{J}_\theta}
```

The right-hand side $1/\mathcal{J}_\theta$ is the **Minimum Variance Bound** (MVB). If an estimator achieves $V[\hat{\theta}] = 1/\mathcal{J}_\theta$ it is the **Minimum Variance Unbiased Estimator** (MVUE), the best possible unbiased estimator.

**Example.** For the Gaussian mean, $\mathcal{J}_\mu = N/\sigma^2$, so MVB $= \sigma^2/N = V[\bar{X}]$. The sample mean $\hat{\mu}=\bar{x}$ attains this bound — it is the MVUE.

For a **biased** estimator with bias $b = E[\hat{\theta}]-\theta$, the full RCF inequality is:

```{math}
:label: eq:rcf-biased
V[\hat{\theta}] \geq \left(1 + \frac{db}{d\theta}\right)^2 \bigg/ \mathcal{J}_\theta
```

and the corresponding lower bound on the [MSE](#eq:mse) is:

```{math}
:label: eq:mse-bound
\text{MSE} \geq \left(1 + \frac{db}{d\theta}\right)^2 \bigg/ \mathcal{J}_\theta \;\; + \;\; b^2
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

Let's import some useful libraries:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

# scipy.special.binom is the binomial coefficient function
from scipy.special import binom as bnc

# scipy.stats.binom is the probability distribution
from scipy.stats import binom

# for parameter estimation
from scipy.stats import fit
import scipy.optimize as opt

from sympy import symbols, diff, solve, lambdify
from sympy import log, expand_log, factorial, expand, simplify, collect,cancel
```



Here we look at what can be done with ```sympy``` and at a few of the (many) methods ```scipy``` offers:
* ```fit``` [ [scipy.stats.fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fit.html) ] is a particularly easy way to get the Maximum Likelihood pararameter estimate when using specific probability distributions
* ```fsolve``` [ [scipy.optimize.fsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html)
 ] is a method for doing the same if you can figure out the first derivative of the function.
* ```minimize_scalar```  [ [scipy.optimize.minimize_scalar](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar
) ] is another method for doing the same, and you don't need to calculate the derivative function.


### General Introduction to Minimisation


We will look at this function, which is the one chosen for scipy's `minimize_scalar` example:

$f(x) = (x-2) \cdot x \cdot (x+2)^2$

Let's plot the function before we do anything else. The scipy example does not specify a domain for x, so let's make one up:

$-10 \leq x \leq 10$

```{code-cell} python
# define the function
def f1(x):
    return (x-2) * x * (x+2)**2
```

```{code-cell} python
# plot the function over our chosen range with some sub-ranges to check the close-up behaviour

def plot_func(x0=None,x1=None):
    

    if(x0 and x1):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), layout='tight')
        x = np.linspace(x0,x1,1000)

        ax.plot(x, f1(x), label="f(x)")
        ax.set_xlabel('X')
        ax.set_ylabel('f(x)')
    else:
      fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8,4), layout='tight')
      x0s=[-10,-5,-2.5]
      x1s=[10,5,2.5]
      for i,ax in enumerate(axs):

        x = np.linspace(x0s[i],x1s[i],1000)

        ax.plot(x, f1(x), label="f(x)")
        ax.set_xlabel('X')
        ax.set_ylabel('f(x)')

    plt.show()

plot_func()
```

Plotting the function over various ranges allows us to see that what looks like a long, flat minimum over the range -10,10 actually has what looks like two minima and a maximum in the region -2.5,2.5.

Let's use the `minimize_scalar` method to find the coordinates of the global minimum.

### `minimize_scalar` for f(x)

```{code-cell} python
def minimize_demo(f1):

    result = opt.minimize_scalar(f1)

    return result.x, result.fun

mini_x, mini_f = minimize_demo(f1)    

print(f" * minimize_scalar minimum at x = {mini_x:.3f}, f = {mini_f:.3f}")
```

Lets replot around the lowest of the two minima, $x\approx 1.28$, to check the result by eye.

```{code-cell} python
plot_func(1,1.5)
```

Scipy's `minimize_scalar` result of `(x=1.28, f(x)=-9.91)` looks legit. How did it do that? Calculus! And we know how to do it by hand.


### Using `sympy (diff, solve, lambdify)` for the math behind `minimize_scalar` :

We find the minimum of a function by differentiating it, as the derivative of a function evaluated at some point is its gradient (or slope), and the gradient is zero at points where the function turns over.

1. Find the function's turning points. We do this by taking the derivative and using sympy's `solve` to find the roots.
2. Determine if the points are minima or maxima. We use the second derivative for this. If it is **positive**, we have a **minimum**, because the function is curving upwards in the positive direction on either side of the turning point. Vice versa for **negative: maximum**. If the second derivative is zero, the turning point is an "inflection point", rather than a maximum or minimum.
3. Select the minimum with the lowest function value.

```{code-cell} python
def sympy_minimise_function(f):

    x = symbols('x')

    #---------------------------
    # 1) Find the turning points
    #---------------------------

    # Calculate the first derivative of the function
    d1 = diff( f(x), x)

    # Find the values of x for which df/dx is zero
    x_when_derivative_is_zero = solve(d1, x)

    # Make a list of the numeric values of x at the turning points
    x_turning_points = [float(xi) for xi in x_when_derivative_is_zero] 

    #---------------------------
    # 2) Determine if the turning points are minima or maxima.
    #---------------------------

    # Calculate the second derivative of the function
    d2 = diff( d1, x)

    # get the numeric values of the second derivative by providing a value for x
    evaluate_d2 = lambdify(x, d2)

    # take the values of x for which the second derivative is positive: minima
    x_minima= [x for x in x_turning_points if evaluate_d2(x)>0 ]

    #---------------------------
    # 3) Determine the global minimum, which is the one with the lowest function value.
    #---------------------------

    # List of function values at each minimum
    f_at_x_minima = [f(x) for x in x_minima]

    # List index of the lowest minimum
    minindex = f_at_x_minima.index(min(f_at_x_minima))

    # x and f(x) at the lowest minimum
    x_for_fmin = x_minima[minindex]
    f_for_fmin = f_at_x_minima[minindex]

    return x_for_fmin, f_for_fmin

sympy_mini_x, sympy_mini_f = sympy_minimise_function(f1)

print(f"Find the lowest minimum coordinates `by hand' (using sympy): ")
print(f" * x = {sympy_mini_x:.3f}, f = {sympy_mini_f:.3f}")



print(f"Find the lowest minimum coordinates using minimize_scalar: ")    
print(f" * x = {mini_x:.3f}, f = {mini_f:.3f}")
```

## Parameter Estimation : Binomial `p`

Now that we understand how a function is minimised, let's try doing this with a Negative Log Likelihood Function.

We will use the Binomial distribution and ask the question "what is the most likely value of p, given the data?"

We will try this a few different ways.

Reminder of the forms of the **Binomial PMF**
$$p_K(k) = C_k^n\, p^k\, (1-p)^{n-k}\;\;\;\;\; 0\leq k \leq n$$,

and the **Binomial Coefficient**:
$$C_k^n = \dfrac{n!}{k!(n-k)!}$$,

and the general form for the **Negative Log Likelihood**:

$$\mathsf{NLL} = - \ln\mathcal{L} = \sum\limits_i^N \ln {p_K(k_i)}$$.

### `sympy` methods

This is essentially the "by-hand" method, with a bit of help from `sympy` on the derivatives, and perhaps most helpfully, on finding the roots with `solve`. All of this can also be done with a pen and paper, but `sympy` can be very useful.

The NLL has a minimum at the most likely value of `p`, aka the Maximum Likelihood Estimate (MLE) for p.

The code below has the following steps:
* The symbolic functional form of the Binomial PMF, `sym_pmf`.
* The negative log of this, `sym_nll`, which is the Negative Log Likelihood (NLL),
* The first derivative of the NLL, `sym_dnll_dp`, provides the Objective Function which has a value of zero at the mle estimate for `p`.

Note that the Binomial Coefficient does not depend on `p`. As such, $C_k^n$ will vanish when we take the derivative of the NLL with respect to p. We do not need to calculate it explicitly in this example, but we would need to calculate it for the PMF probabilities, in which case we would use:

```
from scipy.special import binom as bnc
binco = bnc(n,k)
```

There are several subtleties needed for maximising the helpfulness of ```sympy```. Relevant here:
1. When defining the ```symbols```, if they are constrained to positive values you should mark them as such with the flag ```positive=True```.
2. Sympy will not simplify/expand/cancel the results it finds unless you explictly ask it to. In the example below we use ```expand_log``` to put the result in the helpful log rules form of ```log(abc) = log(a) + log(b) + log(c)```, and we use ```cancel``` to cancel terms like ```a**b * a**-b = a**0 = 1```

```{code-cell} python
# let's do this first of all with sympy
def estimate_binomial_p_sympy():

    # Important Note: the positive=True flag is crucial for getting a sensible answer from solve
    c,n,p,k = symbols('c n p k', positive=True)

    #  does not depend on p, so when we take the derivative with respect to p it will disappear. So can just call it c.
    def sym_pmf(c, n, p, k):
        return  c * p**k * (1-p)**(n-k)

    sym_L = sym_pmf(c, n, p, k)
    print(f"Binomial Likelihood:\n {sym_L}")

    def sym_nll(c, n, p, k):
        return -log( sym_L )
    sym_NLL_notsimp = sym_nll(c, n, p, k)
    #print(f"Unexpanded Log of Binomial Likelihood:\n {sym_NLL_notsimp}")

    # important step: tell sympy to use log rules
    # if we take the derivative of the sym_NLL directly, we get a weird second estimate...
    sym_NLL= expand_log(sym_NLL_notsimp, factor=1)
    print(f"Negative Log Likelihood:\n {sym_NLL}")

    def sym_dnll_dp(c, n, p, k):      
        return diff(sym_NLL, p)

    sym_dNLL_dp = sym_dnll_dp(c, n, p, k)
    # note that the first part of the second term (1 - p)**(-k + n)*(1 - p)**(k - n) = (1-p)^0 = 1
    # cancel the unhelpful complexity
    sym_dNLL_dp = cancel(sym_dNLL_dp)

    print(f"Derivative of Negative Log Likelihood:\n {sym_dNLL_dp}")

    # find the values for p at which this first derivative is equal to zero.
    from sympy.solvers import solve
    estimate1_p = solve(sym_dNLL_dp, p)
    print(f"Maximum Likelihood estimate(s) for p:\n {estimate1_p }")

estimate_binomial_p_sympy()
```

```Sympy``` did the derivative for us and found the symbolic maximum likelihood estimate $\hat{p}_{_\mathsf{MLE}}$=k/n.

This solution makes perfect sense. If we toss the coin n=3 times and find k=1 heads, the information in the data is that there is a 1/3 chance of heads on each toss. So this is the logical value for the MLE of the parameter p.



### `scipy` Methods

Now, let's try out some of the ```scipy``` methods!

1. ```fit```
2. ```fsolve```
3. ```minimize_scalar```
4. ```minimize```

Note that ```scipy``` methods are numeric rather than symbolic, so we need to pass in values for n and k (k is the data). We are not providing a value for p because this is the parameter we are estimating, but we will often pass an initial guess and/or bounds on this parameter, depending on the method in question.

### Set the fake data and ```n``` parameter value

```{code-cell} python
# n parameter: number of coin tosses
n=50
# some fake data points I made up, for the number of heads from 50 tosses
data=np.array([27,28,29,11])
```

### ```scipy.stats.fit```

This is a lovely simple way to estimate the parameter of a ```scipy.stats``` probability distribution. Note that it cannot be used for a regular function.

In this example we pass three things to fit:
* The binomial distribution, ```dist=binom``` without setting the parameters, straight "out-of-the-box" from ```scipy.stats import binom```.
* The data, ```data=data```, which is an array of values for the RV (e.g. number of heads)
* The bounds, ```bounds = {'n': (n,n), 'p': (0.001, 0.999) }```, on the parameters `n` and `p` of the Binomial distribution. For the `n` parameter we fix it by setting the lower and upper bound to the same value. For the `p` parameter we let it vary between 0.001-0.999, which is close to the maximum range allowed for a probability ($0\leq\,$`p`$\,\leq 1$).

The fit function returns a dictionary with the keys ```params, success, message```. The ```params``` key has three elements for the three possible parameters of the binomial distribution, in the order ```n, p, loc```.

```{code-cell} python
# ------------------------#
# METHOD: scipy.stats.fit
# ------------------------#
def estimate_binomial_p_scipy_fit(n,data):

    bounds = {'n': (n,n), 'p': (0.001, 0.999) }

    result_fit = fit(dist=binom, data=data, bounds=bounds)

    mle_est_p = result_fit.params[1]

    return mle_est_p 

mle_est_fit = estimate_binomial_p_scipy_fit(n,data)

print(f"scipy.stats.fit MLE estimate for p={mle_est_fit:.3f}")

print(f"\n (full precision: {mle_est_fit})")
```

### ```scipy.optimize.fsolve```

The job of ```fsolve``` is to provide the roots of a function or system of functions. For ```fsolve``` to work, the function must have roots, meaning it must be equal to zero at one or more values of its variable(s).

This method is well-suited for the first derivative of the Log Likelihood function, $\dfrac{d}{dp} \ln \mathcal{L}$, as this is zero at the maximum likelihood estimate for `p`; or in other words, the root of the derivative LL function is `p`.

In this example we pass three things to ```fsolve```:
* The objective function, ```func=dLL_dp```, is the derivative, $\dfrac{d}{dp} \ln \mathcal{L}$, which the algorithm sets to zero in order to determine the root, ```p```.
* An initial guess at the parameter, ```x0=p_guess```, which gives the algorithm a starting point.
* The names of the objective function arguments that we are **not** seeking to estimate, ```args=spectators```. In this the objective function requires the data and n parameter, but these are known and are therefore not the job of the algorithm to solve for.

The ```fsolve``` function returns an array of the roots. In our example here there is only one root, as there is only one peak in a well-constructed Likelihood (or Log Likelihood, or Negative Log Likelihood) distribution.

```{code-cell} python
# ------------------------------#
# METHOD: scipy.optimize.fsolve
# ------------------------------#
def estimate_binomial_p_scipy_fsolve(n,data):

    # we have some fail-safes in this function to catch cases where p=0,1, because there is no option for bounds in fsolve
    # the dLL_dp form was provided to us above in the printout from sympy:
    # Derivative of Negative Log Likelihood:
    # (k - n*p)/(p**2 - p)
    def dLL_dp(p, n, data):
        k = data
        if(p==0 or p==1):
            return -np.inf         
        return np.sum( (k - n*p)/(p**2 - p) )

    p_guess = 0.1

    spectators = (n, data)

    result_fsolve = opt.fsolve(func=dLL_dp, x0=p_guess, args=spectators )

    mle_est_p = result_fsolve[0]

    return mle_est_p 

mle_est_fsolve = estimate_binomial_p_scipy_fsolve(n,data)

print(f"scipy.optimize.fsolve MLE estimate for p={mle_est_fsolve:.3f}")

print(f"\n (full precision: {mle_est_fsolve})")
```

### ```scipy.optimize.minimize_scalar```

The  ```minimize_scalar``` function finds the minimum of the passed scalar function.

This method does not require the function to have "roots", and so does not need the derivative; the Negative Log Likelihood itself is just fine. We must take care to pass the Negative, which has a minimum, and not the Log Likelihood, which has a maximum and therefore will not work.

In this example we pass three things to ```minimize_scalar```:
* The function to be minimised, ```fun=NLL```. Note the name of this is ```fun``` rather than ```func``` for ```fsolve```.
* As per ```fsolve```, the names of the function arguments that we are **not** seeking to estimate, ```args=spectators```.
* Similar to ```fit```, we pass the bounds on the parameter we are seeking, ```bounds=bounds_p```.

The ```minimize_scalar``` returns a dictionary with several keys. The key we are interested in is ```x```, which is the maximum likelihood estimate for the ```p``` parameter.

```{code-cell} python
# ---------------------------------------#
# METHOD: scipy.optimize.minimize_scalar
# ---------------------------------------#
def estimate_binomial_p_scipy_minimize_scalar(n,data):

    def NLL(p, n, data ):
        k = data
        c = bnc(n,k) 
        log_pmf = np.log(c) + k * np.log(p) + (n-k) * np.log(1-p) 
        return -1*np.sum( log_pmf )

    # we will pass the same spectators as we defined for fsolve.
    spectators = (n, data)

    # we also need to give some bounds on p (not needed for all methods, but we don't have time for that!)
    bounds_p = (0.001,0.999)

    result_minimize_scalar = opt.minimize_scalar(fun=NLL, args=spectators, bounds=bounds_p)

    mle_est_p = result_minimize_scalar.x

    return mle_est_p 

mle_est_minimize_scalar = estimate_binomial_p_scipy_minimize_scalar(n,data)

print(f"scipy.optimize.minimize_scalar MLE estimate for p={mle_est_minimize_scalar:.3f}")

print(f"\n (full precision: {mle_est_minimize_scalar})")
```

### ```scipy.optimize.minimize```

The use of this method is not necessary for the Binomial, as it is a scalar function, but I am including it in case future-you needs to minimize a vector function. It works perfectly well on scalar functions too.

The differences to note with respect to ```minimize_scalar``` include:
* There is a required argument, ```x0=p_guess```, which is the initial guess at the parameter or parameters to be estimated.
* The ```bounds``` must be passed as a list of one or more tuples, rather than a tuple
* The ```x``` returned is an array, so in our case where just one parameter value is being sought, we access it with ```x[0]```

```{code-cell} python
# ---------------------------------------#
# METHOD: scipy.optimize.minimize
# ---------------------------------------#
def estimate_binomial_p_scipy_minimize(n,data):

    def NLL(p, n, data ):
        k = data
        c = bnc(n,k) 
        log_pmf = np.log(c) + k * np.log(p) + (n-k) * np.log(1-p) 
        return -1*np.sum( log_pmf )

    p_guess=0.1
    spectators = (n, data)
    bounds_p = [(0.001,0.999)]

    result_minimize = opt.minimize(fun=NLL, x0=p_guess, args=spectators, bounds=bounds_p )

    mle_est_p = result_minimize.x[0]

    return mle_est_p 

mle_est_minimize = estimate_binomial_p_scipy_minimize(n,data)

print(f"scipy.optimize.minimize MLE estimate for p={mle_est_minimize:.3f}")

print(f"\n (full precision: {mle_est_minimize})")
```

## Model Tests

In [](#chapter:normtest) we looked at some test statistics for comparing the means of a distribution and a hypothesis, and for comparing the means of more than one distribution. These tests were limited to use with approximately Normal data.

Here we will look at tests for comparing the performance of two models. We will limit the model differences to a difference in parameter values, but note that the AIC and BIC tests can be used for different distributions.

### Log Likelihood Ratio (LLR)

The Log Likelihood Ratio test compares two different Log Likelihoods calculated on the same data, but with different models (e.g. different parameter choices).

```{math}
:label: eq:llr
   \mathsf{LLR} = -2(\ln\mathcal{L}_B - \ln\mathcal{L}_A).
```

To use this method, $\ln\mathcal{L}_A$ must be a simplified version of $\ln\mathcal{L}_B$.


Example 1
: $\mathcal{L} = \prod\limits_i^N f_{X} $, where $f_{X,A} =  ax^3$ and $f_{X,B} =  ax^3 / ( 1+ bx) $. 

  Model A has $N_p=1$ parameters, Model B has $N_p=2$. 

  Model A is nested in Model B, because if $b=0$, they are the same. 

  The difference in degrees of freedom $\nu$ is $|\nu_A - \nu_B| = 1$.

Example 2
: $f_{X,A} =  ax^3,\;\;\; 0.1< a < 0.2$ and $f_{X,B} =  ax^3,\;\;\;0 < a < 1$. 

  Model A has an additional **constraint** with respect to model B. 

  $|\nu_A - \nu_B| = 1$



<!--We could then use `pvalue=chi2.sf(LLR, df=1)`: if  `pvalue` $< \alpha$, we reject the simpler model, A, in favour of its more complicated friend, B. But, there are other ways of doing this with LLR.-->


### Akaike and Bayes Information Criteria (AIC and BIC)

If we want to compare un-nested models, we can use the AIC or BIC statistics. They
both balance the simplicity of the model (k: number of model parameters) with the
quality of fit to data.

The Akaike Information Criterion (AIC):

```{math}
:label: eq:aic
   \mathsf{AIC} = 2k - 2\ln\mathcal{L}_{\hat{\theta}}.
```

The Bayesian Information Criterion (BIC):

```{math}
:label: eq:bic
   \mathsf{BIC} = k\ln N - 2\ln\mathcal{L}_{\hat{\theta}}.
```

A smaller Information Criterion value is better, eg $\mathsf{AIC}_{a} < \mathsf{AIC}_{b}$ identifies the preferred model as a.


Last week we tossed a coin 50 times and it came up heads 27 times.

The Likelihood under Model $a$ (p=0.5)
:  $$\mathcal{L}_a = \dfrac{50!}{27! 23!} 0.5^{27} 0.5^{23} = 0.09596$$

The Likelihood under Model $b$ (p=27/50)
: $$\mathcal{L}_b = \dfrac{50!}{27! 23!} (27/50)^{27} (23/50)^{23} = 0.11263$$


$\mathsf{AIC}_a =  2k - 2 \ln \mathcal{L}_a = 2 - 2 \ln (0.09596)  = 6.6876$

$\mathsf{AIC}_b =  2k - 2 \ln \mathcal{L}_b = 2 - 2 \ln (0.11263)  = 6.3673$  $\leftarrow$ smaller preferred 



### AIC Probabilities
We identify the minimum: $\mathsf{m} = \mathsf{min}( \mathsf{AIC}_{a}, \mathsf{AIC}_{b}) = \mathsf{min}( 6.6876, 6.3673) = 6.3673 $

The Relative Likelihoods for AIC are:

$r_{\alpha} = \exp[\frac{1}{2}( m - \mathsf{AIC}_{\alpha})] = \exp[\frac{1}{2}( 6.3673 - 6.6876)] = 0.852$

$r_{\beta} = \exp[\frac{1}{2}( m - \mathsf{AIC}_{\beta})] = \exp[\frac{1}{2}( 6.6876 - 6.6876)] = 1$


The Akaike Weights are then:

$p_{\alpha} = r_{\alpha}/(r_{\alpha} + r_{\beta}) = 0.852/1.852 = 0.460$

$p_{\beta} = r_{\beta}/(r_{\alpha} + r_{\beta}) = 1/1.852 = 0.540$

The weight $p_{b}$ is the probability that $p=27/50$ is the best of the two options, given the data. 

Generally, if the difference $|\mathsf{AIC_{\alpha}} - \mathsf{AIC_{\beta}}| < 2 $ it is classified as "not worth more than a bare mention" - see [Kass 1995](https://sites.stat.washington.edu/raftery/Research/PDF/kass1995.pdf).


<!---
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

`scipy.optimize` provides several routines for minimising the NLL.
-->
 <!--For continuous distributions `scipy.stats.fit()` (which uses MLE by default) is the most convenient:-->

<!--
```{code-cell} python
import numpy as np
from scipy.stats import binom, expon
import scipy.optimize as opt

def binomial_mle_numerical(n=50, k=27):
    """Numerical MLE for Binomial p using minimize and fsolve"""
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


    print(f"True p: {true_p:.4f}")

binomial_mle_numerical()
```
-->
<!--
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
-->
<!--
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
-->

<!--
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
-->