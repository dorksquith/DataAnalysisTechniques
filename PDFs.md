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

(chapter:PDFs)=
# Continuous Probabilities & Likelihoods

(sec:uniform)=
## Uniform

A Continuous RV with a flat probability distribution is 

$X\sim$ Uniform($a,b$).

The parameters are $a,b$: indicating the lower and upper bound of the support.


### PDF & Kernel

The Uniform PDF is:

```{math}
:label: eq:uniform_pdf
f_X(x) = \dfrac{1}{b-a} \;\; ; \;\;  a\leq x \leq b
```

The **Support** of $X$ is $a \leq x \leq b$.


A plot of the Uniform PDF for chosen parameter values $a=-2,b=2$ is shown in [](#fig:uniformPDF) along with the Cumulative Distribution Function (CDF).


:::{figure}
:label: fig:uniformPDF

![image](figures/Uniform.png)

The Uniform PDF is shown in blue with reference to the left-hand $y$-axis, and the CDF is shown in red with reference to the right-hand $y$-axis. Because $X$ is a continuous RV, the left-hand $y$-axis is the Density rather than the Probability. The CDF is the Cumulative probability, which must total 1 as a sum over all possible outcomes.
:::


For a PDF to be valid, it must satistify [Kolmogorov's normalisation axiom](#eq:kolmogorov2).

Is $\int\limits_{x_i \in S}\, f_X(x_i)\, dx =1$ ?

Normalisation check
: $\int\limits_{x=a}^b\, \dfrac{1}{b-a}\, dx = \dfrac{b-a}{b-a} = 1$: the Uniform PDF is valid (by definition).


### Probabilities

The Cumulative Probability (CDF) for a continuous RV is the integral over the range of the lower bound of the support to the value at which we want to know the CDF.

The integral is over the PDFs calculated at each possible value, $x_i$.

```{math}
:label: eq:continuous_cdf
F_X(x) = P(X\leq x) = \int\limits_{-\infty}^{x}\, f_X(x) \, dx
```

The Uniform CDF is:

```{math}
:label: eq:uniform_cdf
F_X(x) = P(X\leq x) = \int\limits_{a}^{x}\, \dfrac{1}{b-a} \, dx = \dfrac{x-a}{b-a}
```

### Expectation & Variance

The Uniform PDF has
: $$ \label{eq:uniform_expect} E[X] =  \dfrac{b+a}{2}$$
  
  $$ \label{eq:uniform_var} V[X] =  \dfrac{(b-a)^2}{12}$$


The **Expectation** is calculated as:

$E[X]= \int\limits_{a}^{b}\, \dfrac{x}{b-a} \, dx = \dfrac{b^2-a^2}{2(b-a)}= \dfrac{b+a}{2}$

The expectation of the square is:

$ E[X^2] = \int\limits_{a}^{b}\, \dfrac{x^2}{b-a} \, dx =  \dfrac{b^3-a^3}{3(b-a)} =  \dfrac{b^2+ab+a^2}{3}$

The **Variance** is calculated as:

$V[X]  = E[X^2] - E^2[X] = \dfrac{b^2+ab+a^2}{3} -  \dfrac{b^2 + 2ab + a^2}{4} = \dfrac{(b-a)^2}{12}$


### Python

```{code-cell} python

def ExploreUniform(a,b):

  # import the uniform distribution
  from scipy.stats import uniform

  # set up an instantiation of the distribution with our chosen parameter values a,b
  uni =  uniform(loc=a, scale=b-a) 

  # access the lower bound and upper bound of the support
  lb, ub = uni.support()

  print(f"Support is {lb} <= x <= {ub}")

  # calculate the cumulative probability of x<=(a+b)/2
  cdf1 = uni.cdf((a+b)/2)
  print(f"P(X<={(a+b)/2}) = {cdf1}")

  # extract the true mean and variance
  mean, var = uni.stats(moments='mv')
  print(f"Expectation = {mean}, Variance = {var}")

ExploreUniform(0.45,5)

```


### Likelihoods

Suppose we observe $N$ measurements $\mathbf{x} = (x_1, x_2, \ldots, x_N)$ drawn independently from Uniform($a,b$), but the parameters $a$ and $b$ are unknown.

The Likelihood is the joint PDF evaluated at the observed data:

```{math}
:label: eq:uniform_lik
\mathcal{L}(a,b;\mathbf{x}) = \prod_{i=1}^{N} \dfrac{1}{b-a} = \dfrac{1}{(b-a)^N}
\qquad \text{subject to } a \leq x_i \leq b \;\forall\; i
```

Taking the log:

```{math}
\ln \mathcal{L}(a,b;\mathbf{x}) = -N\ln(b-a)
```

Unlike the PMF likelihoods, **we cannot differentiate** to find the MLE. This is because the support itself depends on the parameters: the constraints $a \leq x_i \leq b$ mean the likelihood is zero for any $a > x_{(1)}$ or $b < x_{(N)}$, where $x_{(1)} = \min(x_i)$ and $x_{(N)} = \max(x_i)$ are the **order statistics**.

Within the feasible region, $\ln\mathcal{L}$ is maximised by minimising $(b-a)$. The tightest feasible interval is:

```{math}
:label: eq:uniform_mle
\hat{a} = x_{(1)} = \min(x_i), \qquad \hat{b} = x_{(N)} = \max(x_i)
```

This makes intuitive sense: the widest the interval needs to be is just enough to include all observed values.


(sec:expon)=
## The Exponential Distribution


$X\sim$ Expon($\lambda$)

The parameter is $\lambda$, which is a rate parameter (similar to [](#sec:poisson) ). Note though that for Poisson($\lambda$), the RV $K$ is the count in a given time interval (a rate), whereas for Expon($\lambda$), the RV $X$ is a time for a given count.

### PDF & Kernel

The Exponential PDF is:

```{math}
:label: eq:expon_pdf
f_X(x) = \lambda e^{-\lambda x} \;\; ; \;\;  0\leq x \leq \infty
```

The **Support** of $X$ is $0 \leq x \leq \infty$.

The **Kernel** of the Exponential PDF is given in [](#eq:expon_kernel).

```{math}
:label: eq:expon_kernel
 e^{-\lambda x}
```

A plot of the Exponential PDF for chosen parameter values $\lambda=1.3$ is shown in [](#fig:exponPDF) along with the Cumulative Distribution Function (CDF).

:::{figure}
:label: fig:exponPDF

![image](figures/Expon.png)

The Exponential PDF is shown in blue with reference to the left-hand $y$-axis, and the CDF is shown in red with reference to the right-hand $y$-axis. The (rate) parameter value is $\lambda=1.3$, meaning the expected waiting time between successful events is the inverse of this: $1/1.3$. The CDF is the Cumulative probability, which must total 1 as an integral over all possible outcomes.
:::


For a PDF to be valid, it must satistify [Kolmogorov's normalisation axiom](#eq:kolmogorov2).

Is $\int\limits_{x_i \in S}\, f_X(x_i) \, dx =1$ ?

Because we have a nice easy integral (exponentials always are), we will just do it:

```{math}
\int\limits_0^{\infty}\, \lambda e^{-\lambda x} \, dx & = -\dfrac{\lambda}{\lambda} e^{-\lambda x} \bigg|_0^{\infty}\\
= - \left( e^{-\infty} - e^{0} \right) \\
= - \left( 0 - 1 \right) \\
= 1
```

Normalisation check
: $\int\limits_0^{\infty}\, \lambda e^{-\lambda x} \, dx = 1$: the
Exponential PDF is valid.


### Probabilities

The [CDF](#eq:continuous_cdf) is calculated as the sum over PDFs.

The Probability of $X \leq x$, given some value of the parameters $\lambda$, is:

```{math}
:label: eq:expon_cdf
P(X\leq x |\lambda) = \int\limits_{t=0}^{x}  \lambda e^{-\lambda t} \, dt = 1-e^{-\lambda x}

```

Example
: The probability that I will wait less than or equal to one hour to hitch a lift on a road with an average rate of hitch-hiker pickups of $\lambda= 1.3$ per hour.

  $P(X\leq 1 | \lambda) = 1-e^{-\lambda x} = 1-e^{-1.3} = 0.727$

  What would Poisson give us? The question is now "what is the probability that a count of at least one occurs in a time period of 1H?"

  $1 - P(K=0 | \lambda) = 1 - e^{-\lambda} \dfrac{\lambda^k}{k!}= 1- e^{-1.3}\dfrac{1.3^0}{0!} = 1- 0.273  = 0.727$


### Expectation & Variance

The Exponential PDF has
: $\label{eq:expon_expect}E[X] = \dfrac{1}{\lambda}$

  $\label{eq:expon_var}V[X] = \dfrac{1}{\lambda^2}$

The **Expectation** of Expon $X$ is an integral product of two terms in $x$, so we have to do it piecewise (integration by parts). This is an excellent exercise, try it if you have time. 

I used: $u = x, u'= 1, v = −e^{-\lambda x}, v' = \lambda e^{-\lambda x}$.

The **Variance** of Expon $X$ is also calculated using integration by parts - I found I needed two rounds of by-parts integration for the variance.



### Python

```{code-cell} python

def ExploreExpon(mu):

  # import the exponential distribution
  from scipy.stats import expon

  # set up an instantiation of the distribution with our chosen parameter value mu (for lambda, the rate parameter)
  
  expo = expon(scale=1/mu) 

  # access the lower bound and upper bound of the support
  lb, ub = expo.support()

  print(f"Support is {lb} <= x <= {ub}")

  # calculate the cumulative probability of x<=1/mu
  cdf1 = expo.cdf(1/mu)
  print(f"P(X<={1/mu}) = {cdf1}")

  # extract the true mean and variance
  mean, var = expo.stats(moments='mv')
  print(f"Expectation = {mean}, Variance = {var}")

ExploreExpon(12.3)

```

### Likelihoods

Suppose we observe $N$ waiting times $\mathbf{x} = (x_1, x_2, \ldots, x_N)$ drawn independently from Expon($\lambda$).

The Likelihood is the joint PDF evaluated at the observed data:

```{math}
:label: eq:expon_lik
\mathcal{L}(\lambda;\mathbf{x}) = \prod_{i=1}^{N} \lambda\, e^{-\lambda x_i}
= \lambda^N\, e^{-\lambda \sum_{i=1}^{N} x_i}
```

The log-likelihood is:

```{math}
:label: eq:expon_loglik
\ln\mathcal{L}(\lambda;\mathbf{x}) = N\ln\lambda - \lambda\sum_{i=1}^{N} x_i
```

Differentiating and setting to zero:

```{math}
\dfrac{d\ln\mathcal{L}}{d\lambda} = \dfrac{N}{\lambda} - \sum_{i=1}^{N} x_i = 0
\implies \hat{\lambda} = \dfrac{N}{\displaystyle\sum_{i=1}^{N} x_i} = \dfrac{1}{\bar{x}}
```

The MLE of the rate $\lambda$ is the reciprocal of the sample mean waiting time — exactly what we would expect from $E[X]=1/\lambda$.

Example
: A hitch-hiker records the following waiting times (in hours): $1.2, 0.5, 0.8, 1.6, 0.4$. The MLE of the rate is:

  $\hat{\lambda} = \dfrac{5}{1.2+0.5+0.8+1.6+0.4} = \dfrac{5}{4.5} \approx 1.11 \text{ per hour}$


## The Normal Distribution


## The $\chi^2$ Distribution


(sec:transforms)=
## Transforms: Changing Variables

In real life we often cannot measure the RV we want directly. We measure $X$ and want the PDF for $Y = f(X)$.

### 1D Monotonic Transform

Suppose $Y = f(X)$ is a **monotonically increasing** function of $X$ (no turning points), with inverse $X = g(Y)$.

Because $Y$ increases whenever $X$ increases:

```{math}
P(Y \leq y) = P(X \leq g(y)) = F_X(g(y))
```

Differentiating the CDF of $Y$ to get its PDF:

```{math}
f_Y(y) = \dfrac{d}{dy} F_Y(y) = \dfrac{d}{dy} F_X(g(y)) = \dfrac{dg}{dy}\, f_X(g(y)) = \dfrac{dx}{dy}\, f_X(x)
```

For a **monotonically decreasing** function the same argument applies with a sign flip, so in general:

```{math}
:label: eq:transform_1d
f_Y(y) = \left|\dfrac{dx}{dy}\right| f_X(x)
```

where $x = g(y)$ is the inverse function evaluated at $y$.

:::{note}
If $f(X)$ is **not** monotonic, we split the domain into intervals on which it is monotonic and sum the contributions. This is beyond the scope of this module.
:::

Example
: Suppose $X \sim \text{Expon}(\lambda)$ and we define $Y = 2X$ (scaling the waiting time). Then $x = y/2$, $dx/dy = 1/2$, and:

  $f_Y(y) = \dfrac{1}{2}\, \lambda\, e^{-\lambda y/2} = \dfrac{\lambda}{2}\, e^{-(\lambda/2)\, y}$

  This is $\text{Expon}(\lambda/2)$: halving the rate doubles the expected waiting time, as expected.

### Multivariate Transform and the Jacobian

For a multivariate RV $\mathbf{X} = (X_1, X_2, \ldots, X_Z)$ transformed to $\mathbf{Y} = (Y_1, Y_2, \ldots, Y_Z)$ via some set of relationships $Y_{(i)} = f(X_{(1)}, \ldots, X_{(Z)})$, the generalisation of [](#eq:transform_1d) involves the **Jacobian**:

```{math}
:label: eq:transform_jacobian
f_{\mathbf{Y}}(\mathbf{y}) = \dfrac{1}{|J|}\, f_{\mathbf{X}}(\mathbf{x})
```

where $J$ is the **determinant** of the $Z\times Z$ matrix of partial derivatives:

```{math}
:label: eq:jacobian
J = \begin{vmatrix}
\dfrac{\partial y_1}{\partial x_1} & \cdots & \dfrac{\partial y_1}{\partial x_Z} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial y_Z}{\partial x_1} & \cdots & \dfrac{\partial y_Z}{\partial x_Z}
\end{vmatrix}
```

Example: Cartesian to Polar
: Points $(x,y)$ are uniformly distributed in a plane. We transform to polar coordinates $(r,\theta)$ via $r = \sqrt{x^2+y^2}$, $\theta = \arctan(y/x)$.

  The partial derivatives give:

  $\dfrac{\partial r}{\partial x} = \dfrac{x}{r}, \quad \dfrac{\partial r}{\partial y} = \dfrac{y}{r}, \quad \dfrac{\partial \theta}{\partial x} = -\dfrac{y}{r^2}, \quad \dfrac{\partial \theta}{\partial y} = \dfrac{x}{r^2}$

  The Jacobian determinant is:

  $J = \dfrac{x}{r}\cdot\dfrac{x}{r^2} - \dfrac{y}{r}\cdot\left(-\dfrac{y}{r^2}\right) = \dfrac{x^2+y^2}{r^3} = \dfrac{1}{r}$

  So $f(r,\theta) = r\, g(x,y)$: if $(x,y)$ are uniform (constant $g$), then $f(r,\theta) \propto r$, meaning the density in polar coordinates increases linearly with $r$. This makes geometric sense — concentric rings of equal width have area $\propto r$.


## Summary of Kernels

```{table} Uniform, Expon, Norm,  ChiSquared

|  Name | Kernel  | E[X] | V[X] | Support | Parameters  | `scipy.stats` |
| ---   |  ---     | ---  | ---  | ---     |  ---        |  ---          |
| Uniform($a,b$) | $1$ | $\frac{b+a}{2}$ | $\frac{(b-a)^2}{12}$ | $[a,b]$ | $a,b$ | `uniform(loc=a,scale=b-a)` |
| Expon($\lambda$) | $e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ | $[0,\infty)$ | $\lambda$ | `expon(scale=1/λ)` |
| Normal($\mu,\sigma$) | $e^{-x^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | $(-\infty,\infty)$ | $\mu,\sigma$ | `norm(loc=μ,scale=σ)` |
| $\chi^2(k)$ | $x^{k/2-1}e^{-x/2}$ | $k$ | $2k$ | $[0,\infty)$ | $k$ | `chi2(df=k)` |



```


