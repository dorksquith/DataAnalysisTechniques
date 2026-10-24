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


(chapter:GenPDFs)=
# Generalised Distributions

## The Beta Distribution

The Beta distribution models probabilities. 

$X\sim$ Beta($a,b$)

The parameters of the Beta distribution are $a, b > 0$, which are known as **shape parameters** (or **hyperparameters** when the Beta is used as a prior distribution).

The Beta function $B(a,b)$ in the denominator is the normalisation constant:

```{math}
B(a,b) = \dfrac{\Gamma(a)\,\Gamma(b)}{\Gamma(a+b)}
```

where $\Gamma$ is the Gamma function (see [](#sec:gamma)).



### PDF & Kernel

The Beta PDF is:

```{math}
:label: eq:beta_pdf
f_X(x;\, a,b) = \dfrac{x^{a-1}(1-x)^{b-1}}{B(a,b)} \;\; ; \;\; 0 \leq x \leq 1
```

The **Support** of $X$ is $0 \leq x \leq 1$: all values of the measured probability $p$ with non-zero probability. 

The **Kernel** of the Beta PDF is given in [](#eq:beta_kernel).

```{math}
:label: eq:beta_kernel
x^{a-1}(1-x)^{b-1}
```

Plots of the Beta PDF for four different choices of the HyperParameters $a,b$ are shown in [](#fig:betaPDF) along with the CDFs.


:::{figure}
:label: fig:betaPDF

![image](figures/Beta.png)

The Beta PDFs are shown in blue with reference to the left-hand $y$-axis, and the CDFs are shown in red with reference to the right-hand $y$-axis. 
:::

Normalisation check
: The Beta PDF is valid by definition.


### Probabilities

The [CDF](#eq:continuous_cdf) is calculated as the integral over PDFs.

The Probability of measuring a probablity $X \leq x$, given some values of the parameters $a,b$, is:

```{math}
:label: eq:beta_cdf
F_X(x;\,a,b) = I_x(a,b)
```

Example
: The probability that a randomly selected person is less than 30% left-handed in their motor tasks, given prior belief Beta($a=2, b=5$):

  ```{math}
  P(X \leq 0.3) = I_{0.3}(2,5) \approx 0.580
  ```

  This regularised incomplete Beta function $I_x(a,b)$ has no simple closed form in general, but is computed by `scipy.stats.beta.cdf`.


### Expectation & Variance

The Beta PDF has
: $\label{eq:beta_expect}E[X] = \dfrac{a}{a+b}$

  $\label{eq:beta_var}V[X] = \dfrac{ab}{(a+b)^2(a+b+1)}$

The **Expectation** can be calculated by writing $E[X] = \int_0^1 x\, f_X(x)\, dx$ and using the identity $B(a+1,b)=\tfrac{a}{a+b}B(a,b)$:

$E[X] = \dfrac{1}{B(a,b)}\int_0^1 x^a (1-x)^{b-1}\, dx = \dfrac{B(a+1,b)}{B(a,b)} = \dfrac{a}{a+b}$

The **Variance** follows similarly using $E[X^2]$ and $V[X]=E[X^2]-E^2[X]$:

$V[X] = \dfrac{ab}{(a+b)^2(a+b+1)}$


### Python


```{code-cell} python

def ExploreBeta(a, b):

  # import the beta distribution
  from scipy.stats import beta

  # set up an instantiation of the distribution with our chosen parameter values a, b
  bee = beta(a, b)

  # access the lower bound and upper bound of the support
  lb, ub = bee.support()

  print(f"Support is {lb} <= x <= {ub}")

  # calculate the PDF at the midpoint of the support
  pdf_mid = bee.pdf(0.5)
  print(f"f_X(x=0.5) = {pdf_mid:.4f}")

  # calculate the cumulative probability up to the mean
  mean_val = a / (a + b)
  cdf1 = bee.cdf(mean_val)
  print(f"P(X<={mean_val:.4f}) = {cdf1:.4f}")

  # extract the true mean and variance
  mean, var = bee.stats(moments='mv')
  print(f"Expectation = {mean:.4f}, Variance = {var:.4f}")

ExploreBeta(2, 5)
```

### Likelihoods

The Beta distribution is most commonly encountered not as the distribution of observed data, but as a **prior distribution** on a probability parameter $p \in [0,1]$. It is the **conjugate prior** to the Bernoulli and Binomial likelihoods: if the prior on $p$ is Beta($a, b$), then after observing $n_1$ successes and $n_0$ failures, the posterior is still a Beta:

```{math}
:label: eq:beta_conjugate
p(\theta) = \text{Beta}(a,b) \quad \xrightarrow{\text{update with } n_1, n_0} \quad p(\theta\,|\,\mathbf{k}) = \text{Beta}(a + n_1,\; b + n_0)
```

This conjugate update follows directly from multiplying the Beta prior by the Bernoulli likelihood (from [](#eq:bern_kernel)):

```{math}
\underbrace{p(\theta\,|\,\mathbf{k})}_{\text{posterior}} \propto \underbrace{\theta^{n_1}(1-\theta)^{n_0}}_{\text{likelihood}} \times \underbrace{\theta^{a-1}(1-\theta)^{b-1}}_{\text{prior}} = \theta^{(a+n_1)-1}(1-\theta)^{(b+n_0)-1}
```

which is the kernel of Beta($a+n_1$, $b+n_0$). The parameters $a$ and $b$ can be interpreted as **pseudo-counts**: $a-1$ prior successes and $b-1$ prior failures.

Example
: A new blood test is trialled. Our prior belief on its sensitivity $p$ (true positive rate) is Beta($a=2, b=2$) — weakly centred on $p=0.5$. In a trial of $N=10$ patients known to have the disease, $n_1=8$ test positive.

  The posterior is Beta($2+8$, $2+2$) $=$ Beta($10$, $4$), giving:
  
  $E[p\,|\,\mathbf{k}] = \dfrac{10}{10+4} \approx 0.71$


(sec:gamma)=
## The Gamma Distribution


$X\sim$ Gamma($\alpha,\beta$)

The parameters are $\alpha > 0$ (shape) and $\beta > 0$ (rate). The Gamma distribution generalises the Exponential: $\text{Expon}(\lambda) = \text{Gamma}(1, \lambda)$, and the sum of $N$ iid Expon($\lambda$) RVs is Gamma($N$, $\lambda$).

### PDF & Kernel

The Gamma PDF (in the rate parameterisation) is:

```{math}
:label: eq:gamma_pdf
f_X(x;\,\alpha,\beta) = \dfrac{\beta^\alpha}{\Gamma(\alpha)}\, x^{\alpha-1}\, e^{-\beta x} \;\; ; \;\; 0 \leq x < \infty
```

where $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t}\, dt$ is the Gamma function ($\Gamma(n) = (n-1)!$ for positive integers).

The **Support** of $X$ is $0 \leq x < \infty$.

The **Kernel** of the Gamma PDF is:

```{math}
:label: eq:gamma_kernel
x^{\alpha-1}\, e^{-\beta x}
```

Normalisation check
: The Gamma PDF is valid by definition (the prefactor $\beta^\alpha/\Gamma(\alpha)$ is chosen to make it so).

Plots of the Gamma PDF for several choices of $\alpha$ and $\beta$ are shown in [](#fig:gammaPDF) along with the CDFs.

:::{figure}
:label: fig:gammaPDF

![image](figures/Gamma.png)

The Gamma PDFs are shown in blue with reference to the left-hand $y$-axis, and the CDFs are shown in red with reference to the right-hand $y$-axis. 
:::

### Probabilities

The [CDF](#eq:continuous_cdf) is calculated as the integral over the PDF.

The Probability of $X \leq x$, given parameters $\alpha, \beta$, is:

```{math}
:label: eq:gamma_cdf
F_X(x;\,\alpha,\beta) = \dfrac{\gamma(\alpha,\, \beta x)}{\Gamma(\alpha)} = P(\alpha,\, \beta x)
```

where $\gamma(\alpha, z) = \int_0^z t^{\alpha-1} e^{-t}\, dt$ is the lower incomplete Gamma function, and $P(\alpha, z)$ is the regularised form. Like the Beta CDF, this has no simple closed form in general and is evaluated numerically.

Example
: A call centre receives calls at a Poisson rate of $\lambda = 2$ per minute. The time until the 3rd call arrives is Gamma($\alpha=3, \beta=2$). The probability of waiting less than 2 minutes for the 3rd call:

  $P(X \leq 2) = P(3,\, 4) \approx 0.762$


### Expectation & Variance

The Gamma PDF has
: $\label{eq:gamma_expect}E[X] = \dfrac{\alpha}{\beta}$

  $\label{eq:gamma_var}V[X] = \dfrac{\alpha}{\beta^2}$

The **Expectation** is computed using the identity $\int_0^\infty x^\alpha e^{-\beta x}\, dx = \Gamma(\alpha+1)/\beta^{\alpha+1}$:

$E[X] = \dfrac{\beta^\alpha}{\Gamma(\alpha)} \cdot \dfrac{\Gamma(\alpha+1)}{\beta^{\alpha+1}} = \dfrac{\alpha}{\beta}$

The **Variance** follows from $V[X] = E[X^2] - E^2[X]$ using the same technique:

$V[X] = \dfrac{\alpha}{\beta^2}$

Note that for the special case $\alpha=1$ (i.e. Expon($\lambda$) with $\beta=\lambda$): $E[X]=1/\lambda$ and $V[X]=1/\lambda^2$, consistent with [](#sec:expon).


### Python

```{code-cell} python

def ExploreGamma(alpha, beta):

  # import the gamma distribution
  from scipy.stats import gamma

  # set up an instantiation of the distribution with shape alpha and rate beta
  # scipy uses the scale parameterisation: scale = 1/beta
  gam = gamma(a=alpha, scale=1/beta)

  # access the lower bound and upper bound of the support
  lb, ub = gam.support()

  print(f"Support is {lb} <= x <= {ub}")

  # calculate the cumulative probability of x <= alpha/beta (the mean)
  mean_val = alpha / beta
  cdf1 = gam.cdf(mean_val)
  print(f"P(X<={mean_val:.4f}) = {cdf1:.4f}")

  # extract the true mean and variance
  mean, var = gam.stats(moments='mv')
  print(f"Expectation = {mean:.4f}, Variance = {var:.4f}")

ExploreGamma(alpha=3, beta=2)

```


### Likelihoods

The Gamma distribution is the **conjugate prior** to the Exponential (and Poisson) likelihood. If the prior on the rate $\lambda$ is Gamma($\alpha, \beta$), then after observing $N$ iid waiting times $\mathbf{x}=(x_1,\ldots,x_N)$ from Expon($\lambda$), the posterior is also Gamma:

```{math}
:label: eq:gamma_conjugate
p(\lambda) = \text{Gamma}(\alpha, \beta) \quad \xrightarrow{\text{update with } \mathbf{x}} \quad p(\lambda\,|\,\mathbf{x}) = \text{Gamma}\!\left(\alpha + N,\;\; \beta + \sum_{i=1}^N x_i\right)
```

This follows from multiplying the Gamma prior by the Exponential likelihood (from [](#eq:expon_lik)):

```{math}
\underbrace{p(\lambda\,|\,\mathbf{x})}_{\text{posterior}}
\propto \underbrace{\lambda^N e^{-\lambda \sum x_i}}_{\text{likelihood}}
\times \underbrace{\lambda^{\alpha-1} e^{-\beta\lambda}}_{\text{prior}}
= \lambda^{(\alpha+N)-1}\, e^{-(\beta + \sum x_i)\lambda}
```

which is the kernel of Gamma($\alpha+N$, $\beta + \sum x_i$). The hyperparameter $\alpha$ can be interpreted as a prior count of events, and $\beta$ as the prior total waiting time.

Example
: Our prior belief about the rate of hitch-hiker pickups is Gamma($\alpha=2, \beta=1$) — suggesting a typical rate near 2 per hour. We observe 5 waiting times summing to 4.5 hours ($\sum x_i = 4.5$).

  The posterior is Gamma($2+5$, $1+4.5$) = Gamma($7$, $5.5$), giving:

  $E[\lambda\,|\,\mathbf{x}] = \dfrac{7}{5.5} \approx 1.27 \text{ per hour}$

  The posterior mean is a compromise between our prior ($E[\lambda]=2$) and the MLE ($\hat\lambda = 5/4.5 \approx 1.11$).

