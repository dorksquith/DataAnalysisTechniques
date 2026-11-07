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
# Linear Regression

* `numpy` methods: `linalg.inv`
* `scipy.optimize` methods: `least_squares`

Fitting means estimating the parameters, and sometimes also the functional form, that best describes a dataset. A functional form can have one parameter or lots of them.

The **Least Squares** approach minimises the [Residuals](#sec:ls-residuals) for all parameters simultaneously, to provide the best set of parameter estimates. The parameter estimates from the Least Squares Method are the **Least Squares Estimates (LSE)**, and are often indicated with the notation $\hat{\theta}_{\mathsf{LSE}}$ - akin to the Maximum Likelihood Estimates being denoted  $\hat{\theta}_{\mathsf{MLE}}$.  When the measurement uncertainties are Gaussian, minimising the sum of squared residuals is identical to minimising the [Negative Log Likelihood](#eq:nll), so $\hat{\theta}_{\mathsf{LSE}} \equiv \hat{\theta}_{\mathsf{MLE}}$ for Gaussian Errors. 

An example of a Least Squares Fit (straight line) is shown in [](#fig:lsq) for an example of Plant growth as a function of time. The control variable is the number of days since planting ($x$-axis). The control variable has evenly spaced intervals and negligible uncertainty. The Random Variable is the measured plant growth ($y$-axis) which has measurement uncertainty indicated by the vertical error bars on the data points.

:::{figure} 
:label: fig:lsq

![](figures/LinearLeastSquaresExample_a0.3_b2.0_noise1.0_outliers4.png)

The Least Squares fit (red dashed line) is a simultaneous estimation of the parameters of the function $m(x)$. In this case the function is a straight line, $m(x)=ax+b$, with parameter estimates $\hat{a}=-4.17$ (the gradient or slope of the line) and $\hat{b}=0.70$ (the intercept: the value of $y(x)$ when $x=0$). Least Squares estimates these parameters by minimising^[minim] the sum of the squared Residuals. The Residuals are shown as blue dotted lines for each data point.
:::

[^minim]: minimising a function means setting the first derivative of the function to zero, and checking the second derivative is positive. See [](#sec:minim_intro).





(sec:ls-residuals)=
## Residuals

The differences between each data point $d(x_i)$[^n1] and the Least Squares fit function at that point $m(x_i;\,\hat{\theta})$ are the **Residuals**:

$d(x_i) - m(x_i;\,\hat{\theta})\;\;\; \rightarrow \;\;\; d_i - m_i$,

with the simplified notation, $d_i - m_i$, chosen for clarity below.


[^n1]: $d(x_i)$ are the $y$-values of the black points in [](#fig:lsq))

[^n2]: $m(x_i;\,\hat{\theta})$ are the $y$-values of the red dashed line in [](#fig:lsq), where $\hat{\theta}={\hat{a}, \hat{b}}$ are the parameter estimates from the fit.

Data points must always be provided with error bars, of course, because all measurements have uncertainty! We denote the uncertainty on each data point as $\sigma_i$. To take account of these uncertainties on the data points, we can Standardise the Residuals (divide each residual by the measurement uncertainty):

```{math}
:label: eq:ls_standresid

r_i = \dfrac{ d_i - m_i }{ \sigma_i}

```

To consider all the data, rather than a single point at a time[^seeL] we could just add all of the residuals together... but this would be a mistake. If some of them are positive and some of them are negative, they could cancel out and give us a tiny total residual, despite the data, $d$, being very far away from the Least Squares fit model, $m$.

[^seeL]: Compare with the calculation of Likelihoods (week 6 lecture) - when we calculate the Likelihood from many data points we combine them by multiplying the PMFs, because the Multiplication Rule tells us that $P(A\cap B)= P(A) P(B)$ for IID $A,B$.


If we **Square the Residuals** before summing over them, this cancellation problem goes away[^pro].

[^pro]: Another option would be to take the absolute values of the residuals, rather than squaring them. Laplace used this approach, before Gauss came along with the idea to square them - see for example [Stephen Sigler's paper](https://doi.org/10.1214/aos/1176345451). The absolute value can be a good choice for reducing the "pull" of outliers - see [](#sec:ls-outliers).

The **Sum of Squared Residuals** is then:

```{math}
:label: eq:ls_ssresid

\sum_i r^2_i = \sum\limits^N_i \left( \dfrac{d_i - m_i}{\sigma_i}\right)^2

```

The statistic defined in [](#eq:ls_ssresid) is known as **Pearson's Chi Squared**, $\chi^2$, where the sum is over $N$ measurements, $d_i \pm \sigma_i$ and model predictions, $m_i$ (with estimated parameters $\hat{\theta} = {\hat{a}, \hat{b}}$).


:::{tip} Fun Fact
Gauss came up with the Normal distribution to justify the Least Squares method. See [this nice blog](https://notarocketscientist.xyz/posts/2023-01-27-how-gauss-derived-the-normal-distribution/) for the story.
:::


(sec:ls-chisquared)=
## The Chi Squared Statistic, $\chi^2$

Rewriting [](#eq:ls_ssresid) with the notation $\chi^2 \equiv \sum\limits_i^N r^2_i$, The **Chi Squared Statistic** is:

```{math}
:label: eq:chisq
\chi^2 = \sum_{i=1}^{N} \left(\frac{d_i - m_i}{\sigma_i}\right)^2.
```

If we think of each **measurement $d_i \pm \sigma_i$** as representative of a normal distribution, Norm($d_i, \sigma_i$), we can generate the "data-centric" picture given in [](#fig:lsq_gaus).


:::{figure} 
:label: fig:lsq_gaus

![](figures/LinearLeastSquaresGaussian_a3.0_b2.0_noise1.0_outliers4_modelcentered_False.png)

Data-centric picture: each measurement $d_i \pm \sigma_i$ represents the mean and standard deviation of a Normal($d_i, \sigma_i$) distribution.
:::

If instead we think of each **fit point $m_i \pm \sigma_i$** as representative of a normal distribution, Norm($d_i, \sigma_i$), we can generate the "model-centric" picture given in [](#fig:lsq_gaus_model). Note that we are using the measurement uncertainty $\sigma_i$ at each data point as an estimator for the uncertainty on the model fit point.


:::{figure} 
:label: fig:lsq_gaus_model

![](figures/LinearLeastSquaresGaussian_a3.0_b2.0_noise1.0_outliers4_modelcentered_True.png)

Model-centric picture: each fit point $m_i \pm \sigma_i$ represents the mean and standard deviation of a Normal($m_i, \sigma_i$) distribution, where we are using the measured uncertainty on the data point as before for $\sigma_i$.
:::


Either of the choices are perfectly valid, with the [data-centric picture](#fig:lsq_gaus) perhaps being more intuitive, as it keeps the uncertainties attached to the data. We will use the [model-centric picture](#fig:lsq_gaus_model) to show that miniminising the Negative Log Likelihood is equivalent to minimising the Pearson Chi Squared statistic.

The Norm$(m_i, \sigma_i)$ PDF is:

```{math}
:label: eq:norm_pdf_lsq
f_Y(y) = \dfrac{1}{\sigma_i\sqrt{2\pi}}\exp\!\left[-\frac{1}{2}\left(\frac{y-m_i}{\sigma_i}\right)^2\right] 
```

The Likelihood (of the $N$ data measurements $d_i$ being described by model parameters $\theta_i = m_i, \sigma_i$) is then:

```{math}
:label: eq:norm_lik_lsq

\begin{aligned}

\mathcal{L}_{\theta}(\theta;d) &= \prod_{i=1}^{N}\dfrac{1}{\sigma_i\sqrt{2\pi}}\exp\!\left[-\dfrac{1}{2}\left(\dfrac{d_i-m_i}{\sigma_i}\right)^2\right]\\ \\

\end{aligned}
```

The Negative Log Likelihood is[^note]:

```{math}
:label: eq:chisq_ll
\begin{aligned}

\mathsf{NLL} = -\ln \mathcal{L} &= \sum\limits_i^N \ln(\sigma_i) + \sum\limits_i^N \frac{1}{2}\ln(2\pi) +\sum\limits_i^N \left[\dfrac{1}{2}\left(\dfrac{d_i-m_i}{\sigma_i}\right)^2\right] \\
& =  \sum\limits_i^N \ln\sigma_i +\frac{N}{2}\ln(2\pi) + \frac{1}{2}\chi^2.

\end{aligned}
```


[^note]: In [](#eq:norm_nll) we considered $\sigma$ as a fixed value (a parameter of the true underlying distribution), so the Likelihood product (over densities) gave us the term $N\ln \sigma$. In this case, the $\sigma$ are (potentially) different in each case, as they represent measurement uncertainties. So the term is $\sum\limits_i^N \ln \sigma_i$


Minimising the NLL over $\theta$ is therefore *identical* to minimising $\chi^2$.  If all uncertainties are equal (the *homoscedastic* case - see [](#fig:lsq_hetero_homo) ) the $\sigma_i$ terms are constant, so $\hat\theta_{\rm LS} = \hat\theta_{\rm MLE}$.

> **Note:** If we also optimise over the $\sigma_i$ individually the MLE gives $\hat\sigma_i = 0$ and $\chi^2 = 0$, which is trivially perfect but useless.  Least squares is interesting when the $\sigma_i$ are fixed by the measurement process.


::::{figure}
:label: fig:lsq_hetero_homo

:::{list-table}
:header-rows: 0

* - ![](figures/Homo.png)
  - ![](figures/Hetero.png)
:::

Homoscedastic data have equal variances ($\sigma_i = \sigma$ for all $i$); heteroscedastic data do not.
::::



(sec:ls-binned)=
## Chi Squared with Binned Data

When data are binned (e.g. a histogram), the observed count $n_i$ in each bin follows a Poisson distribution ([](#sec:poisson)). An attempt to illustrate this is shown in [](#fig:lsq_poisson_data). The $K\sim$ Poisson($\lambda$) distribution has expectation $E[K]=\lambda$ and variance $V[K]=\lambda$, where the parameter $\lambda$ is the rate, e.g. the count per bin. The **count in each bin, $n_j$** is therefore the mean, $\lambda$, of a Poisson distribution, and the **uncertainty on the count in each bin, $\sigma_j$** is the square root of the Poisson variance: $\sqrt{V[K]} = \sqrt{\lambda} =  \sqrt{n_j}$. We can generate the "data-centric" picture given in [](#fig:lsq_poisson_data).


:::{figure} 
:label: fig:lsq_poisson_data

![](figures/LeastSquaresPoisson_modelcentFalse.png)

Data-centric picture of histogrammed (binned) data with Poisson uncertainties on the "counts per bin" RV.
:::


Just as we did the unbinned (Gaussian $\sigma$) example, we can instead think of each **fit point $m_j \pm \sigma_j$** as representative of a Poisson distribution, Poisson($m_j$), and generate the "model-centric" picture given in [](#fig:lsq_poisson_model). 

:::{figure} 
:label: fig:lsq_poisson_model

![](figures/LeastSquaresPoisson_modelcentTrue.png)

Model-centric picture of histogrammed (binned) data with Poisson uncertainties on the "counts per bin" RV. The uncertainties $\sigma_j = \sqrt{n_j}$ are now applied to the model fit points $\hat{n}_j$ at the centre of each bin.

:::

Let's consider the Chi Squared test statistic for the binned data:


```{math}
:label: eq:chisq-binned
\chi^2 = \sum\limits_j^{Nbins} \left( \frac{n_j - \hat{n}_j}{\sqrt{n_j}} \right)^2,
```

where $\hat{n}_j$ is the model prediction for bin $j$.


The **binned** standardised residuals [](#eq:ls_standresid) are often written in terms of the observed counts ($n_j$) and the expected counts ($\hat{n}_j$), as:

```{math}
:label: eq:chisq-binned-pop
 \frac{n_j - \hat{n}_j}{\sqrt{n_j}} = \frac{\mathsf{observed} - \mathsf{expected}}{\mathsf{\sqrt{observed}}}
```



(sec:ls-straightline)=
## Estimating the Parameters of a Straight Line

In the (unbinned) examples above, [](#fig:lsq) and [](#fig:lsq_gaus_model), the model is $m(x; \hat{\theta}) = \hat{a}x + \hat{b}$: a straight line. The parameter estimates $\hat{\theta} = {\hat{a}, \hat{b}}$ are made via the **Method of Least Squares, (LS)**. We will briefly cover the math here.


Just as for the [Maximum Likelihood Method](#chapter:MaxLike), we find the parameter estimates by minimising the objective function. For the MLE, the objective function is the [NLL](#eq:nll). For Least Squares, the objective function is the $\chi^2$.

For a straight line model $m(x; a,b) = ax + b$, the Chi Squared statistic is:

```{math}
:label: eq:chisq-line
\chi^2 = \sum_{i=1}^{N} \left( \frac{d_i - (ax_i + b) }{\sigma_i} \right)^2.
```

Expanding the square in each term of the numerator:

```{math}
:label: eq:chisq-line-expanded
\chi^2 = \sum_{i=1}^{N} \frac{1}{\sigma_i^2}
\bigl(d_i^2 + a^2 x_i^2 + b^2 + 2ab\,x_i - 2b\,d_i - 2a\,x_i\,d_i\bigr).
```

:::{tip}
There is going to be some quite "wordy" algebra below - let's make it more manageable using shorthand notation for the sums:

```{math}
:label: eq:ls-sums
s_1 = \sum_i \frac{1}{\sigma_i^2}, \quad
s_x = \sum_i \frac{x_i}{\sigma_i^2}, \quad
s_{xx} = \sum_i \frac{x_i^2}{\sigma_i^2}, \quad
s_y = \sum_i \frac{d_i}{\sigma_i^2}, \quad
s_{xy} = \sum_i \frac{x_i d_i}{\sigma_i^2}
```
:::



Just as for the MLE, we find the LS parameter estimates by taking the (partial) derivatives of the objective function (with respect to each of the parameters we seek to estimate) and setting them to zero. 


[1] Take the derivative of [](#eq:chisq-line-expanded) with respect to $a$
: $$ \label{eq:chisq_deriv_a} \frac{\partial \chi^2}{\partial a}= \sum_i \frac{1}{\sigma_i^2}(0 + 2ax_i^2 + 0 + 2bx_i - 0 - 2x_id_i)$$




[2] Set [](#eq:chisq_deriv_a) to zero and rearrange
: $$\label{eq:chisq_deriv_a_r}\sum_i \frac{x_id_i}{\sigma_i^2}= \hat a \sum_i \frac{x_i^2}{\sigma_i^2} + \hat b \sum_i \frac{x_i}{\sigma_i^2}$$

[3] Tidy up with our [shorthand notation](#eq:ls-sums)
: $$\label{eq:chisq_deriv_a_s}s_{xy} = \hat a \, s_{xx}+ \hat b\, s_x$$

The same three steps for the derivative with respect to the other parameter, $b$, give us:

```{math}
:label: eq:chisq_deriv_b_s
s_y = \hat a\, s_x + \hat b\,   s_1
```






<!--
```{math}
:label: eq:chisq_deriv_b
\frac{\partial \chi^2}{\partial b}
= \sum_i \frac{1}{\sigma_i^2}(0 + 0 + 2b + 2ax_i - 2d_i - 0) = 0.
```
-->




The simultaneous equations [](#eq:chisq_deriv_a_s) and [](#eq:chisq_deriv_b_s) (known as the **Normal Equations**) can be solved to find the parameter estimates.

We eliminate $\hat b$ as follows:

1. Multiply [](#eq:chisq_deriv_a_s) by $s_1$: $s_{xy}\, s_1 = \hat a \, s_{xx}\, s_1 + \hat b\, s_x\, s_1$

2. Multiply [](#eq:chisq_deriv_b_s) by $s_x$: $s_y s_x = \hat a\, s^2_x + \hat b\, s_x\,  s_1$

3. Subtract the two equations to eliminate solve for $\hat a$:

$$\label{eq:lsq_a} s_{xy} s_1  - s_y s_x  = (s_{xx} s_1 - s_x^2)\,\hat a $$

Following equivalent steps to solve for $\hat b$, we find:

$$\label{eq:lsq_b}
s_{xx} s_y - s_x s_{xy} = (s_{xx}s_1  - s_x^2)\,\hat b
$$



**The second derivatives** of [$\chi^2$](#eq:chisq-line-expanded) with respect to the parameters are:

```{math}
:label: eq:chisq_deriv_ab
\frac{\partial^2\chi^2}{\partial a^2} = \sum_i \frac{2x_i^2}{\sigma_i^2} = 2s_{xx}, \qquad
\frac{\partial^2\chi^2}{\partial b^2} = \sum_i \frac{2}{\sigma_i^2} = 2s_1, \qquad
\frac{\partial^2\chi^2}{\partial a\,\partial b} = \sum_i \frac{2x_i}{\sigma_i^2} = 2s_x.
```

So the **Hessian matrix** [](#sec:mma_hessian) is:

```{math}
:label: eq:chisq_hess
H = 2
\begin{bmatrix}
s_{xx} & s_x \\[6pt]
s_x & s_1 \\
\end{bmatrix}
```


The Hessian/2 has **determinant**[^det] $\Delta = \left( s_{xx}s_1 - s_x^2\right)$. We can recognise $\Delta$ as the common factor in [](#eq:lsq_a) in [](#eq:lsq_b), and write our parameter estimates as:

[^det]: The determinant of a matrix $\begin{bmatrix} A & B \\ C & D \\ \end{bmatrix}$ is $AD-BC$.

```{math}
:label: eq:ls-ahat
\hat a = \frac{ s_{xy} s_1 - s_y s_x }{\Delta},
```



```{math}
:label: eq:ls-bhat
\hat b = \frac{s_{xx} s_y - s_x s_{xy}}{\Delta}.
```

<!--
### Solution by Cramer's rule

The two equations can be written in matrix form as

$$
\begin{bmatrix} s_{xx} & s_x \\ s_x & s_1 \end{bmatrix}
\begin{bmatrix} \hat m \\ \hat c \end{bmatrix}
=
\begin{bmatrix} s_{xy} \\ s_y \end{bmatrix}.
$$

The determinant of the coefficient matrix is $\Delta = s_{xx}s_1 - s_x^2$.  Cramer's rule replaces each column in turn by the right-hand side vector and divides by $\Delta$:

$$
\hat m = \frac{1}{\Delta}\begin{vmatrix} s_{xy} & s_x \\ s_y & s_1 \end{vmatrix}
= \frac{s_{xy}s_1 - s_x s_y}{\Delta},
\qquad
\hat c = \frac{1}{\Delta}\begin{vmatrix} s_{xx} & s_{xy} \\ s_x & s_y \end{vmatrix}
= \frac{s_{xx}s_y - s_x s_{xy}}{\Delta}.
$$
-->
Note that $\Delta=s_{xx}s_1 - s_x^2$ depends only on $x_i$ and $\sigma_i$, not on the data values $d_i$:

<!--
```{math}
\begin{aligned}
\Delta &= s_{xx}s_1 - s_x^2 \\
       &= \sum_i \frac{x_i^2}{\sigma_i^2} \sum_j \frac{1}{\sigma_j^2} - \sum_i \frac{x_i}{\sigma_i^2}\sum_j \frac{x_j}{\sigma_j^2} \\

\end{aligned}
```
-->




<!--
```{math}
:label: eq:delta_homo
\begin{aligned}
\Delta &= \frac{1}{\sigma^4}\left( N\sum_i x_i^2 - \left(\sum_i x_i \right)^2 \right)\\
       &= \frac{1}{\sigma^4}\left( N\sum_i x_i^2 - N^2\bar{x}^2 \right) \\[6pt]
       &= \frac{N}{\sigma^4}\left( \sum_i x_i^2 - N\bar{x}^2 \right) \\[6pt]
       &= \frac{N}{\sigma^4}\sum_i (x_i - \bar{x})^2
\end{aligned}
```

 the numerators and denominators of [](#eq:ls-ahat) [](#eq:ls-bhat) have cancelling powers $\sigma$.


::::{dropdown} Show parameter estimates for homoscedastic data
In the homoscedastic case ($\sigma_i = \sigma$), the sums simplify to
$s_1 = N/\sigma^2$, $s_x = N\bar{x}/\sigma^2$, $s_{xx} = \sigma^{-2}\sum_i x_i^2$,
$s_y = N\bar{d}/\sigma^2$, $s_{xy} = \sigma^{-2}\sum_i x_i d_i$,
and $\Delta = \frac{N}{\sigma^4}\sum_i(x_i-\bar{x})^2$.

Substituting into [](#eq:ls-ahat):

```{math}
:label: eq:ls-ahat-homo
\begin{aligned}
\hat a &= \frac{s_{xy}s_1 - s_y s_x}{\Delta} \\[6pt]
       &= \frac{\dfrac{N}{\sigma^4}\sum_i x_i d_i
                - \dfrac{N^2\bar{x}\bar{d}}{\sigma^4}}
              {\dfrac{N}{\sigma^4}\sum_i(x_i-\bar{x})^2} \\[10pt]
       &= \frac{\sum_i x_i d_i - N\bar{x}\bar{d}}
              {\sum_i(x_i-\bar{x})^2} \\[6pt]
       &= \frac{\sum_i(x_i-\bar{x})(d_i-\bar{d})}
              {\sum_i(x_i-\bar{x})^2}
\end{aligned}
```

For $\hat b$, the second normal equation [](#eq:chisq_deriv_b_s) in the homoscedastic case gives
$\bar{d} = \hat a\,\bar{x} + \hat b$, so immediately:

```{math}
:label: eq:ls-bhat-homo
\hat b = \bar{d} - \hat a\,\bar{x}.
```
::::
-->


(sec:ls-uncertainties)=
### Parameter Uncertainties

The propagation of uncertainties is treated in full in [Propagation of Uncertainties](#sec:dd-propuncert). Here we apply the [Taylor expansion](#eq:mma_taylor) method to the least-squares problem to obtain the uncertainties on the fitted parameters $\hat{\theta} = \{ \hat a, \hat b \}$.


$\chi^2(\theta) = \chi^2(\hat{\theta}) + \dfrac{\partial \chi^2}{\partial \theta}\bigg|_{\hat{\theta}} \, (\theta-\hat{\theta}) + \dfrac{1}{2!}\dfrac{\partial^2 \chi^2}{\partial \theta^2}\bigg|_{\hat{\theta}}\, (\theta-\hat{\theta})^2 + ...$


The first term
: $\chi^2(\hat{\theta})$ is the statistic evaluated at the estimated parameters, $a=\hat{a}$ and $b=\hat{b}$

The second term (first order derivative)
: $\dfrac{\partial \chi^2}{\partial \theta}\bigg|_{\hat{\theta}} =0$: the first derivative of the statistic evaluated at the estimated parameter values is zero by definition. This is because the derivative of a function at its minimum (or maximum) is zero.

The third term (second order derivative)
: $\dfrac{\partial^2 \chi^2}{\partial \theta^2}\bigg|_{\hat{\theta}}$: is the Hessian matrix of partial derivatives [](#eq:chisq_hess), evaluated at the parameter estimates.

   

Setting the second term to zero, and noting $(\theta-\hat{\theta}) = \begin{bmatrix} a-\hat a \\ b-\hat b \end{bmatrix}$, we have:

```{math}
:label: eq:chisq-taylor
\chi^2(a,b) = \chi^2(\hat a,\hat b) +
\frac{1}{2!}
\begin{bmatrix} a-\hat a \\ b-\hat b \end{bmatrix}^{\!\top}
\mathsf{H}
\begin{bmatrix} a-\hat a \\ b-\hat b \end{bmatrix},
```


The factor of $\tfrac{1}{2!}$ cancels with the factor of $2$ in [$\mathsf{H}$](#eq:chisq_hess), so the quadratic form simplifies to

```{math}
:label: eq:ls_chi2_taylor
\chi^2(a,b) = \chi^2(\hat a,\hat b) +
\begin{bmatrix} a-\hat a \\ b-\hat b \end{bmatrix}^{\!\top}
\begin{bmatrix} s_{xx} & s_x \\ s_x & s_1 \end{bmatrix}
\begin{bmatrix} a-\hat a \\ b-\hat b \end{bmatrix}.
```

The **Covariance Matrix** of the estimates is the inverse of the matrix in [](#eq:ls_chi2_taylor):

```{math}
:label: eq:ls_hess_hetero
\boldsymbol{\Sigma} = \begin{bmatrix} s_{xx} & s_x \\ s_x & s_1 \end{bmatrix}^{-1}
= \frac{1}{\Delta}\begin{bmatrix} s_1 & -s_x \\ -s_x & s_{xx} \end{bmatrix},
```

giving

```{math}
:label: eq:ls-cov
\sigma^2_{\hat a} = \frac{s_1}{\Delta}, \qquad
\sigma^2_{\hat b} = \frac{s_{xx}}{\Delta}, \qquad
\operatorname{cov}(\hat a,\hat b) = -\frac{s_x}{\Delta}.
```

The *linear correlation coefficient* between $\hat a$ and $\hat b$ is

$$
\rho(\hat a,\hat b) = \frac{\operatorname{cov}(\hat a,\hat b)}{\sigma_{\hat a}\,\sigma_{\hat b}}.
$$

A negative correlation is typical: if the slope is steeper than expected, the intercept tends to be lower to compensate.

<!--
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
-->



(sec:ls-homo)=
### Homoscedastic Least Squares 

If we follow the procedure above but with the assumption that our data are homoscedastic - constant variance - we have a simpler set of tasks.


:::{tip}
In the **homoscedastic** case ($\sigma_i = \sigma$ for all $i$), the shorthand sums [](#eq:ls-sums) simplify to:


```{math}
:label: eq:ls-sums-homo
s_1 = \frac{N}{\sigma^2}, \quad
s_x = \frac{N\bar{x}}{\sigma^2}, \quad
s_{xx} = \frac{1}{\sigma^2}\sum_i x_i^2, \quad
s_y = \frac{N\bar{d}}{\sigma^2}, \quad
s_{xy} = \frac{1}{\sigma^2}\sum_i x_i d_i
```

:::

<!--
Simultaneous Equations
: $\sum x_i y_i =a \sum x^2_i + b \sum x_i$
  
  $\sum y_i =a \sum x_i + Nb$
-->

Hessian/2
: $\mathsf{H/2} =  \begin{bmatrix} s_{xx} & s_x \\[6pt] s_x & s_1 \end{bmatrix} = \dfrac{1}{\sigma^2} \begin{bmatrix} \sum x_i^2 & N\bar{x} \\[6pt] N\bar{x} & N \end{bmatrix}$

Determinant
: $\Delta = \left( s_{xx} s_1 - s_x^2 \right) = \dfrac{1}{\sigma^4} \left( N \sum x^2_i - N^2\bar{x}^2 \right)$

Covariance Matrix = Inverse Hessian/2
: $\mathsf{H/2}^{-1} = \dfrac{1}{\Delta} \begin{bmatrix} s_1 & -s_x \\[6pt] -s_x & s_{xx} \end{bmatrix} = \dfrac{1}{\sigma^2\Delta} \begin{bmatrix} N & -N\bar{x} \\[6pt] -N\bar{x} & \sum x_i^2 \end{bmatrix} $

<!--
$\chi^2$ Statistic 
: $E[\chi^2] = E\left[ \sum\limits_i^N \dfrac{1}{\sigma^2} (d_i - m_i) \right]
             = \dfrac{N}{\sigma^2} E\left[ \sum\limits_i^N (d_i - \hat{a}x_i + \hat{b}) \right]
 = \nu = N-2$
  
  $V[\chi^2] = 2\nu =2(N-2)$

  $\nu = \dfrac{(d_i-m_i)^2}{\sigma^2_y} $
-->

Parameter Uncertainties
: $\sigma^2_{\hat{a}} = \dfrac{s_1}{\Delta} = \dfrac{N}{\sigma^2\Delta},\;\;\;\;\;\; \sigma^2_{\hat{b}} = \dfrac{s_{xx}}{\Delta} = \dfrac{\sum x_i^2}{\sigma^2\Delta}$


---

(sec:ls-python)=
## LSE with Python 

:::{note} `scipy.optimize` method choice
`scipy.optimize` provides several optimisation routines; the right choice depends on the problem structure:

| Function | Use when | Returns |
|----------|----------|---------|
| `least_squares` | Minimising a **sum of squares** $\\sum_i r_i^2$; supports robust losses and returns the Jacobian for covariance estimation | `result.x`, `result.jac`, `result.fun` |
| `minimize` | Minimising an **arbitrary scalar** function (e.g.\ a negative log-likelihood); general-purpose | `result.x`, optionally `result.hess_inv` |
| `minimize_scalar` | Minimising a **1-D scalar** function of a single variable only | `result.x` |
| `fsolve` | Finding **roots** of a system of equations $f(x)=0$ (e.g.\ score equations); not an optimiser | array of root values |
:::

Let's generate some fake data that follows $y(x) = ax + b$ but has some added noise to make it more realistic.

```{code-cell} python
import numpy as np
from numpy.random import default_rng

def gen_data(seed=None):

  rng = default_rng(seed)

  # True parameter values
  a = -3.0
  b =  2.0

  # independent variable x
  x = np.linspace(0, 10, 10)

  # Gaussian Noise term: standard normal (sigma=1)
  #noise = 0.1 * rng.standard_normal(x.size)
  noise = rng.standard_normal(x.size)

  # Fake data followin y(x) = ax + b with added noise
  data = a * x + b + noise 

  return x, data


# only set the seed for demos!
xvals, data = gen_data(seed=123)

print(f"data measurements: {data}")
```

Always plot the data before doing anything else, even when it is fake!


```{code-cell} python
import matplotlib.pyplot as plt

def plot_data(xvals, data):

  fig, ax = plt.subplots()

  ax.scatter(xvals, data, color='black', label="data")
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  plt.legend()
  plt.show()


# only set the seed for demos!
plot_data(xvals, data)

```



Now let's do a by-hand implementation of a least squares fit ([homoscedastic](#sec:ls-homo)) to find the LSE for the parameters a and b:


```{code-cell} python
def byhand_LSE(x, data):

  # the number of data points
  N = len(data)

  # the shorthand sums for the homoscedastic case
  s1  = N
  sxx = np.sum(x**2)
  sx  = np.sum(x)
  sy  = np.sum(data)
  sxy = np.sum(x * data)

  # determinant
  delta = N * sxx - sx**2

  # parameter estimates
  a_est = (sxy * N  - sx * sy)  / delta
  b_est = (sxx * sy - sxy * sx) / delta

  model = a_est * x + b_est

  # residuals - not standardised
  resids    = data - model

  # number of degrees of freedom = Ndata - Nparams
  nu = N - 2

  # homoscedastic sigma^2
  sigma2  = np.sum(resids**2) / nu

  # parameter uncertainties
  sigma_a = np.sqrt(sigma2 * N    / delta)
  sigma_b = np.sqrt(sigma2 * sxx  / delta)
  cov_ab  = sigma2 * (-sx / delta)
  rho_ab  = cov_ab / (sigma_a * sigma_b)

    
  print(f"Slope      a: {a_est:.4f}  ±  {sigma_a:.4f}")
  print(f"Intercept  b: {b_est:.4f}  ±  {sigma_b:.4f}")
  print(f"cov(a,b):  {cov_ab:.4f}")
  print(f"rho(a,b):  {rho_ab:.4f}")

  return a_est, b_est, sigma_a, sigma_b, cov_ab

a_est, b_est, sigma_a, sigma_b, cov_ab = byhand_LSE(xvals, data)

```

Let's plot the fit result on top of the fake data:


```{code-cell} python


def plot_data_and_fit(x, data, a_est, b_est, sigma_a, sigma_b, cov_ab):

  import matplotlib.pyplot as plt

  fig, ax = plt.subplots()

  ax.scatter(x, data, color='black',label="data d(x)")
  model = a_est * x + b_est

  # Propagated uncertainty on model prediction at each x 
  sigma_model = np.sqrt(x**2 * sigma_a**2 + sigma_b**2 + 2 * x * cov_ab)

  ax.plot(x, model,'r--', label=f"model m(x) = {a_est:.2f}x + {b_est:.2f}")

  ax.fill_between(x, model-sigma_model, model+sigma_model, alpha=0.25, color='red')

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  plt.legend()
  plt.show()

plot_data_and_fit(xvals, data, a_est, b_est, sigma_a, sigma_b, cov_ab)

```

The uncertainty band (shaded red) indicates the uncertainty on the model prediction at each value of the independent variable, $x$. We calculate this using Propagation of Uncertainty for a linear function of two RVs [](#sec:dd_linear2).

<!--
:::{dropdown} Show Reminder: Taylor Method

For a model $m(x; a,b) = ax +b$ where $x$ is known and $a,b$ are unknown, we treat $a,b$ as the RVs.

The Jacobian matrix $\mathsf{J}$  and its transpose $\mathsf{J^T}$ are:

```{math}
\mathsf{J} =  \begin{bmatrix} \dfrac{\partial m}{\partial a} &  \dfrac{\partial m}{\partial b} \end{bmatrix},\;\;\; \mathsf{J^T} = \begin{bmatrix} \dfrac{\partial m}{\partial a} \\[6pt]  \dfrac{\partial m}{\partial b} \end{bmatrix}
``` 

The Covariance matrix $\mathsf{\Sigma}$ is:

```{math}
\Sigma  = \begin{bmatrix} \sigma^2_{\hat{a}} & \mathsf{cov}(\hat{a},\hat{b})\\
 \mathsf{cov}(\hat{a},\hat{b}) & \sigma^2_{\hat{b}} \\
 \end{bmatrix} 
``` 

So the variance on the function (the model) is:

```{math}
\begin{aligned}
V[m] &= \begin{bmatrix} \dfrac{\partial m}{\partial a} &  \dfrac{\partial m}{\partial b} \end{bmatrix}
\begin{bmatrix} \sigma^2_{\hat{a}} & \mathsf{cov}(\hat{a},\hat{b})\\
 \mathsf{cov}(\hat{a},\hat{b}) & \sigma^2_{\hat{b}} \\
 \end{bmatrix} 
 \begin{bmatrix} \dfrac{\partial m}{\partial a} \\[6pt]  \dfrac{\partial m}{\partial b} \end{bmatrix} \\[6pt]
 &= 
\dfrac{\partial m}{\partial a}\bigg|^2_{\hat{a}} \sigma^2_{\hat{a}} + \dfrac{\partial m}{\partial b}\bigg|^2_{\hat{b}} \sigma^2_{\hat{b}} + 2 \dfrac{\partial m}{\partial a} \dfrac{\partial m}{\partial b} \mathsf{cov}(\hat{a},\hat{b})
\end{aligned}
```

For this model, we have $\dfrac{\partial m}{\partial a} = x\;\;\; \dfrac{\partial m}{\partial b} = 1$.

:::
-->

:::{tip}
For a model $m(x; a,b) = ax +b$ where $x$ is known and $a,b$ are unknown, we treat $a,b$ as the RVs.
:::

The Expectation [](#eq:exp-f2-exp) is $E[m] = E[ax+b] = xE[\hat{a}] + E[\hat{b}]$

The Variance [](#eq:var-f2-exp) is $V[m] = x^2 V[\hat{a}] + V[\hat{b}] + 2x\, \mathsf{cov}(\hat{a},\hat{b})$

The model uncertainty is therefore:

```{math}
\sigma_m = \sqrt{ x^2 \sigma^2_{\hat{a}} + \sigma^2_{\hat{b}} + 2x\, \mathsf{cov}(\hat{a},\hat{b}) }
``` 

indicated in the code as:

```
# Propagated uncertainty on model prediction at each x 
  sigma_model = np.sqrt(x**2 * sigma_a**2 + sigma_b**2 + 2 * x * cov_ab)
```

The output of `byhand_LSE` includes the covariance between the parameters, `cov_ab`, and we see it is negative. This makes sense: if we over-estimate the intercept parameter $\hat{a}$, the fit has to under-estimate the slope parameter $\hat{b}$ to compensate.

Notice that the uncertainty band is narrowest towards the center of the plot. This narrowest region corresponds to the minimum of the model uncertainty. The minimum of a function is found by taking its first derivative:


```{math}
\dfrac{d}{dx} \sigma^2_m = \dfrac{d}{dx} \left( x^2 \sigma^2_{\hat{a}} + \sigma^2_{\hat{b}} + 2x\, \mathsf{cov}(\hat{a},\hat{b})\right) = 2x\,\sigma^2_{\hat{a}} + 2\,\mathsf{cov}(\hat{a},\hat{b}) 
``` 

This is equal to zero (indicating a minimum, a maximum, or an inflection point) when:

```{math}
x = -  \dfrac{\mathsf{cov}(\hat{a},\hat{b})}{\sigma^2_{\hat{a}}} = \dfrac{s_x}{s_1} = \bar{x},
``` 

confirming that the uncertainty is at a minimum[^notemin] at the mean value of the independent variable, $\bar{x}$.

[^notemin]: We confirm this is a minimum, rather than a maximum or inflection point, by taking the second derivative $\dfrac{d^2}{dx^2} \sigma^2_m  = 2\sigma^2_{\hat{a}}$: this is positive, so we have a minimum.

---

(sec:ls-scipy)=
### LSE with `scipy.optimize.least_squares` 

The [`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) method allows us to find the LSE for parameters without doing doing any of the math above, but it is very easy to get it wrong without some understanding of what we have covered in this chapter. The method fits (estimates the parameters of) a model by minimising the sum of squares of a user-supplied **unstandardised** residuals function: the numerator $d_i - m_i$ of the standardised residual given in [](#eq:ls_standresid)

The method returns a `result` object from which we can get interesting things such as:

`r = result.fun`
: The vector of (unstandardised) residuals evaluated at the LSE for the parameters

`J = result.jac` 
: the Jacobian matrix of partial derivatives of the (unstandardised) residuals function, at the LSE for the parameters 

`theta = result.x`
: the vector of LSE parameter estimates.


**Important Note about `scipy`'s `least_squares`**

:::{warning}
Because `scipy`'s `least_squares` does not take a standardised residuals function, the Jacobian returned by the method is missing the $1/\sigma_i^2$ weights on the partial derivatives we calculated above e.g. [](#eq:chisq_deriv_a). This means that when we construct the Hessian with `H = J.T@J` it is not the same as the $\mathsf{H}/2$ from [](#eq:chisq_hess), which has $1/\sigma_i^2$ weights on all its elements.
:::


<!-- The scipy statistic `result.fun` is not the standardised $\chi^2$ statistic from [](#eq:chisq-line), but -->

The **unweighted sum of squared residuals** is:

$$
S = \sum_i (d_i - m_i)^2.
$$

In the homoscedastic case ($\sigma_i = \sigma$: the uncertainty on all measurements is equal), after fitting $Z=2$ parameters, the quantity $Q = S/\sigma^2$ follows a $\chi^2$ distribution with $\nu = N - Z$ degrees of freedom, whose mean is $\nu$.

$Q \sim \chi^2(\nu) \;\;\;\; E[Q] = E[\frac{S}{\sigma^2}] = \nu $ - see [](#eq:exp_chisq)

Multiplying the expectation by the constant $\frac{\sigma^2}{\nu}$ yields $E\left[\frac{S}{\nu}\right] = \sigma^2$, so $S/\nu$ is an unbiased estimate of $\sigma^2$: the formula is using the scatter of the residuals to estimate the uncertainty on the measurements, $\sigma$, when this is unknown.  

**When constructing the Covariance Matrix using the `result.jac` returned by `least_squares`, we must multiply through by the estimator for $\sigma^2$:  $\frac{S}{\nu}$**, where $\nu = N-N_p$.

:::{seealso}
When using `scipy.optimize.minimize` for MLE, the inverse Hessian of the log-likelihood — $[-\partial^2 \ln\mathcal{L}/\partial\theta^2]^{-1}$, which equals the parameter covariance matrix — is available directly as `result.hess_inv` when `method='BFGS'` is specified. See [](#sec:ml_scipy_minimize) for details and caveats.
:::

```
# Vector of unstandardised residuals: [d_i - m_i] =  [d_i - (est_a * x_i + est_b) ]
r = result.fun

# Sum of squared unstandardised residuals
S = np.sum(r**2)

# Estimated sigma^2 from  E[\frac{S}{\sigma^2}] = \nu
sigma2_est = S / nu

# Jacobian of partial first derivatives without 1/\sigma^2 weights
J = result.jac

# Hessian of partial second derivatives without 1/\sigma^2 weights
H = J.T @ J

# Inverse of Hessian without \sigma^2 weights
Hinv = np.linalg.inv(H)

# Covariance Matrix: all terms must be multiplied through by sigma2_est
cov  = Hinv * sigma2_est 
```



Now that we have very clearly heeded the warnings above, let's fit!


```{code-cell} python

def scipy_LSE(x, data):
  from scipy.optimize import least_squares

  # residuals function, parameters are theta = [a, b]
  def residuals(theta, x, data):
      # model: y = ax + b
      model = theta[0] * x + theta[1]
      # residual (not standardised!) : (data - model ) 
      return data - model

  # we are estimating two parameters: a and b
  N_params = 2

  # number of degrees of freedom = N - N_params
  nu = len(data) - N_params

  # choose some starting values for the parameters a=1,b=1
  theta0 = np.array([1.0, 1.0])

  # call the fit function
  # first argument - residuals - is the d-m function we defined above
  # second argument - theta0 - is the array of starting parameter values for the fit
  # third argument - args=(x, data) - tells the fit that the x,data arguments of the residuals function are "spectators", we are not trying to estimate their values 

  result = least_squares(residuals, theta0, args=(x, data))

  # Vector of unstandardised residuals: [d_i - m_i] =  [d_i - (est_a * x_i + est_b) ]
  r = result.fun

  # Sum of squared unstandardised residuals
  S = np.sum(r**2)

  # Estimated sigma^2 from  E[\frac{S}{\sigma^2}] = \nu
  sigma2_est = S / nu

  # Jacobian of partial first derivatives without 1/\sigma^2 weights
  J = result.jac

  # Hessian of partial second derivatives without 1/\sigma^2 weights
  H = J.T @ J

  # Inverse of Hessian without \sigma^2 weights
  Hinv = np.linalg.inv(H)

  # Covariance Matrix: all terms must be multiplied through by sigma2_est
  cov  = Hinv * sigma2_est 
  var   = np.diagonal(cov)            # [sigma_a^2, sigma_b^2]
  sigma = var**0.5                    # [sigma_a, sigma_b]   
  cov_ab = cov[0, 1]
  rho_ab = cov_ab / (sigma[0] * sigma[1])


  print(f"Slope      a: {result.x[0]:.4f}  ±  {sigma[0]:.4f}")
  print(f"Intercept  b: {result.x[1]:.4f}  ±  {sigma[1]:.4f}")
  print(f"cov(a,b):  {cov_ab:.4f}")
  print(f"rho(a,b):  {rho_ab:.4f}")

scipy_LSE(xvals, data)
```

Note that the `scipy_LSE` function returns exactly the same parameters and uncertainties as the `byhand_LSE` function above.

---

(sec:ls-outliers)=
### Extreme outliers and robust least squares

Because the residuals appear **squared** in $\chi^2$, a single outlier can dominate the fit and pull the estimated parameters far from their true values.


**Robust Least Squares** replaces the squared residuals with a loss function that is less sensitive to large deviations.  The *Least Absolute Deviations* (LAD) method uses $|d_i - m_i|$ instead of $(d_i - m_i)^2$.  `scipy.optimize.least_squares` provides the `loss` argument with several built-in options:

| `loss`     | Effect |
|-----------|--------|
| `'linear'` | Standard LS (default) |
| `'soft_l1'`| Smooth approximation to LAD |
| `'huber'`  | Quadratic near zero, linear in the tails |
| `'cauchy'` | Very robust, maximally down-weights outliers |

**Why is the default loss called `'linear'`?** The loss function is characterised by the *rho function* $\\rho(z)$, where $z = r_i^2$ is the squared residual.  For the default `'linear'` choice, $\\rho(z) = z$, which is linear in $z$ — hence the name.  This gives the ordinary sum of squares $\\sum_i \\rho(r_i^2) = \\sum_i r_i^2$.  The model itself can still be linear *or* nonlinear in $x$; the word “linear” here refers to the rho function, not to the functional form of $m(x)$.  For `'soft_l1'`, $\\rho(z) = 2(\\sqrt{1+z}-1)$, which grows more slowly for large $z$ and therefore down-weights outliers.

:::{note} Loss function vs.\ distributional assumption
The `loss` argument and the distributional assumption of the model are **two separate choices**:

* **`loss`** controls *outlier robustness* — how much a large residual is allowed to influence the parameter estimates.  It says nothing about the distribution of the measurement uncertainties.
* **Distributional assumption** determines *which likelihood* underpins the fit.  Gaussian errors → standard LS (minimise $\\chi^2$).  Non-Gaussian errors require a different model.

For non-Gaussian/non-Poisson measurements, a **Generalised Linear Model (GLM)** provides the appropriate framework.  `statsmodels.genmod` implements GLMs for the most common families:

| Distribution | Typical use | `statsmodels` family |
|--------------|-------------|----------------------|
| Gaussian | Continuous measurements with additive noise | `families.Gaussian()` |
| Poisson | Count data (events per bin, photons, etc.) | `families.Poisson()` |
| Binomial | Binary outcomes, proportions | `families.Binomial()` |
| Gamma | Positive-valued, skewed continuous data | `families.Gamma()` |

Using `loss='soft_l1'` with standard `least_squares` does **not** make your model Poisson or Binomial; it only changes how outliers are weighted.  To fit a Poisson model properly, maximise the Poisson log-likelihood (see [](#chapter:MaxLike)).
:::

The `f_scale` parameter is crucial for all `loss` functions other than `linear`. It sets the transition scale between the near-quadratic and near-linear regimes.


Let's generate some more fake data, this time having a quadratic form, $ y(x) = ax + bx^2$. We will make two of our data points into outliers for this example by inflating the noise on their positions by a factor $10^3$.

```{code-cell} python

def gen_quad_data(seed=None):

  rng = default_rng(seed)

  # True parameter values
  a = 0.5
  b = 2.0
  true_params = [a,b]

  # independent variable x
  x = np.linspace(-10, 10, 15)

  # Gaussian Noise term: 1% (to make fake data more realistic)
  noise = 0.01 * rng.standard_normal(x.size)
  
  # Add a couple of outliers with inflated residuals
  outliers = rng.integers(0, x.size, 2)
  noise[outliers] *= 10000

  # Fake data following y(x) = ax + bx^2 with added noise
  data = a * x + b * x**2 + noise 

  return x, data, true_params


# only set the seed for demos!
xvals_quad, data_quad, true_params = gen_quad_data(seed=123)

#print(f"data measurements: {data_quad}")
```

As usual, we will plot the fake data next. We are using `scatter` rather than `errorbar` in the python snippet below, because we are assuming we have no information about the uncertainty on our fake data measurements.


```{code-cell} python
import matplotlib.pyplot as plt

def plot_quad_data(xvals, data):

  fig, ax = plt.subplots()

  ax.scatter(xvals, data, color='black', label="data")
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  plt.legend()
  plt.show()


# only set the seed for demos!
plot_quad_data(xvals_quad, data_quad)

```

We can see the data follow a quadratic shape, as expected given the choice we made for generating the fake data. We can also see the very obvious outliers: two points that do not seem to follow the same function as the others.

**Now let's fit!** We will define two models this time. One will use the default `least_squares`, which uses `linear` as the loss function. The other will use `loss='soft_l1'` and `f_scale=0.1`.


```{code-cell} python
def LSE_fit(x, data):
  from scipy.optimize import least_squares

  def residuals_quad(p, x, y):
      return p[0] * x + p[1] * x**2 - y

  p0 = np.array([1.0, 1.0])

  # standard LS
  result_linear  = least_squares(residuals_quad, p0, args=(x, data))

  # robust LS with loss function soft_l1
  result_softl1  = least_squares(residuals_quad, p0, args=(x, data), loss='soft_l1', f_scale=0.1)

  print("True:        a={:.3f}, b={:.3f}".format(true_params[0], true_params[1]))
  
  print("Standard LS: a={:.3f}, b={:.3f}".format(*result_linear.x))
  
  print("soft_l1 LS:  a={:.3f}, b={:.3f}".format(*result_softl1.x))

  # --- Covariance matrices from Jacobian ---
  nu = len(x) - len(p0)  # degrees of freedom = N - N_params

  # Standard LS: exact covariance, Cov = (J^T J)^{-1} * (S / nu)
  J_lin = result_linear.jac
  S_lin = np.sum(result_linear.fun**2)
  cov_lin = np.linalg.inv(J_lin.T @ J_lin) * (S_lin / nu)

  # Robust LS: approximate covariance (Jacobian is computed on the
  # re-weighted residuals, so the resulting Cov is approximate)
  J_sl1 = result_softl1.jac
  S_sl1 = np.sum(result_softl1.fun**2)
  cov_sl1 = np.linalg.inv(J_sl1.T @ J_sl1) * (S_sl1 / nu)

  return result_linear.x[0], result_linear.x[1], result_softl1.x[0], result_softl1.x[1], cov_lin, cov_sl1

a_lin, b_lin, a_sl1, b_sl1, cov_lin, cov_sl1 = LSE_fit(xvals_quad, data_quad)

```

Notice that we have printed the parameter estimates **rounded up** to 3 decimal places using `:.3f` in the f-string. For this demo I chose 3 decimal places because I wanted to keep it as short as possible (3 significant figures = 2 decimal places is usually a good rule of thumb for this sort of thing). Using a precision of 2 decimal places I found the parameter estimates with `soft_l1` were identical to the true parameters, so I dialled up the precision as a sanity check.

Reminder: **never, ever round up estimates within calculations**. We only round up numerical values when we are displaying them as final results, and when we do that we must be thoughtful and consistent in the precision we give.

We see that the default `least_squares` method finds $m = -1.271 x + 2.162 x^2$, while the fit using `soft_l1` loss finds $m = 0.498 x + 2.000 x^2$. In this example with two extreme outliers we find that using a `soft_l1` loss results in significantly better parameter estimates.


**Now let's plot the fit results!** I will use a blue dotted line for `soft_l1` using the matplotlib shorthand `b:`, and a red dashed line for the default loss function using `r--`.

```{code-cell} python


def plot_data_quad_and_fit(x, data, a_lin, b_lin, a_sl1, b_sl1 ):

  import matplotlib.pyplot as plt

  fig, ax = plt.subplots()

  ax.scatter(x, data, color='black',label="data")

  model_lin = a_lin * x + b_lin * x**2

  ax.plot(x, model_lin ,'r--', label=f"linear LS")

  model_softl1 = a_sl1 * x + b_sl1 * x**2

  ax.plot(x, model_softl1 ,'b:', label=f"soft l1 LS")

  ax.set_xlabel('x')
  ax.set_ylabel('y')

  # ncols=3 makes the three legend entries appear in a row rather than in one column.
  plt.legend(loc='upper center', ncols=3)
  plt.show()

plot_data_quad_and_fit(xvals_quad, data_quad,a_lin, b_lin, a_sl1, b_sl1)

```

The `soft_l1` loss function with `f_scale=0.1` prevents the outliers from pulling the fit away from the more reliable data points, resulting in significantly better parameter estimates and an improved fit, visible by eye.

**Parameter uncertainties from the Jacobian** can be visualised as uncertainty bands on the model.
For the model $m(x) = \hat{a}x + \hat{b}x^2$, error propagation gives:

$$
\sigma^2_m(x) = x^2\,\sigma^2_{\hat{a}} + x^4\,\sigma^2_{\hat{b}} + 2x^3\,\mathsf{cov}(\hat{a},\hat{b}),
$$

where $\sigma^2_{\hat{a}}$, $\sigma^2_{\hat{b}}$ and $\mathsf{cov}(\hat{a},\hat{b})$ are the diagonal and off-diagonal elements of the 2×2 covariance matrix returned by `LSE_fit`.  For the robust `soft_l1` fit this covariance is **approximate**, as noted above.

```{code-cell} python
def plot_data_quad_fit_and_bands(x, data, a_lin, b_lin, a_sl1, b_sl1, cov_lin, cov_sl1):
  import matplotlib.pyplot as plt

  x_fine = np.linspace(x.min(), x.max(), 300)

  # Model predictions on a fine grid
  model_lin = a_lin * x_fine + b_lin * x_fine**2
  model_sl1 = a_sl1 * x_fine + b_sl1 * x_fine**2

  # 1-sigma band: sigma^2_m = x^2*cov[0,0] + x^4*cov[1,1] + 2*x^3*cov[0,1]
  sigma_lin = np.sqrt(
      x_fine**2 * cov_lin[0, 0] +
      x_fine**4 * cov_lin[1, 1] +
      2 * x_fine**3 * cov_lin[0, 1]
  )
  sigma_sl1 = np.sqrt(
      x_fine**2 * cov_sl1[0, 0] +
      x_fine**4 * cov_sl1[1, 1] +
      2 * x_fine**3 * cov_sl1[0, 1]
  )

  fig, ax = plt.subplots()
  ax.scatter(x, data, color='black', label='data')

  ax.plot(x_fine, model_lin, 'r--', label='linear LS')
  ax.fill_between(
      x_fine, model_lin - sigma_lin, model_lin + sigma_lin,
      color='red', alpha=0.15, label=r'linear LS $\pm1\sigma$'
  )

  ax.plot(x_fine, model_sl1, 'b:', label='soft l1 LS')
  ax.fill_between(
      x_fine, model_sl1 - sigma_sl1, model_sl1 + sigma_sl1,
      color='blue', alpha=0.15, label=r'soft l1 LS $\pm1\sigma$'
  )

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  plt.legend(loc='upper center', ncols=3)
  plt.show()

plot_data_quad_fit_and_bands(
    xvals_quad, data_quad, a_lin, b_lin, a_sl1, b_sl1, cov_lin, cov_sl1
)
```

Notice that the uncertainty bands on the robust `soft_l1` fit are much narrower than those on the standard fit.  The robust fit correctly identifies most points as inliers with small residuals, giving tight parameter estimates; the standard fit is dragged towards the outliers, inflating $S$ and thus $\hat{\sigma}^2 = S/\nu$ and the uncertainty bands.

:::{note} Gradient descent and stochastic gradient descent
**Gradient descent** is an iterative algorithm for minimising a loss function $L(\boldsymbol\theta)$ when no closed-form solution exists.  Starting from an initial guess $\boldsymbol\theta_0$, it updates the parameters by stepping in the direction of steepest descent:

$$\boldsymbol\theta_{j+1} = \boldsymbol\theta_j - \gamma\,\nabla_{\boldsymbol\theta} L(\boldsymbol\theta_j),$$

where $\gamma > 0$ is the **learning rate** (step size).  Choosing $\gamma$ too small makes convergence slow; too large and the algorithm overshoots or oscillates.  Many practical optimisers (including BFGS, used in [](#sec:ml_scipy_minimize)) adaptively rescale the gradient using curvature information.

In **Stochastic Gradient Descent (SGD)**, the gradient is estimated from a randomly selected **mini-batch** of data points rather than the full dataset.  Each update is noisier but computationally cheaper, and the stochasticity helps the algorithm escape shallow local minima.  SGD and its variants (Adam, RMSProp) are the backbone of modern **neural network** training, where the gradient of the loss with respect to millions of parameters is computed efficiently by **backpropagation** — a systematic application of the chain rule through the network’s layered structure.
:::

<!--
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


:::{figure} 
:label: fig:lsq_outliers

![](figures/robust-quadratic-ls.png)

Outliers.
:::

-->

(sec:ls-summary)=
## Summary

**Least Squares** minimises the Chi Squared statistic ([](#eq:chisq)) over the model parameters.  For Gaussian measurement uncertainties this is equivalent to maximising the Likelihood ([](#eq:chisq_ll)).

**The Chi Squared statistic**
: $\chi^2 = \sum_{i=1}^{N}\left(\dfrac{d_i - m_i}{\sigma_i}\right)^2$,
  where $d_i \pm \sigma_i$ are the data points and $m_i$ are the model predictions.

**Parameter estimates for a straight line** $m(x;a,b)=ax+b$ ([](#eq:ls-ahat), [](#eq:ls-bhat))
: $\hat a = \dfrac{s_{xy}s_1 - s_y s_x}{\Delta}, \qquad \hat b = \dfrac{s_{xx}s_y - s_x s_{xy}}{\Delta}$,
  where the shorthand sums $s_1,\,s_x,\,s_{xx},\,s_y,\,s_{xy}$ are defined in [](#eq:ls-sums) and $\Delta = s_{xx}s_1 - s_x^2$.

**Covariance matrix** ([](#eq:ls_hess_hetero), [](#eq:ls-cov))
: $\boldsymbol{\Sigma} = \dfrac{1}{\Delta}\begin{bmatrix}s_1 & -s_x \\ -s_x & s_{xx}\end{bmatrix},
  \qquad \sigma^2_{\hat a} = \dfrac{s_1}{\Delta},\quad
  \sigma^2_{\hat b} = \dfrac{s_{xx}}{\Delta},\quad
  \operatorname{cov}(\hat a,\hat b) = -\dfrac{s_x}{\Delta}.$
  The covariance matrix is the inverse of $\mathsf{H}/2$, where $\mathsf{H}$ is the Hessian of $\chi^2$ ([](#eq:chisq_hess)).

**Homoscedastic case** ($\sigma_i = \sigma$ for all $i$, [](#sec:ls-homo))
: The sums simplify ([](#eq:ls-sums-homo)) and $\Delta = \frac{1}{\sigma^4}(N\sum_i x_i^2 - N^2\bar{x}^2)$.
  When $\sigma$ is unknown it is estimated from the residuals: $\hat\sigma^2 = S/\nu$, where $S=\sum_i(d_i-m_i)^2$ and $\nu = N-2$.

**Using `scipy.optimize.least_squares`** ([](#sec:ls-scipy))
: The function minimises the **unweighted** residuals $r_i = d_i - m_i$ (no $1/\sigma_i$ weights).
  The Jacobian $\mathsf{J}$ returned by `result.jac` satisfies $\mathsf{J}^\top\mathsf{J} \neq \mathsf{H}/2$.
  The covariance matrix must be constructed as
  $\boldsymbol{\Sigma} = (\mathsf{J}^\top\mathsf{J})^{-1}\,\hat\sigma^2$, where $\hat\sigma^2 = S/\nu$.

**Robust Least Squares** ([](#sec:ls-outliers))
: When outliers are present, the `loss` argument of `scipy.optimize.least_squares`
  (`'soft_l1'`, `'huber'`, `'cauchy'`) replaces squared residuals with a function that grows
  more slowly for large deviations, reducing the influence of outliers on the parameter estimates.
  The scale parameter `f_scale` sets the transition between the near-quadratic and near-linear regimes.

<!--
Least squares minimises $\chi^2 = \sum_i (d_i - \mu_i)^2/\sigma_i^2$, which is equivalent to Gaussian MLE.

For a straight line $y = mx + c$, the analytical estimators are $\hat m = (s_{xy}s_1 - s_x s_y)/\Delta$ and $\hat c = (s_{xx}s_y - s_x s_{xy})/\Delta$ with $\Delta = s_{xx}s_1 - s_x^2$.

Parameter uncertainties come from the covariance matrix $\boldsymbol\Sigma = \mathbf{H}^{-1}$, giving $\sigma^2_{\hat m} = s_1/\Delta$, $\sigma^2_{\hat c} = s_{xx}/\Delta$, $\operatorname{cov}(\hat m,\hat c) = -s_x/\Delta$.

In `scipy.optimize.least_squares` the covariance is reconstructed as $(\mathbf{J}^\top\mathbf{J})^{-1} \times \chi^2/\nu$.

If data are heteroscedastic, or if outliers are present, standard OLS is no longer optimal and robust methods or weighted LS should be used.
-->
