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

# Describing Data

## Mean and Expectation

The **Sample Mean**, $\overline{x}$, of a RV is calculated as a normalised sum over a finite number of measurements [](#eq:mean). It is a **summary statistic** calculated from the data.


The **True Mean** (aka **Expectation**), $E[X]$, is a **parameter** of the true underlying probability distribution of our RV. 

### Discrete Data

For [Discrete data](#discrete), if we known the underlying probabilities we can calculate the Expectation in a similar way to [the sample mean](#eq:mean):

```{math}
:label: eq:expect
E[X]  = \sum\limits_i^\infty  x_i p_i
```

* The sum is infinite because in Truth, there are $\infty$ possible values for X.
* The Probability of observing $x_i$ is $p_i$. Because Probabilities must sum (or integrate) to 1 [Kolmogorov's normalisation axiom](#eq:kolmogorov2), the normalisation factor $\dfrac{1}{N}$ is not needed. Via use of the probability $p_i$ we are normalising every term in the sum individually, and will get the same result.


### Continuous Data

If our data is Continuous rather than Discrete, then it does not make sense to "sum over all possible values", because continuous RVs have an Uncountable Infinity[^infinities] of possible values. Instead, we integrate:

[^infinities]: See this nice blog by Joel Hamkins for an accessible intro to the difference between countable and [uncountable infinites](https://www.infinitelymore.xyz/p/uncountable-infinity)

```{math}
:label: eq:expectC

E[X]  = \int\limits_{-\infty}^\infty  x_i\, f_X\, dx
```

Note that in [](#eq:expectC) we have $f_X$: a **Probability Density Function (PDF)** instead of the individual probabilities $p_i$ used in [](#eq:expect). Lots more on PDFs later.


:::{figure} 
:label: fig:mean-expect-mp4
:align: left
![](figures/MeanExpect.mp4)

A series of images showing a data histogram in grey and the true underlying PDF as a red dashed line. The size of the dataset is increased by a factor 10 for each step in the sequence of images.
:::



### Notation: $E[X]  \equiv \mu$

> An alternative notation for the True Mean, aka Expectation, is $\mu$. This has  exactly the same meaning as $E[X]$. I'm sorry this is confusing, but I think there are strong (pedagogical and aesthetic) arguments for using both forms.

```{math}
:label: eq:expectMu

E[X]  \equiv \mu
```

(ExpectationAlgebra)=
### Expectation Algebra

Some very useful properties of Expectation:

The expectation of a sum (difference) is the sum (difference) of the expectations. 
$$\label{eq:expect_sum} E(X \pm Y) = E(X) \pm E(Y)$$

If we add a constant $b$ to a RV, we add the same constant to its expecation.
$$\label{eq:expect_addc} E(X + b) = E(X) + b$$
 
If we multiply an RV by a constant $a$ , we multiply its expecation by the same constant.
$$\label{eq:expect_multc} E(aX) = aE(X)$$
 

For the special case of Independent RVs only, the expectation of the product is the product of the expectations.

$$\label{eq:expect_multind} E(XY) = E(X) E(Y) $$

[](#eq:expect_multind) **is only true if X and Y are mutually independent**. See [](#eq:multrule).


(LLN)=
## The Law of Large Numbers (LLN)

In [](#fig:mean-expect-mp4) we see that the Sample Mean $\overline{x}$ does not exactly correspond to the True Mean $E[X]$.

The **Law of Large Numbers (LLN)**[^1] tells us that if we were to increase the size of the data sample to infinity (not possible in real life), the sample mean $\overline{x}$ would approach the true mean $\mu$ with high probability.

[^1]: This is the Weak LLN. The Strong LLN is subtly different, explained quite nicely on [wikipedia](https://en.wikipedia.org/wiki/Law_of_large_numbers).

```{math}
:label: eq:lln

\lim\limits_{N\to \infty} P(| \overline{x} - \mu | \geq \alpha) =0\;\; \mathsf{for}\;\; \alpha > 0 
```

The symbol $\alpha$  is a small number of our choosing, which we can think of as a "tolerance level" for the difference between the sample mean and the true mean.

As is often the case with Statistics, it is helpful to think about what the LLN is **not** telling us. 

>The LLN does not imply that if we add some new measurements to a dataset, the sample mean will (necesarily) get closer to the true mean. We can observe fluctuations of $\overline{x}$ away from $\mu$ for any number of measurements $N<\infty$. 

We can see a fluctuation in the image series [](#fig:mean-expect-mp4) when we compare the $N=1k$ dataset with the $N=10k$ dataset. This is examined in [](#fig:fluc).

:::{figure} 
:label: fig:fluc
:align: left
![](/figures/LLN-fluc.png)

For $N=1k$ measurements, we observe $|\overline{x} - \mu |=0.0044$. For $N=10k$ measurements, we observe $|\overline{x} - \mu |=0.0088$.
:::


The mean measured in the smaller 1k sample is closer to the expectation $\mu$ than the mean of the 10k sample. This is a natural **Statistical Fluctuation**, and does not imply that the LLN is wrong.


## Variance and Standard Deviation

The [sample **Standard Deviation**](#eq:std) is the square root of the [sample **Variance**](#eq:variance). You would be forgiven for wondering why we don't just pick one of these summary statistics and ditch the other for simplicity; we keep this redundancy because they are each crucial in their own worlds, as we shall see.

The problem with the sample variance is that we can't relate it directly to the measurement, because it has units of $x^2$ rather than $x$. As such, we cannot provide a visual description of the variance on axes with units of $x$. But we can draw the sample Standard Deviation $\sigma_x = \sqrt{V[x]}$.


:::{figure} 
:label: fig:var-std-cartoon
:align: left
![](figures/variance-cartoon)

Cartoon showing $N=5$ measurements of X, with the standard deviation $\sigma_x$ indicated on the plot.
:::

It is the sample Standard Deviation that we use to quantify the spread of our data points, giving us an intrinsic **Uncertainty** on each measurement. This is usually provided visually as error bars or bands.


### True Variance and Standard Deviation

The most basic definition of the [sample variance](#eq:variance) is with respect to the sample mean. Analogous to the [true mean](#eq:expect), if we know the underlying probabilities we can write down the true variance as:

$$\label{eq:truevar1} V[x]  = \sum\limits_i^{\infty}  (x_i - \overline{x})^2\, p_i$$

To write [](#eq:truevar1) in a more useful form, we can then follow these logical steps:

Sample Mean to True Mean
: Use the expectation $\mu \equiv E[x]$ in place of the sample mean $\overline{x}$
  
  $(x_i - \overline{x})^2\;\; \rightarrow \;\;(x_i - \mu)^2$

Expectation definition
: Use [](#eq:expect) to write:  
  
  $E[X^2]  = \sum\limits_i^\infty  x^2_i p_i$
  
  $E[(X-\mu)^2]  = \sum\limits_i^\infty  (x_i-\mu)^2 p_i$

The above steps allow us to write the true variance in terms of the expectation:

```{math}
:label: eq:truevar

V[X] = E[(X-\mu)^2]
```

Or, with [a bit of algebra](#a-bit-of-algebra), in the equally useful form:

```{math}
:label: eq:truevar2

V[X] = E[X^2] - E^2[X]
```

:::{dropdown} Proof that [](#eq:truevar) and [](#eq:truevar2) are equivalent
:label: a-bit-of-algebra
```{math}
\begin{aligned}
V[X] & = E[\, X^2 + E^2[X] - 2X\,E[X] \,]\\
	& = E[\, X^2\,] + E[\, E^2[X] \,]- E[\,2X\,E[X] \,]\;\; because\; E[A+B] = E[A] + E[B] \\
	& = E[ X^2] + E^2[X] - 2E[X]  E[X]\;\; because\; E[E[A]] = E[A]\\
	& = E[ X^2] + E^2[X]  - 2E^2[X]   \\	
	& = E[ X^2] - E^2[X]   \\
\end{aligned}
```

:::

(VarianceAlgebra)=
### Variance Algebra

Some very useful properties of Variance:

If we multiply an RV by a constant $a$ , we multiply its expecation by the same constant **squared**.
$$\label{eq:var_multc} V(aX) = a^2 V(X)$$


The variance of a constant $b$ is zero.
$$\label{eq:var_const} V[b] = 0 $$
  
Adding a constant to an RV does not alter its variance.
$$\label{eq:var_addc} V(X + b) = V[X]$$
 
For the special case of Independent RVs only, the variance of a sum is the sum of the variances:

$$\label{eq:var_sumind} V(X + Y) = V(X) + V(Y)$$

[](#eq:var_sumind) **is only true if X and Y are mutually independent.** Compare with [](eq:expect_multind).




## Multivariate Data

In [](#intro-scatter) we made a scatter plot of height versus weight; the heights and weights together form a two dimensional dataset.


```{code-cell} python
import numpy as np
rng = np.random.default_rng()

heights = rng.normal(loc=165, scale=15, size=1000)

weights = rng.normal(loc=65, scale=10, size=1000)

data = np.stack((heights, weights)) # <= you need nested brackets

print(f" dataset shape: {data.shape }")

```

Note that our ```data``` is a **Matrix** (a two-dimensional array) with 2 rows and 1000 columns.



:::{dropdown} Notation - not pretty but you should probably read.

Common notation for multiple RVs is as follows:

$Z$ is the number of Random Variables under consideration, indexed by $j$.


The set of $Z$ **Random Variables** is denoted $X = \{X_{(1)}, X_{(2)}, ..., X_{(Z)}\}$
 

Note that the subscripts indicating the $j^{th}$ RV, $X_{(j)}$, are in brackets. This is because we use the notation $x_i$ (no brackets) to indicate a measurement in a dataset.

For [our example](#intro-scatter): 
* $Z=2$
* $X = \{ X_{(1)}, X_{(2)} \} = \{ \mathsf{Height, Weight} \}$.


The set of $Z$ **datasets** is denoted $x = \{x_{(1)}, x_{(2)}, ..., x_{(Z)}\}$

For [our example](#intro-scatter): 
* $x = \{ \texttt{heights, weights} \}$, where $\texttt{heights}$ and $\texttt{weights}$ are arrays of measurements.

The $j^{th}$ dataset of N **Measurements** is denoted $x_{(j)}= \{x_{j,1}, x_{j,2}, ..., x_{j,N} \}$


For [our example](#intro-scatter):
* $N=1k$ 
* $x_{(1)} = \texttt{heights} = \{h_1, h_2, ..., h_{1000}\}$.
* $x_{(2)} = \texttt{weights} = \{w_1, w_2, ..., w_{1000}\}$.
:::


### Covariance

We use the [variance](#eq:varianceU) (or standard deviation) to quantify how much spread there is in a 1D dataset. For a 2D dataset, we cannot express this with a single summary statistic.

If we have two RVs (eg weight and height) and they are **Independent**, we can calculate the two variances as usual. We can then pop these in a 2D array (a matrix):

```{math}
:label: eq:covmat-indep

\mathbf{\Sigma} = \begin{bmatrix}
\sigma^2_X & 0 \\
0 & \sigma^2_Y \\
\end{bmatrix}

```

If our RVs are **Dependent**, (if changing one of them affects the other, ie if y is a function of x) then we will also have **cross-terms** between their variances. These cross-terms are known as **Covariance** terms. 

```{math}
:label: eq:covmat-dep

\mathbf{\Sigma} = \begin{bmatrix}
\sigma^2_x & \sigma_{xy} \\
\sigma_{yx} & \sigma^2_y \\
\end{bmatrix}
\equiv
\begin{bmatrix}
\sigma^2_x & \mathsf{cov}(x,y) \\
\mathsf{cov}(y,x) & \sigma^2_y \\
\end{bmatrix}
```

The 2D arrays in [](#eq:covmat-indep) and [](#eq:covmat-dep) are **Covariance Matrices**. They are also sometimes called **Error Matrices**, because the (squared) Standard Deviations they hold are used as the Uncertainties or "Errors" on the measurements. 

:::{tip}
Covariance Matrices are always square (same number of rows and columns) and they are always symmetric (the pairs of off-diagonal terms are equal, so in our example case $cov(x,y) = cov(y,x)$).
:::

We can calculate covariances very easily using ```numpy```:

```{code-cell} python
import numpy as np

# make up some datasets for two RVs 

x = [0.88472455, 0.96232018, 0.10713343, 0.41198233, 0.06543451,0.40068931, 0.54846767, 0.46301972, 0.4534936 , 0.2064886 ]

y = [1.45373536, 0.48405628, 1.54298594, 0.21109356, 1.38629819, 1.00615679, 1.63028672, 0.34275748, 1.24847874, 1.03419514]

# put the two lists of fake data points in a 2D array

xy=np.stack((x,y))

# let numpy calculate the covariance matrix

cov_xy = np.cov(xy)

print(f"Cov matrix: {cov_xy}")

```

The datasets above are defined exactly (no random sampling) so we will all see the same output from the above snippet every time we run it.

Comparing this to our [mathematical definition](#eq:covmat-dep), we see that:

* $\sigma^2_x \equiv V[x] = 0.08777293$
* $\sigma^2_y \equiv V[y] = 0.26879134$
* $\sigma_{xy} \equiv \mathsf{cov}(x,y) = -0.03624434$

:::{important}
Check the variances on x and y with those you found from your hand-written functions/ numpy's built in method for variance.  **You will find they do not match the values in the covariance matrix**.

The mismatch between the values returned by the ```numpy``` methods ```cov``` and ```var``` is because the values in ```np.cov``` are by default the Unbiased Variances, with normalisation $\dfrac{1}{N-1}$ , while the values returned by ```np.var``` are by default the Biased Variances, with normalisation $\dfrac{1}{N}$.

These differences are very small with large datasets, but for our very small  datasets, the difference is substantial.

We can calculate the Unbiased Variance using ```np.var(x, ddof=1)``` and that will give us the same result as the ```np.cov``` default.

:::

Let's consider the off-diagonal covariance term $\sigma_{xy} \equiv \mathsf{cov}(x,y)$. 

The individual covariance of each data point is analogous to the [individual deviations](#eq:variancei) of each point in a 1D dataset:

$$
\label{eq:covtermi}
\mathsf{cov}(x_i,y_i) = (x_i - \overline{x}) (y_i - \overline{y})
$$

The covariance cov(x,y) is then the (unbiased) normalised sum over these, analagous to [V[x]](#eq:varianceU):

```{math}
:label: eq:covterm
\mathsf{cov}(x,y) =\mathsf{cov}(y,x) =  \dfrac{1}{N-1} \displaystyle\sum\limits_i \mathsf{cov}(x_i,y_i)
```


Let's make some very small (N=3) datasets to check this out by hand[^note]:

[^note]: I have tried to make this snippet clear, which means it is long. In my experience there is no such thing as over-commented code, and most code is horribly under-commented.

```{code-cell} python

# datasets:
a = np.array([1.1, 2.4, 3.3])
b = np.array([ 7.0, 5.0, 2.0])

# calculate the means:
mean_a = np.mean(a)
mean_b = np.mean(b)

# calculate the deviation of each point in a from the mean of a:

# individual steps for clarity:
# da_0 = mean_a - a[0] # 1.17
# da_1 = mean_a - a[1] # -0.13
# da_2 = mean_a - a[2] # -1.03

# one-liner, giving us da = [da_0 , da_1, da_2]
da =  [mean_a - ai for ai in a]

# same thing for b:

# db_0 = mean_b - b[0] # -2.33
# db_1 = mean_b - b[1] # -0.33
# db_2 = mean_b - b[2] # 2.67
db =  [mean_b - bi for bi in b]

# multiply together point-by-point:
# cov_0 = da_0 * db_0 # -2.72
# cov_1 = da_1 * db_1 #  0.04
# cov_2 = da_2 * db_2 # -2.76
cov =  [a*b for a,b in zip(da,db) ]


# sum and normalise
norm = 1 / ( len(a)-1 ) # 1/(N-1) unbiased version for small dataset
cov_ab =  norm * np.sum(cov)

print(f"cov_ab: {cov_ab}")

```

The snippet above returns the covariance values we will find in the off-diagonal of our Covariance Matrix:

```{code-cell} python

ab = np.stack((a,b))

print(f" np.cov(ab): {np.cov(ab)}")

```

Note that ```np.cov``` returns the whole covariance matrix rather than the covariance terms in the off-diagonal. We can access them like so:

```{code-cell} python

covmat_01 = np.cov(ab)[0][1]
print(f" cov(a,b): {covmat_01}")

covmat_10 = np.cov(ab)[1][0]
print(f" cov(b,a): {covmat_10}")

```

We can visualise the covariance terms with a [scatter plot of a versus b](#fig:mda-cov). The 3 shaded areas indicate the covariances of each of the three data points with respect to the mean of a (denoted $\langle a \rangle$, which is equivalent to $\overline{a}$) shown as a dotted red line and the mean of b, $\langle b \rangle$, shown as a dotted blue line.


:::{figure} 
:label: fig:mda-cov
:align: left
![](figures/mda_cov.png)
:::



### Linear Correlations

We can see from the above scatter plot that as a increases, b decreases. This could indicate that the variables a and b are somehow dependent on one another.


The covariance term $\mathsf{cov}(a,b)\approx  -2.72$ is negative, encoding the negative relationship between a and b. But the magnitude of the covariance term, $|\mathsf{cov}(a,b)|\approx  2.72$ is not at all helpful - the size of this value only tells us about the range of one or both of the datasets rather than how they "vary together".


To make the magnitude of the covariance term more useful, we can define the **Linear Correlation Coefficient**:

```{math}
:label: eq:rho
\rho(a,b) = \dfrac{ \mathsf{cov}(a,b) }{ \sigma_a \sigma_b }
```

For the datasets a and b defined in our snippet above, we have (from our covariance matrix, top left element $\Sigma_{00}$ ):

* $\sigma_a = \sqrt{1.223...} \approx 1.106...$ 
* $\sigma_b = \sqrt{6.333...} \approx 2.517...$
* $\rho(a,b) \approx \dfrac{-2.717...}{(1.106... )(2.517...)} \approx -0.976$

:::{important}
I have put ellipses (...) on the rounded numbers above to indicate that I am absolutely not rounding those numbers at any point before reaching the end of the calculation. Doing that would introduce horrible and unecessary inaccuracies. 
:::

We can access the Linear Correlation Coefficient $\rho(a,b)$ directly with ```numpy```:

```{code-cell} python

rho_ab = np.corrcoef(ab)

print(f"rho_ab: {rho_ab}")
```

The math notation for the **Linear Correlation Matrix** is:

```{math}
:label: eq:rhomat


\mathbf{\Rho} = 

\begin{bmatrix}
\dfrac{\sigma^2_a}{\sigma_a\sigma_a} & \dfrac{\mathsf{cov}(a,b)}{\sigma_a\sigma_b} \\
\dfrac{\mathsf{cov}(b,a)}{\sigma_b\sigma_a} &\dfrac{\sigma^2_b}{\sigma_b\sigma_b} \\
\end{bmatrix}

\equiv


\begin{bmatrix}
1 & \rho(a,b) \\
\rho(b,a) & 1 \\
\end{bmatrix}
```


### Limitations of the Linear Correlation Coefficient


The [Linear Correlation Matrix](#eq:rhomat) is the normalised covariance matrix, with diagonal elements equal to 1, and off-diagonals symmetric and equal to the linear correlation coefficients $\rho(a,b)$.

The correlations can have any values between -1 and 1, with extremes indicating:

* $\rho(i,j)=1 $ : Perfect Positive Linear Correlation between the two RVs.
* $\rho(i,j)=0 $ : Zero Correlation
* $\rho(i,j)=-1$ : Perfect Negative Linear Correlation between the two RVs.


A non-zero $\rho(i,j)$ indicates some linear correlation between the two datasets, but **the reverse is not true**. The linear correlation can be zero between datasets that are obviously related, as illustrated in [](#fig:rho-wiki).

:::{figure} 
:label: fig:rho-wiki
:align: left
![](figures/rho_wiki.png)

By DenisBoigelot, CC0, from [wikipedia](https://commons.wikimedia.org/wiki/File:Correlation_examples2.svg).
:::


All of the x,y distributions on the bottom row have a zero linear correlation coefficient, despite being very obviously related. If x,y were independent, we would expect something more like the middle section of the top row. The bottom row of distributions have $\rho(x,y) =0$ because **the correlations between x and y are not linear**. Hopefully this makes it clear how limited the linear correlation coefficient is!


## Learning Objectives Checklist

- [ ] Explain what is meant by the Expectation and how it relates to the Mean
- [ ] Describe in words the statement of the Law of Large Numbers (LLN)
- [ ] Calculate the expectation for discrete and continuous probability distributions.
- [ ] Express the True Variance in terms of the Expectation
- [ ] Understand the meaning of the terms in the Covariance Matrix
- [ ] Calculate covariance with numpy
- [ ] Be aware of the different results returned by numpy's cov and var, and know how to harmonise
- [ ] Calculate the linear correlations between two datasets with numpy
- [ ] Understand that an absence of linear correlation does not imply independence 
