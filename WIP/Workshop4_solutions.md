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

# Workshop 4: Z values, P values, Oh my!

<!-- from the MDA module I taught last year, as a warm up-->
## Task 1: True or False

1. Finding a p value of 7% when I have set a Significance Level of 5% means I have confirmed the Null Hypothesis.

2.  Finding a p value of 1% when I have set a Significance Level of 2% means I have rejected the Null Hypothesis.

3.  Finding a p value of 1% means that there is a 1% probability my conclusions are wrong

4. The False Positive rate is the probability of incorrectly rejecting the Null Hypothesis, given that it is true.

5. The sum of the False Positive and True Positive rates must equal 1 

6. The sum of the False Positive and True Negative rates must equal 1 

7. To test the Null Hypothesis, I need to specify the test by choosing a Significance Level and a Critical Region.

8. It is okay to look at my data before designing the test if I am an honest person, as I won't be prone to bias.

### Answers

1. False. One cannot confirm a hypothesis.

2. True. If $p < \alpha$, we reject.

3. False. 

4. True.

5. False. The FPR and TPR belong in different universes.

6. True. 

7. True.

8. False.



<!-- from the MDA module I taught last year, as a warm up-->
## Task 2: CDF, Z value, P Value

Make a `scipy.stats.norm` distribution with $\mu=1$, $\sigma=2$. This is our null hypothesis.

1. Plot the CDF for 100 values of X from $-5\sigma < x < 5\sigma$

2. Find the critical values of x for which we can define a two-sided critical region with a significance level of $\alpha=5\%$

3. What would be the p value corresponding to a measurement of $x=-4.5$?

4. What is the Z value for this measurement, in sigmas?

5. What is the probability that our measurement will fall within 5 standard deviations of the mean?

6. How would you interpret finding a measurement outside the 5 sigma region?

### Example Solution

First make the distribution:

```{code-cell} python
from scipy.stats import norm

mu=1
sigma=2

dist = norm(loc=mu, scale=sigma)

```

Next, plot the CDF (I have chosen to also plot the PDF as I find it helpful):

```{code-cell} python
import numpy as np

# 100 values of x in mu +/- 5sigma
x = np.linspace( mu - 5*sigma, mu + 5*sigma, 100 )

pdf_dist = dist.pdf(x)
cdf_dist = dist.cdf(x)

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(1, 1)

ax1.plot(x,pdf_dist, color='lightsteelblue', linewidth=1)

color = 'tab:blue'
ax1.set_ylabel('PDF', color=color, fontsize=16)
ax1.set_ylim(bottom=0)
ax1.tick_params(axis='y', labelcolor=color)

# make a twinned y-axis on which we can plot the CDF
# (we are sharing the x-axis)

ax2 = ax1.twinx()  

ax2.plot(x, cdf_dist, 'r--', linewidth=2)

color = 'tab:red'
ax2.set_ylabel('CDF', color=color, fontsize=16)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(bottom=0)


plt.show()
```

Calculate the critical region for $\alpha=$ 1%:

```{code-cell} python

alpha = 0.05

# the distribution is two-sided, so the critical region is, too.
# 5% => 2.5% on each side

alpha1 = alpha/2

# cumulative probability from -infty to upper bound: 

cdf_rhs = 1 - alpha1 

print( f"P(X <= x+) = {cdf_rhs}")

# RV value at the upper bound 
# dist.ppf(0.99) returns the RV value corresponding to a cumulative 99% of the distribution

x_rhs = dist.ppf(cdf_rhs) 

# cumulative probability from -infty to lower bound:

cdf_lhs = alpha1 

print( f"P(X <= x-) = {cdf_lhs}")

# RV value at the lower bound

x_lhs = dist.ppf(cdf_lhs) 

print(f"Critical region for a significance level alpha={alpha} is {x_lhs:.3f} < X < {x_rhs:.3f}")

```

Calculate the p value and z value for $x=-4.5$:

```{code-cell} python

# a measurement value.

x = -4.5

# probability P( X <= -4.5) 

cdf = dist.cdf(x)

# In this case, the p value is the CDF, because we are on the left tail of the distribution

prob_more_x = dist.sf(x)

print(f" {100*prob_more_x:.3f}% of distribution is to the right of x={x}")

prob_less_x = 1 - dist.sf(x)
print(f" {100*prob_less_x:.3f}% of distribution is to the left of x={x}")

pvalue = prob_less_x
print(f" p value = {pvalue:.3f} ({100*pvalue:.3f} %)")

# z value in number of sigmas
zvalue = (x - mu) / sigma

# state the z value as a magnitude
zvalue = abs(zvalue)

print(f" z value = {zvalue} sigma")
```


Calculate the probability of a $5\sigma$ event:


```{code-cell} python
# probability of a measurement being within 5 sigmas

# rv value at + 5sigma and -5sigma
x_up = mu + 5 * sigma
x_down = mu - 5 * sigma

# right-sided p-value = P(X > x_up)
pval_rhs = dist.sf(x_up)

#print(f" right-sided p value, P(X > x_up), = {pval_rhs}")

# left-sided p-value = P(X < x_down)
pval_lhs = 1-dist.sf(x_down)

#print(f" left-sided p value, P(X < x_down), = {pval_lhs}")

# notice the right-sided and left-sided p values are identical, because the normal distribution is symmetric about its mean

p5 = 2 * pval_rhs

print(f"Two-sided p value for a z value of 5 sigma = {p5:.3e}")

Nmeas_for_5sigma = int( 1/p5 )
print(f"A measurement outside the 5sigma region is expected to occur once in {Nmeas_for_5sigma:,} repeated experiments.")

```


---

## Task 3: Optional Extension Task (if this is not your first rodeo)

The module notes have a python snippet [#multivariate-gaussian](https://dorksquith.github.io/DataAnalysisTechniques/normal/#multivariate-gaussian) for plotting a bivariate Gaussian distribution with two choices of covariance matrix.

a) Can you choose any values for the covariance matrix elements? What are the limitations?

b) The contours in the example I provided are automatically selected in `seaborn`'s `plot_joint`, by the `kdeplot` method. See if you can set the contours according to the 1,2,3,.. sigma probabilities. 

c) Can you make a 3D plot of the bivariate Gaussian, with the z-axis being the PDF value?

### Example Solution

a) A covariance matrix must be square and symmetric.

b) An example of setting the contours at N sigma, by setting the `levels` keyword agrument in `plot_joint`.

```{code-cell}python
def Gaussian_SigmaContours():
    from scipy.stats import multivariate_normal, norm
    import seaborn as sns
    import pandas as pd

    # set the means and the covariance matrix
    mu = [0, 0]
    cov = [[10,5],[5,10]]
    sigma = np.sqrt(cov[0][0])

    # Define the sigma levels (inverted order, because seaborn wants it)
    sigmas = [3, 2, 1]

    # Calculate the excluded percentage for each sigma
    levels=[]
    for s in sigmas:

        # Probability within ±s sigma
        p_within = norm.cdf(s) - norm.cdf(-s)
    
        print(f"{s}σ contains {p_within*100:.2f}% of the sample")

        excluded = (1 - p_within)
        levels.append(excluded)
    

    XYdist = multivariate_normal(mu, cov)

	# N rows, Z columns: need transpose for JointGrid- dataframe fixes this
    rvs= XYdist.rvs(size=100_000) 

    df = pd.DataFrame(rvs, columns=["X1", "X2"])

    g = sns.JointGrid(data=df, x="X1", y="X2")

    g.plot_marginals(sns.histplot,fill=True)

    sns.kdeplot(
        data=df, 
        x="X1", 
        y="X2", 
        levels=levels, 
        ax=g.ax_joint, 
        cmap="coolwarm"
    )

    g.ax_marg_x.set_xlim(-15,15)
    g.ax_marg_y.set_ylim(-15,15)
    g.ax_joint.text(-14, 14, r"$\mu$:{}, $\Sigma$:{}".format(mu,cov), fontsize=16)

    g.ax_joint.set_xlabel(r'X$_{(1)}$',fontsize=16)
    g.ax_joint.set_ylabel(r'X$_{(2)}$',fontsize=16)
    
    plt.show()

Gaussian_SigmaContours()

```

c) An example of a 3D plot using `plot_surface`

```{code-cell} python
def ThreeD_biv_gaus():
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    mu = np.array([0.0, 0.0])
    cov = np.array([[10.0, 0.0], [ 0.0,10.0]])

    # Build a grid (adjust limits/resolution as needed)
    x = np.linspace(-10, 10, 300)
    y = np.linspace(-10, 10, 300)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Analytic PDF via SciPy
    rv = multivariate_normal(mean=mu, cov=cov)
    Z = rv.pdf(pos)

    # 3D surface plot
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                       linewidth=0, antialiased=True, alpha=0.95)

    # Labels and view
    ax.set_xlabel(r'X$_1$'); ax.set_ylabel(r'X$_2$');
    ax.set_zlim(0, Z.max()*1.1)
    ax.view_init(elev=30, azim=-135)
    fig.colorbar(surf, shrink=0.6, pad=0.1, label='pdf')
    plt.tight_layout()
    plt.show()

ThreeD_biv_gaus()
```

---

