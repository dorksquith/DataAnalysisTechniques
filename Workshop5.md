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

(workshop-5)=
# Workshop 5: Normal Tests

This workshop uses the Three Cities dataset introduced in [Normal Tests](#chapter:normtest). Run the `three_cities_data` function from that notebook, reproduced [here](#gen-3cities) for convenience.

---

## Task 1: True or False

1. The Standard Error on the Mean (SEM) becomes smaller as the number of measurements $N$ increases.

2. For a dataset of size $N=30$, the T distribution and the Normal distribution give indistinguishable p values on a linear scale.

3. The Z test requires knowledge of the true variance $\sigma^2$.

4. A p value of 3% means there is a 3% probability the Null Hypothesis is true.

5. The two-sample T test requires both samples to be the same size.

6. The "Variance Within" in the F test is called the "unexplained" variance because it represents scatter around each group's own mean — scatter that exists even within a single group, and so cannot be explained by any difference between the groups.

7. For an F test comparing two groups each of size $N=15$, the degrees of freedom within is $\nu_{within} = 28$.

8. A one-sample T test and a Z test can both be used when the true variance $\sigma^2$ is unknown.

---

::::{dropdown} Show Code for generating 3 Cities Datasets

(gen-3cities)=
```{code-cell}python
import numpy as np
np.random.default_rng()
from scipy.stats import norm

def three_cities_data( reproducible=True ):

    if reproducible:
        np.random.seed(42)

    #--------------------------------------------------------------------------------------------#
    # Parameters for Normal distribution in theoretical universe with infinite data points
    #--------------------------------------------------------------------------------------------#
    
    mu    = 7.33
    sigma = 1.21
 
    dist = norm(loc=mu, scale=sigma)

    # Number of fake people asked
    sample_sizes = [2, 3, 10, 30, 100, 3_000]

    iterations = range( len(sample_sizes) )

    # Difference between subsequent sample sizes eg 3-2=1, 10-3=7
    size_inc     = [ j-i for i, j in zip( sample_sizes[:-1], sample_sizes[1:] ) ]
    
    # want first number to be 2 for our N=2 sample
    size_inc.insert(0, 2)

    #-----------------------#
    # Fake data for Brixton
    #-----------------------#
    B_data1 = dist.rvs(size = max(10_000, max(sample_sizes) ) )  
    B_data = []

    #-----------------------#
    # Fake data for Stockport
    #-----------------------#
    S_data1 = dist.rvs(size = max(10_000, max(sample_sizes) ) ) 
    S_data = []

    #-----------------------#
    # Fake data for Glasgow
    #-----------------------# 
    G_data1 = dist.rvs(size = max(10_000, max(sample_sizes) ) )
    G_data = []

    #-----------------------#
    # Loop over sample sizes
    #-----------------------#   
    for i in iterations:
       
        N = sample_sizes[i]
        Nplus = size_inc[i]

        # randomly draw N **additional** fake measurements

        for j in range(Nplus):
        
            B_data.append(np.random.choice(B_data1,  replace=False) )
            S_data.append(np.random.choice(S_data1,  replace=False) )
            G_data.append(np.random.choice(G_data1,  replace=False) )

    return mu, sigma, B_data, S_data, G_data

reproducible = True

true_mu, true_sigma, B_data, S_data, G_data = three_cities_data(reproducible)
```

::::


## Task 2: One-sample T Test

Using the Three Cities dataset (see the bottom of this worksheet), take a sample of $N=30$ measurements from Glasgow.

a) State a sensible Null Hypothesis and choose an appropriate significance level and alternative hypothesis.

b) Look at your data: calculate the sample mean, standard deviation, and Estimated SEM.

c) Check that using the T test is appropriate here (rather than the Z test).

d) Calculate the T test statistic and the p value. Do you reject the Null Hypothesis?

e) Repeat for $N=3$. Do your conclusions change? Comment on why the sample size matters here.

---

## Task 3: Two-sample T Test

Using the Three Cities dataset, take samples of $N=30$ from Stockport and from Glasgow.

a) State the Null Hypothesis.

b) Check that the equal-variance condition for the two-sample T test is satisfied.

c) Calculate the pooled variance $\sigma^2_p$, the multiplier $\psi$, the T test statistic $T_2$, and the p value. Is the result positive or negative?

d) Verify your result using `scipy.stats.ttest_ind`.

---

## Task 4: F-Test

a) Using `scipy.stats.f_oneway`, perform an F test comparing all three cities (Brixton, Stockport, and Glasgow) simultaneously, using $N=30$ for each. Report the F statistic and p value.

b) Manually compute the F statistic for this three-city comparison using [](#eq:varbetween) and [](#eq:varwithin), and verify that your result matches `f_oneway`.

c) For two groups, the F statistic satisfies $F = T_2^2$. Verify this numerically using the Stockport vs Glasgow datasets from Task 3.

---

