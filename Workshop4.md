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

## Task 1: True or False

1. Finding a p value of 7% when I have set a Significance Level of $\alpha=$5% means I have confirmed the Null Hypothesis.

2.  Finding a p value of 1% when I have set a Significance Level of $\alpha=$2% means I have rejected the Null Hypothesis.

3.  Finding a p value of 1% means that there is a 1% probability my conclusions are wrong

4. The False Positive rate is the probability of incorrectly rejecting the Null Hypothesis, given that it is true.

5. The sum of the False Positive and True Positive rates must equal 1. 

6. The sum of the False Positive and True Negative rates must equal 1. 

7. To test the Null Hypothesis, I need to specify the test by choosing a Significance Level and a Critical Region.

8. It is okay to look at my data before designing the test if I am an honest person, as I won't be prone to bias.


## Task 2: CDF, Z value, P Value

Make a `scipy.stats.norm` distribution with $\mu=1$, $\sigma=2$. This is our null hypothesis.

a) Plot the CDF for 100 values of X from $\mu - 5\sigma < x < \mu + 5\sigma$

b) Find the critical values of X for which we can define a two-sided critical region with a significance level of $\alpha=5\%$

c) What would be the p value corresponding to a measurement of $x=-4.5$?

d) What is the Z value for this measurement, in sigmas?

e) What is the probability that our measurement will fall within 5 standard deviations of the mean?

f) How would you interpret finding a measurement outside the $5\sigma$ region?


---

## Task 3: Optional Extension Task (if this is not your first rodeo)

The module notes have a python snippet [MultiVariateGaus](#seaborn-gaus2) for plotting a bivariate Gaussian distribution with two choices of covariance matrix.

a) Can you choose any values for the covariance matrix elements? What are the limitations?

b) The contours in the example I provided are automatically selected in `seaborn`'s `plot_joint`, by the `kdeplot` method. See if you can set the contours according to the 1,2,3,.. sigma probabilities. 

c) Can you make a 3D plot of the bivariate Gaussian, with the z-axis being the PDF value?


---

## Example Solutions

To follow after the workshop.
