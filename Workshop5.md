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

This workshop uses the Three Cities dataset introduced in [Normal Tests](#chapter:normtest). Run the `three_cities_data` function from that notebook before starting the tasks here.

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

## Task 2: One-sample T Test

Using the Three Cities dataset, take a sample of $N=30$ measurements from Glasgow.

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

## Optional Extension Task (if this is not your first rodeo)

a) Using `scipy.stats.f_oneway`, perform an F test comparing all three cities (Brixton, Stockport, and Glasgow) simultaneously, using $N=30$ for each. Report the F statistic and p value.

b) Manually compute the F statistic for this three-city comparison using [](#eq:varbetween) and [](#eq:varwithin), and verify that your result matches `f_oneway`.

c) For two groups, the F statistic satisfies $F = T_2^2$. Verify this numerically using the Stockport vs Glasgow datasets from Task 3.

---

