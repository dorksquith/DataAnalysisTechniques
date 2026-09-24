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

# Data Analysis Techniques ![ML is stats](/figures/ml-is-stats.png)


This module explores how to use probability and statistics to analyse data and is currently organised in two volumes.


## Volume 1

[//]: # ( Terminology & Concepts)
[//]: # ( Relies on: N/A)
[//]: # ( Notes: Week2-RandomVars/Data.tex random numbers:Week7-Tools/Tools.tex  )
[//]: # ( numpy: sum,min,max,mean,var,std,random )
[//]: # ( scipy.stats: uniform, norm )
[//]: # ( matplotlib.pyplot: plot, hist, scatter )
[//]: # ( openstax: https://openstax.org/books/introductory-statistics-2e/pages/1-practice )
(los:T&C)=
:::{dropdown} Terminology & Concepts
- [ ] Define hypothesis, model, theory
- [ ] Understand the distinction between a population and a sample
- [ ] Give examples of a random variable, a parameter, and a summary statistic
- [ ] Describe the difference between Discrete and Continuous RVs, and give examples of each
- [ ] Generate uniformly distributed and normally distributed pseudorandom numbers
- [ ] Calculate the mean, variance, and standard deviation of a dataset
- [ ] Make simple ```plot```,  ```hist``` and ```scatter``` in matplotlib
:::

[//]: # ( Probability Essentials)
[//]: # ( Relies on: N/A)
[//]: # ( Notes: Week1-Probability/Probability.tex )
[//]: # ( python: set, range, issubset)
[//]: # ( scipy.stats: randint, norm )
[//]: # ( matplotlib.pyplot: stem )
[//]: # ( openstax: https://openstax.org/books/introductory-statistics-2e/pages/3-practice)
(los:Prob)=
:::{dropdown} Probability Essentials
- [ ] Summarise the Frequentist and Bayesian interpretations of probability
- [ ] Understand and express relationships between sets ($\subset$, $\cap$, $\cup$, $A'$)
- [ ] Correctly interpret Venn diagrams
- [ ] Calculate the Union and Intersection of sets
- [ ] Describe the meaning of Independent and Mutually Exclusive (Disjoint) sets
- [ ] Calculating the Conditional Probability from a Contingency Table
- [ ] Describe the terms in Bayes' Theorem
:::



[//]: # ( Describing Data)
[//]: # ( Relies on: )
[//]: # ( Notes: https://canvas.sussex.ac.uk/courses/37537/pages/11-describing-data)
[//]: # ( numpy: random, stack, shape, cov, corrcoef)
[//]: # ( scipy.stats: randint, norm )
[//]: # ( matplotlib.pyplot: stem )
[//]: # ( openstax: 2.5 and 2.7 )

(los:Data)=
:::{dropdown} Describing Data
- [ ] Explain what is meant by the Expectation and how it relates to the Mean
- [ ] Describe in words the statement of the Law of Large Numbers (LLN)
- [ ] Calculate the Expectation for discrete and continuous probability distributions.
- [ ] State the Law of Large Numbers (LLN)
- [ ] Express the True Variance in terms of the Expectation
- [ ] Understand the terms in the Covariance Matrix
- [ ] Calculate covariance with numpy
- [ ] Be aware of the different results returned by numpy's cov and var, and how to harmonise
- [ ] Calculate the linear correlations between two datasets with numpy
- [ ] Understand that an absence of linear correlations does not imply independence 


[//]: # ( The Normal Distribution)
[//]: # ( Relies on: )
[//]: # ( Notes: )
[//]: # ( numpy: linspace)
[//]: # ( scipy.stats: pdf, cdf, multivariate_normal)
[//]: # ( matplotlib.pyplot: semilogy)
[//]: # ( pandas: DataFrame)
[//]: # ( seaborn: JointGrid.plot_joint, kdeplot, JointGrid.plot_marginals, histplot)
[//]: # ( openstax: )
(los:Norm)=
:::{dropdown} The Normal Distribution
- [ ] Explain why the normal distribution is so prevalent
- [ ] Explain why the CDF, rather than PDF, must be used for calculating probabilities for continous RVs
- [ ] Plot the Normal PDF and CDF 
- [ ] State the formula for calculating the Z value, and calculate Z values
- [ ] Describe the terms present in the Gaussian PDF
- [ ] Describe the location and scale parameters, and demonstrate the effect of changing them
- [ ] State the Central Limit Theorem (CLT)
:::


[//]: # ( Hypothesis Tests)
[//]: # ( Relies on: )
[//]: # ( Notes: )
[//]: # ( numpy: )
[//]: # ( scipy.stats: sf, ppf)
[//]: # ( matplotlib.pyplot:)
[//]: # ( openstax: )
(los:Hypo)=
:::{dropdown} Hypothesis Tests
- [ ] Understand the impact of allowing bias to pollute science 
- [ ] State the criteria for constructing a hypothesis test
- [ ] Explain what is meant by Critical Regions and Significance Levels
- [ ] Explain the elements in a Confusion Matrix (True/False Positive/Negative)
- [ ] Distinguish between Type 1 and Type 2 Errors
- [ ] Know what can and cannot be inferred from a p value
- [ ] Calculate the p value for a given Test and dataset
- [ ] Calculate p values from Z values and vice versa
- [ ] Understand the meanings of the terms Confidence Level, Significance, Purity, and Efficiency. 
- [ ] Interpret a ROC curve
:::


[//]: # ( Normal Tests)
[//]: # ( Relies on: )
[//]: # ( Notes: )
[//]: # ( numpy: )
[//]: # ( scipy.stats: sf, ppf)
[//]: # ( matplotlib.pyplot:)
[//]: # ( openstax: )
(los:NormTests)=
:::{dropdown} Normal Tests
- [ ] State what it means for data to be Independent and Identically Distributed (IID)
- [ ] Define the Standard Error on the Mean (SEM)
- [ ] Calculate the SEM 
- [ ] Understand the limitations introduced by small datasets
- [ ] Design a Z Test to compare the mean of a dataset with the null hypothesis, and calculate the test statistic
- [ ] Describe the T Test and state when it is preferred over the Z Test
- [ ] Design a "Student's" T Test to compare the means of two datasets, and calculate the test statistic
- [ ] Design a F Test (AnoVa) to compare the variances of two datasets, and calculate the test statistic
:::


## Volume 2: Parameter Estimation & Fitting

[//]: # ( Estimation Essentials)
[//]: # ( Relies on: )
[//]: # ( Notes: bootstrap: Week11-MCMC/MCMC-Part1.tex, rest:Week7-Tools/Tools.tex )
[//]: # ( numpy: )
[//]: # ( scipy.stats: )
[//]: # ( matplotlib.pyplot:)
[//]: # ( openstax: )
(los:Estimation)=
:::{dropdown} Estimation Essentials
- Point and Interval Estimation
- The Bootstrap
- Consistency, Bias, and Minimum Variance
- Method of Moments
:::


[//]: # ( Discrete Probabilities & Likelihoods)
[//]: # ( Relies on: )
[//]: # ( Notes: Bayes revisited Week10-Bayesian/BayesianStats.tex, Likelihood:Week7-Tools/Tools.tex pmfs: Week4-PDFs/SpecialPDFs.tex Week8-Estimators/MaxLike.tex )
[//]: # ( numpy: )
[//]: # ( scipy.stats: )
[//]: # ( matplotlib.pyplot:)
[//]: # ( openstax: )
(los:PMFLikelihood)=
:::{dropdown} Discrete Probabilities & Likelihoods
- Bayes theorem revisited: for discrete RVs
- Bernoulli: "the single coin toss"
- Binomial: "repeated coin tosses"
- Poisson: "number of emails per hour"

- [ ] Explain what is meant by a Bernoulli Trial
- [ ] Plot the Bernoulli PMF for some data
- [ ] Show that the Bernoulli PMF is a valid probability distribution
- [ ] Calculate Bernoulli probabilities using the PMF
- [ ] Show that the Expectation and Variance of K~Bernoulli(p) are both equal to p 
- [ ] Explain the relationship between the Bernoulli and Binomial distributions and parameters
- [ ] Calculate the Binomial Coefficient
- [ ] Calculate the Expectation and Variance for K~Binomial(n,p)
- [ ] Estimate the fairness of a coin using the Binomial Likelihood
- [ ] Describe the properties of k~Poisson(\lambda)
- [ ] Calculate and use the Poisson Likelihood

:::

[//]: # ( Special Probability Distributions)
[//]: # ( Relies on: )
[//]: # ( Notes: Bayes revisited Week10-Bayesian/BayesianStats.tex, Likelihood:Week7-Tools/Tools.tex pmfs: Week4-PDFs/SpecialPDFs.tex )
[//]: # ( numpy: )
[//]: # ( scipy.stats: )
[//]: # ( matplotlib.pyplot:)
[//]: # ( openstax: )
(los:PDFLikelihood)=
:::{dropdown} Special Probability Distributions
- Bayes theorem revisited: for continuous RVs
- Uniform: "flat"
- Exponential: "waiting time for a taxi"
- Gamma: "waiting time for nth taxi"
- Beta: "the chameleon"

- [ ] Show that the uniform PDF is valid
- [ ] Calculate the Expectation for X~Uniform(a,b)
- [ ] Write down the PDF for X~Expon($\lambda$)
- [ ] Compare X~Expon($\lambda$) with K~Poisson($\lambda$), explaining why one is continuous and the other discrete
- [ ] Explain the Memoryless property of Expon($\lambda$)
- [ ] Know when to use a Beta PDF
- [ ] Describe the Beta PDF parameters
:::


[//]: # ( Maximum Likelihood Method)
[//]: # ( Relies on: Log rules, partial derivatives, expectation algebra, Taylor series for graphical, bias, LLN, chain and product rules, LOTUS )
[//]: # ( Notes: Week8-Estimators/MaxLike.tex )
[//]: # ( numpy: )
[//]: # ( scipy.stats: )
[//]: # ( matplotlib.pyplot:)
[//]: # ( openstax: )
(los:MLM)=
:::{dropdown} Maximum Likelihood Method
- The MLE for Norm(mu,sigma)
- Estimator Bias and Mean Squared Error
- Parameter Uncertainties
-- Monte Carlo
-- Fisher Information and Minimum Variance Bound
-- Graphical Method

[ ] Describe what is meant by the bias of an estimator
[ ] Calculate the MSE
[ ] Understand that sometimes, the best estimator for the job is the biased estimator
[ ] Understand what the Hessian (curvature) matrix represents
:::


[//]: # ( Linear Regression)
[//]: # ( Relies on: Log Like Normal, Poisson, refers to MLE, Cramers rule, Taylor expansion, Hessian)
[//]: # ( Notes: Week9-Fits-Tests/LeastSquares.tex)
[//]: # ( numpy: )
[//]: # ( scipy.stats: )
[//]: # ( matplotlib.pyplot:)
[//]: # ( openstax: )
(los:MLM)=
:::{dropdown} Linear Regression
- Residuals and Pearson's Chi Squared
- Chi Squared with binned data
- The method of Least Squares (LSQ)
- Parameter uncertainties with LSQ
- Python methods
- Outliers
- Nonlinear Methods
:::

[//]: # ( Model Tests)
[//]: # ( Relies on: Least Squares, normal tests, degrees of freedom, variance and expectation)
[//]: # ( Notes: Week9-Fits-Tests/TestStatistics.tex s18-s26 Chi Squared and the Standard Normal RV)
[//]: # ( numpy: )
[//]: # ( scipy.stats: )
[//]: # ( matplotlib.pyplot:)
[//]: # ( openstax: )
(los:ModelTests)=
:::{dropdown} Model Tests
- Chi Squared
- Likelihood Ratio (LLR)
- Akaike and Bayes Information Criteria (AIC and BIC)
:::

[//]: # ( Bayesian Inference)
[//]: # ( Relies on: )
[//]: # ( Notes: Week10-Bayesian/BayesianStats.tex)
[//]: # ( numpy: )
[//]: # ( scipy.stats: )
[//]: # ( matplotlib.pyplot:)
[//]: # ( openstax: )
(los:Bayes)=
:::{dropdown} Bayesian Inference
- Bayes theorem revisited: evidence, prior, likelihood, posterior
- Bernoulli Likelihood example: covid tests 
- Constant Evidence
- Prior Distribution
- Conjugacy
- Bernoulli-Beta
- Maximum A Posteriori (MAP) Estimation
- Credible Intervals
- Poisson-Gamma
- Predictive Posterior
:::

[//]: # ( Predictions)
[//]: # ( Relies on: )
[//]: # ( Notes: Week11-MCMC/MCMC-Part1.tex)
[//]: # ( numpy: )
[//]: # ( scipy.stats: )
[//]: # ( matplotlib.pyplot:)
[//]: # ( openstax: )
(los:Predictions)=
:::{dropdown} Predictions
- Markov Chain Monte Carlo (MCMC)
- Metropolis Hastings
- Hyperparamer Tuning
:::

<!--## What issues might we face?

1. Bad information: on the topics of statistics and probability, there is an especially high amount of trash on the internet. I think this is because truly understanding the concepts behind the tools requires a lot of study and re-study, and the topics needed are quite long and boring with no clear pathways and definitely no quick rewards.
2. Unclear information: this is partly a notation issue - if you don't have a degree in maths the notation is misery-inducing - but I have also found that experts and educators are unneccessarily vague on the particularly hard and/or boring topics. I think this is because they are scared of saying something incorrect as per point 1.
3. Inaccessible information: the assumption of prior knowledge and/or the use of terminology, notation that is unfamiliar to the students.-->



