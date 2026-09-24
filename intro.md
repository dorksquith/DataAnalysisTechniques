---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: 'Python 3'
numbering: false

---

# Data Analysis Techniques 890F3 ![ML is stats](/figures/ml-is-stats.png)


This module explores how to use probability and statistics to analyse data and is currently organised in two volumes.


## Volume 1: Probability & Statistics

The materials for the first volume are largely ready, with the topic pages available via the menu to the ←left of this page. 

On each page you will see a "power button" ⏻ below the page heading on the right ↗; press this to connect the jupyter server. Once connected, a "play button" ▷ will appear - press this to run the python snippets embedded in the corresponding page.

I am in the process of converting and uploading exercises for all the topics. For Terminology & Concepts and Probability Essentials, there are some good exercises in the OpenStax book which I have linked from those pages.

The dropdowns below list the **Learning Objectives** for each topic.

[//]: # ( Terminology & Concepts)
[//]: # ( Relies on: N/A)
[//]: # ( Notes: Week2-RandomVars/Data.tex random numbers:Week7-Tools/Tools.tex  )
[//]: # ( numpy: sum,min,max,mean,var,std,random )
[//]: # ( scipy.stats: uniform, norm )
[//]: # ( matplotlib.pyplot: plot, hist, scatter )
[//]: # ( openstax: https://openstax.org/books/introductory-statistics-2e/pages/1-practice )
(los:Term)=
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
- [ ] Express the True Variance in terms of the Expectation
- [ ] Understand the terms in the Covariance Matrix
- [ ] Calculate covariance with numpy
- [ ] Be aware of the different results returned by numpy's cov and var, and how to harmonise
- [ ] Calculate the linear correlations between two datasets with numpy
- [ ] Understand that an absence of linear correlations does not imply independence 
:::


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
- [ ] Design a F Test (ANOVA) to compare the variances of two datasets, and calculate the test statistic
:::



## Volume 2: Parameter Estimation & Fitting (```wip```)

This volume forms the second part of the module, and the materials are ```wip``` (work in progress). I have put the Outline/ Learning Objectives here so you can get an idea of what will we cover.

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

[ ] Explain what is meant by a Bernoulli Trial
[ ] Plot the Bernoulli PMF for some data
[ ] Show that the Bernoulli PMF is a valid probability distribution
[ ] Calculate Bernoulli probabilities using the PMF
[ ] Show that the Expectation and Variance of $K\sim \mathsf{Bernoulli}(p)$ are both equal to p 
[ ] Explain the relationship between the Bernoulli and Binomial distributions and parameters
[ ] Calculate the Binomial Coefficient
[ ] Calculate the Expectation and Variance for $K\sim \mathsf{Binomial}(n,p)$
[ ] Estimate the fairness of a coin using the Binomial Likelihood
[ ] Describe the properties of $k\sim \mathsf{Poisson}(\lambda)$
[ ] Calculate and use the Poisson Likelihood

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
- Gamma: "waiting time for $n^{th}$ taxi"
- Beta: "the chameleon"

[ ] Show that the Uniform PDF is valid
[ ] Calculate the Expectation for $X\sim \mathsf{Uniform}(a,b)$
[ ] Write down the PDF for $X\sim \mathsf{Expon}(\lambda)$
[ ] Compare $X\sim \mathsf{Expon}(\lambda)$ with $K\sim Poisson(\lambda)$, explaining why one is continuous and the other discrete
[ ] Explain the Memoryless property of $\mathsf{Expon}(\lambda)$
[ ] Know when to use a Beta PDF
[ ] Describe the Beta PDF parameters
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
- Log Likelihood Ratio (LLR)
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


## Resources


:::{tip} Avoiding Overwhelm
If you go through all the links in the dropdown below, you will find yourself in a situation of lots of open browser tabs and no more insight into data analysis techniques than you had before you looked at this page.

Make yourself comfortable, and follow the suggested module structure in these notes. These resources are not going anywhere, and the notes will refer to them at the appropriate moments.

:::

### Bookmarks

I recommend Bookmarking these excellent resources:
-[Stat Proofs](https://statproofbook.github.io/): Hallulajah! Almost all the proofs you will ever need.
-[Probability Playground](https://probabilityplayground.com/normal.html): Essential for visual learners, brilliant way to enhance understanding of PDFs.

### Computing and Math Essentials

[**Essential: Github Quickstart**](https://docs.github.com/en/get-started/git-basics/set-up-git)
: If you don't already have a github, please set one up. Set up a repository for this module. You will not regret getting on top of this.

[**Scientific Python**](https://lectures.scientific-python.org/)
: Superb long-term reference. I expect it will help you in this module and beyond.

:::{dropdown} Other useful computing references

- [Python Book](https://www.acsu.buffalo.edu/~adamcunn/downloads/PythonBook.pdf)

- [Jupyter](https://jupyter.org/)

- [numpy](https://numpy.org/devdocs/user/absolute_beginners.html)

- [scipy stats](https://docs.scipy.org/doc/scipy/tutorial/stats.html#)

- [matplotlib](https://matplotlib.org/stable/users/explain/quick_start.html)

- [seaborn](https://seaborn.pydata.org/tutorial/introduction.html)

- [pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
:::


- [**Math**](https://www.lboro.ac.uk/departments/mlsc/student-resources/helm-workbooks/): If your math is rusty, I recommend looking at the relevant workbooks here.

:::{dropdown} The HELM workbooks I refer to in these notes

 - [HELM 6](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%206%20Exponential%20and%20Logarithmic%20Functions.pdf): Exponential and Logarithmic Functions
 - [HELM 7](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%207%20Matrices.pdf): Matrices
 - [HELM 11](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2011%20Differentiation.pdf): Differentiation
 - [HELM 28](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2028%20Differential%20Vector%20Calculus.pdf): Differential Vector Calculus
 - [HELM 13](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2013%20Integration.pdf): Integration
 - [HELM 16](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2016%20Sequences%20and%20Series.pdf): Sequences & Series (includes binomial and Taylor)
 - [HELM 35](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2035%20Sets%20and%20Probability.pdf): Sets and Probability (probability essentials)
 - [HELM 36](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2036%20Descriptive%20Statistics.pdf): Descriptive Statistics (sample mean and variance)
- [HELM 37](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2037%20Discrete%20Probability%20Distributions.pdf): Discrete Probability Distributions (Binomial and Poisson)
- [HELM 38](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2038%20Continuous%20Probability%20Distributions.pdf): Continuous Probability Distributions (Uniform and Expon)
- [HELM 39](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2039%20The%20Normal%20Distribution.pdf): The Normal Distribution (includes CLT)
-[HELM 40](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2040%20Sampling%20Distributions%20and%20Estimation.pdf): Sampling and Estimation (SEM, CUE, Point:mean, Interval:variance)
- [HELM 41](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2041%20Hypothesis%20Testing.pdf): Hypothesis Testing (includes Z and T)
- [HELM 42](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2042%20Goodness%20of%20Fit%20and%20Contingency%20Tables.pdf): Goodness of Fit (Chi Squared)
- [HELM 43](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2043%20Regression%20and%20Correlation.pdf): Regression and Correlation (Least Squares, Pearson correlation coefficient)
- [HELM 44](https://www.lboro.ac.uk/media/media/schoolanddepartments/mlsc/downloads/HELM%20Workbook%2044%20Analysis%20of%20Variance.pdf): Analysis of Variance (Anova)

:::

- [Stats Course by Glen Cowan](https://www.pp.rhul.ac.uk/~cowan/stat_course.html): If this is not your first rodeo, this is a good lecture course to dip into for more challenging work. It is the one I followed as a PhD student, and is geared towards particle physicists. 
- [Stats Course by Mark Thomson](https://www.hep.phy.cam.ac.uk/~thomson/lectures/lectures.html): If this is not your second rodeo, these is an excellent set of very condensed notes - best suited to experts who need to brush up. Also geared towards particle physicists.


<!--## What issues might we face?

1. Bad information: on the topics of statistics and probability, there is an especially high amount of trash on the internet. I think this is because truly understanding the concepts behind the tools requires a lot of study and re-study, and the topics needed are quite long and boring with no clear pathways and definitely no quick rewards.
2. Unclear information: this is partly a notation issue - if you don't have a degree in maths the notation is misery-inducing - but I have also found that experts and educators are unneccessarily vague on the particularly hard and/or boring topics. I think this is because they are scared of saying something incorrect as per point 1.
3. Inaccessible information: the assumption of prior knowledge and/or the use of terminology, notation that is unfamiliar to the students.-->



