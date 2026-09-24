# Exercises

(ex:mean-var-std)=
## Mean, Variance, and Standard Deviation

Here is a dataset:

```x = [0.88472455, 0.96232018, 0.10713343, 0.41198233, 0.06543451,0.40068931, 0.54846767, 0.46301972, 0.4534936 , 0.2064886 ]```

Write python functions to calculate the mean, variance, and standard deviation of this dataset.

Compare your function outputs with ```numpy```'s built in methods for these calculations



(ex:cov)=
## Covariance

Generate 3 datasets, x, y, z, using RVs drawn from the ```scipy.stats uniform``` distribution:

```
from scipy.stats import uniform
import numpy as np
u = uniform(0,1)
x = u.rvs(size=10)
y = u.rvs(size=10)
z = y**2
xyz = np.stack((x,y,z))

1. What is the standard deviation $\sigma_x$?

2. What are the covariance terms cov(x,y) and cov(x,z)?

3. What is the variance of z?

4. What are the linear correlation coefficients $\rho(x,y)$, $\rho(y,z)$, and $\rho(z,x)$?

Write down in words what this tells us (and does not tell us) about the relationships between each of the three datasets.

```

(ex:prob1)=
## Probability 1

1. If there are 7 blue balls and 9 green balls in a bucket, what is the probability that a ball selected at random will be blue?

2. If the first ball selected is a blue ball, and it is not returned to the bucket, what is the probability that the second ball selected will also be blue?

3. In this experiment, are my data IID?



(ex:prob2)=
## Probability 2

Here is a table of some data relating to a test for some illness.

|      | Positive   | Negative  |
| ---  | --- | --- | 
| Sick   | 25  | 2   | 
| Not Sick  | 37  | 219  | 


1. According to these data, what is the conditional probability that you are sick, given that you test positive?

2. What is the conditional probability that you will test positive, if you are sick?


(ex:bayes1)=
## Bayes Theorem

You are given the following information:

1. In random testing, someone tests positive for a disease.
2. In 5% of cases where the subject does not have the disease, this test shows positive anyway: i.e. there is a 5% false positive rate
3. There are no false negatives.
4. In the population at large, one person in a thousand has the disease.

> What is the probability that the person tested actually has the disease?



(ex:pdf-normalisation)=
## PDF Normalisation

A function of a Random Variable X is $f(X) = Ax^{2}$  in the range $x: [0,1]$.

In order for this function to be used as a probability density, it must have integral=1 (total probability=1).

a) Find the coefficient $A$ that normalises  $f_X$ in the range $x:[0,1]$

b) Calculate the Expected Value E[X] using the normalised $f_X$

c) Calculate the True Variance V[X]


(ex:weighted-mean)=
## Weighted Mean

My good friend Alfred has become obsessed with weighing packets of rolos (this happens to him from time to time - he will be fine).

This is his set of measurements for the weights of five different packs (in grams)

$x_{A} = [52.010, 52.041, 52.105, 51.998, 51.981]$

I decide to help him and make these measurements:

$x_{L} =[52.05, 52.03, 52.07, 51.90, 51.94]$

a) Calculate the weighted mean of these two datasets
b) Which of them holds more weight, and why?


(ex:joint-pmf-and-likelihood)=
## Discrete Joint PMF and Likelihood


The measurements of two discrete RVs $X,Y$  correspond to the values (heads,tails =1,0) from two coin tosses. 

a) Write down the Joint PMF for the pair of coin tosses, and calculate the probability of observing both coins as heads.

b) In this example you will have used your prior knowledge of the probability of heads being 0.5 (a fair coin). What is the Likelihood that the coin is fair?

c) Now assume the coin is not fair, and has a 0.8 probability of heads. Recalculate the Likelihood with this new hypothesis. 

d) Is it more likely that the coin is fair or unfair, given the data?



(ex:uniform-pdf-and-cdf)=
## Uniform PDF and CDF

The Uniform PDF has flat probability in a region between two bounds $x: [a,b]$ and zero probability outside that range:

\begin{equation}
f_X(x) = 
\begin{cases}
     0\; & x < a\\[1ex] 
  \dfrac{1}{b-a}\; & a\leq x \leq b \\[2ex]
  0\; & x>b \\[2ex]
\end{cases}
\end{equation}

**a) Evaluate the CDF $F_X$ for the three ranges in (1) above.** 

*Note that you will need a separate integral for each range, because $f_X(x)$ is not continuous at the boundaries $x=a$ and $x=b$. Also note that the CDF is cumulative, so you will have sums in all but the first region. Format here so you don't waste your precious time:*

```{math}
F_X(x) = P(X<x) =
\begin{cases}
     \int\limits_{-\infty}^{a}\,f_X\,dx\; & x < a\\[1ex] 
  \int\limits_{-\infty}^{a}\,f_X\,dx + \int\limits_{a}^{x}\,f_X\,dx  & a\leq x \leq b \\[2ex]
  \int\limits_{-\infty}^{a}\,f_X\,dx + \int\limits_{a}^{b}\,f_X\,dx + \int\limits_{b}^{x}\,f_X\,dx \; & x>b \\[2ex]
\end{cases}
```

**b) Show that the Uniform PDF described by (1) above is a valid PDF for a continuous RV $X$ in the range $-1\leq x \leq 1$.**

*Valid $\equiv$ Normalised $\equiv$ Integral is one.*

**c) Write down the PDF $f_X(x)$ and CDF $F_X(x)$ for $-1\leq x \leq 1$.**

*Your answer will be a number for the PDF, and a function of $x$ for the CDF.*


(ex-change-of-variables)=
## Change of Variables

Consider a new RV, $Y(X)=X^2$. 

a) What is the range of values for $y$ with non-zero probability?

b) Is your answer to part a) the *support* or *sample space*?

c) Write down the CDF and PDF for $Y$.




(ex-dice-rolling)=
## Dice Rolling

The probability of me rolling a dice and getting a six is $p=\frac{1}{6}$. Last week I was feeling lucky, so I rolled a dice 100 times and got a six 21 times. 

a) Identify the probability distribution you would use in this case, stating why.

b) Use it to calculate the probability of this occurring "naturally".

c) Plot the distribution, indicating this outcome on the plot.


(ex-email-rate)=
## Email Rate

On average, I receive 100 emails between the hours of 08:00 and 18:00 UK time, or ten emails per hour over this period. Last week I got the heeby jeebies after receiving zero emails for just over 2 hours, and wondered if the email server was down. It turns out I was just lucky! 

a) Identify the probability distribution you would use in this case, stating why.

b) Use it to calculate the probability of this occurring "naturally".

c) Plot the distribution, indicating this outcome on the plot.


(ex-expon-and-transform)=
## Expon and Transform

Let X ~ Expon(2) and Y=2+3X

a) Find P(X>2)

b) Find E[Y] and Var[Y]


(ex-covariance-and-correlation-coin-toss)=
## Covariance and Correlation: Coin Toss

I toss a coin three times. I define X as the number of heads, and Y as the number of tails.

Use a pen an paper to: 

a) find the covariance, cov(X,Y).

b) find the correlation coefficient, $\rho(X,Y)$


(ex-covariance-and-correlation-rvs)=
## Covariance and Correlation of RVs

Measurements of two RVs are as follows: 

X1 = [18_841, 20_449, 20_987, 21_854, 22_778, 24_075, 25_253, 26_155, 27_227, 27_092,]

X2 = [11, 16.3333, 23.75, 29.8333, 34.3333, 42.25, 47.5, 53.25, 57.0833, 58.1667,]

a) Calculate the mean, variance, and standard deviation of each RV

b) Find the covariance matrix for these two RVs

c) What do the results indicate to you?


(ex-correlation-and-independence)=
## Correlation and Independence

The kinetic energy E of a sports cars is related to its speed v as $E\propto v^2$.

a) Extract 100 values of speed from a Uniform(-160,160) distribution.

b) Plot energy (y-axis) versus speed values (x-axis)

c) Find the covariance cov(E,v) and correlation coefficient $\rho(E,v)$

d) What do the results indicate to you?



(ex-variance-linear-function-2RVs)=
## Variance of a Linear Function of Two RVs

A function of two RVs is $f(X,Y) = 3x+y$. 

You have the following data measurements: 

x=[62, 69, 67, 63, 63, 69, 67]

y=[12, 11, 10, 14, 12, 13, 13]

a) Express the mean and uncertainty of the function in the form $f(X,Y) = \overline{f} \pm \sigma_f$


(ex-variance-linear-function-3RVs)=
## Variance of a Linear Function of 3 RVs

A function of three RVs is $f(X) = 2X_{(1)} + 3X_{(2)} + 4X_{(3)}$.

Different methods for finding the variance of this function were given in lectures: 

$V[f]  = \sum\limits_{i=1}^{3} \sum\limits_{j=1}^{3} \alpha_i \alpha_j \mathsf{cov} [X_{(i)}, X_{(j)}]$ using expectation algebra, where $f = \alpha_1 X_{(1)} + \alpha_2 X_{(2)} + \alpha_3 X_{(3)}$.    

$V[f]  = \sum\limits_{i=1}^{3} \sum\limits_{j=1}^{3} \dfrac{\partial f}{\partial x_{(i)} }  \dfrac{\partial f}{\partial x_{(j)} } \bigg|_{x_{(j)}=\mu} \mathsf{cov}[x_{(i)}, x_{(j)}]$ using the first two terms of the Taylor expansion for $f$. 

a) Use pen and paper to show that these methods give the same answer for this function.


(ex-binomial-means)=
## Binomial Means are Gaussian Distributed

Gaussian-And-Stats.py has the python for the example of uniform means forming a Gaussian distribution, as we saw in the lectures.

Use this to write a python script to extract values from a Binomial distribution (you choose the parameters), calculate the mean, then repeat such that you have a large set of means.

a) Plot the histogram of binomial means, and fit a gaussian to it.

b) How many means do you need to calculate to get a feasible (by eye) fit from the Gaussian?


(ex-Z-value-dog-show)=
## Z values at a Dog Show

A dog show features 200 female golden retrievers whose weights follow a Gaussian distribution with a mean 27.2kg  and standard deviation 2.58 kg.

My friend Geraldine who is also a golden retriever shows up to check out the competition. She weighs 33kg without her shoes on.

a) What is Geraldine's z-value? Give three significant figures.

b) What is Geraldine's p-value? Give three significant figures.

c) How many of the other dogs are heavier than her?


(ex-bivariate-gaussian-contours)=
## Bivariate Gaussian contours

Gaussian-And-Stats.py has the python for plotting a bivariate Gaussian distribution with two choices of covariance matrix.

a) Can you choose any values for the covariance matrix elements? What are the limitations?

b) The contours in the example I provided are automatically selected. See if you can set them to colour the contours according to the 1,2,3,.. sigma probabilities. You will need to refer back to the lecture slides to set up an array of PDF values at different sigmas, and use the seaborn plot_joint levels flag to set the levels manually to this array.

c) Extension: can you make a 3D plot of the bivariate Gaussian, with the z-axis being the PDF value?



(ex-nonlinear-least-squares)=
## Nonlinear Least Squares

EstimatorsIntro.py has the python for generating fake data points and doing a linear least squares fit. You will modify this to a nonlinear function of x for this exercise.

a) Make a plot of the data points following an $x^2$ distribution, showing the residuals and the Least Squares Fit.


## Likelihood for Normal Sigma

EstimatorsIntro.py has the python for plotting the Likelihood, Log Likelihood, and Negative Log Likelihood for the loc parameter (the expected value, $\mu$) from a normal distribution.

a) Adjust the provided code to make the three Likelihood plots for the scale parameter ($\sigma$, the standard deviation) of the normal distribution.





