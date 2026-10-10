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

# Workshop 3: Covariance & Correlations


## Task 1: Covariance of two datasets

Measurements of two RVs are as follows: 

```
X1 = [18_841, 20_449, 20_987, 21_854, 22_778, 24_075, 25_253, 26_155, 27_227, 27_092,]

X2 = [11, 16.3333, 23.75, 29.8333, 34.3333, 42.25, 47.5, 53.25, 57.0833, 58.1667,]
```

* a) Calculate the mean, variance, and standard deviation of each RV

* b) Find the covariance matrix for these two RVs

* c) What do the results indicate to you?


## Task 2: Linear Correlations

The kinetic energy $E$ of a sports cars is related to its speed $v$ as $E\propto v^2$.

a) Extract 100 values of speed from a Uniform(-160,160) distribution.

b) Plot energy (y-axis) versus speed values (x-axis)

c) Find the covariance cov(E,v) and correlation coefficient $\rho(E,v)$

d) What do the results indicate to you?


## Task 3: Optional Math Task (if this is not your first rodeo)

A function of three RVs is $f(x,y,z) = 2x + 3y + 4z = \alpha_1 x_{(1)} + \alpha_2 x_{(2)} + \alpha_3 x_{(3)}$.

Two methods for finding the variance of this function are: 

(1) Using Expectation Algebra:

$V[f]  = \sum\limits_{i=1}^{3} \sum\limits_{j=1}^{3} \alpha_i \alpha_j \mathsf{cov} [x_{(i)}, x_{(j)}]$.    


(2) Using the first two terms of the Taylor expansion for $f$. 

$V[f]  = \sum\limits_{i=1}^{3} \sum\limits_{j=1}^{3} \dfrac{\partial f}{\partial x_{(i)} }  \dfrac{\partial f}{\partial x_{(j)} } \bigg|_{x_{(j)}=\mu} \mathsf{cov}[x_{(i)}, x_{(j)}]$. 

a) Show that these methods give the same answer for this function.



## Example Solutions

To follow after the workshop.

