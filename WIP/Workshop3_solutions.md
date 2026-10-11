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

## Example Solution

```{code-cell} python

# Data from https://www.tylervigen.com/spurious/correlation/

import numpy as np
import matplotlib.pyplot as plt

X1 = np.array([18_841, 20_449, 20_987, 21_854, 22_778, 24_075, 25_253, 26_155, 27_227, 27_092])

print("\n X1 mean: {},  var: {}, sigma: {}".format( np.mean(X1), np.var(X1), np.std(X1) ) )

X2 = np.array([11, 16.3333, 23.75, 29.8333, 34.3333, 42.25, 47.5, 53.25, 57.0833, 58.1667])

print("\n X2 mean: {},  var: {}, sigma: {}".format( np.mean(X2), np.var(X2), np.std(X2) ) )

# Always plot the data!

fig, ax = plt.subplots()

ax.scatter(X1, 
	    X2, 
	    color='skyblue', 
	    marker='*',
	    label='Data' 
	    )

ax.set_xlabel("X1")
ax.set_ylabel("X2")
plt.legend()
plt.show()


cov_mat = np.cov(X1,X2)
print("\n Covariance matrix: \n",cov_mat)

print(f"\n V[X1] (cov): {cov_mat[0][0]}")
print(f"\n V[X1] (var): {np.var(X1)}")

print("\n Yikes, why does cov(x,y)[0,0] not give me the same answer as var(X1)?!")

# this is because numpy var defaults to the biased variance calculation with 1/N normalisation, whereas cov uses the unbiased 1/(N-1) normalisation

print("\n X1 unbiased var: {}".format( np.var(X1,ddof=1) ) )
print("\n X2 unbiased var: {}".format( np.var(X2,ddof=1) ) )

print("\n With var(Xi,ddof=1) we get the matching diagonals")

cc_mat = np.corrcoef(X1,X2)
print("\n Correlation matrix: \n",cc_mat)

print("\n We notice a very high positive linear correlation of ", cc_mat[0,1])

print("\n This indicates a strong linear correlation between X1 and X2. So it may surprise you to learn the source of these data...")

print("\n X1: USA bachelors degrees awarded in math and stats, per year, ordered by year")

print("\n X2: google searches for reddit (relative to some baseline)")

print("\n For more spurious correlations, see https://www.tylervigen.com/spurious/correlation/")
```

---

## Task 2: Linear Correlations

The kinetic energy $E$ of a sports cars is related to its speed $v$ as $E\propto v^2$.

a) Extract 100 values of speed from a Uniform(-160,160) distribution.

b) Plot energy (y-axis) versus speed values (x-axis)

c) Find the covariance cov(E,v) and linear correlation coefficient $\rho(E,v)$

d) What do the results indicate to you?

## Example Solution for Task 2

```{code-cell}python
from scipy.stats import uniform
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
fig1, ax1 = plt.subplots(1, 1)

uniform_dist = uniform(-160,320)
speeds = uniform_dist.rvs(size=100)
energies = 0.5*speeds**2

ax1.scatter(speeds, energies)
ax1.set_xlim(-160,160)
plt.show()

cov_mat_ev = np.cov(energies, speeds)
print("\n Covariance matrix: \n",cov_mat_ev)

cc_mat_ev = np.corrcoef(energies, speeds)
print("\n Linear Correlation matrix: \n",cc_mat_ev)

print("\n Linear Correlations are tiny: {}".format( cc_mat_ev[0,1] ) )
print("\n Indication is that v and E are not *linearly correlated*. This is expected because E~v^2.")

```

---

## Optional Math Task (if this is not your first rodeo)

A function of three RVs is $f(x,y,z) = 2x + 3y + 4z = \alpha_1 x_{(1)} + \alpha_2 x_{(2)} + \alpha_3 x_{(3)}$.

Two methods for finding the variance of this function are: 

(1) Using Expectation Algebra:

$V[f]  = \sum\limits_{i=1}^{3} \sum\limits_{j=1}^{3} \alpha_i \alpha_j \mathsf{cov} [x_{(i)}, x_{(j)}]$.    


(2) Using the first two terms of the Taylor expansion for $f$. 

$V[f]  = \sum\limits_{i=1}^{3} \sum\limits_{j=1}^{3} \dfrac{\partial f}{\partial x_{(i)} }  \dfrac{\partial f}{\partial x_{(j)} } \bigg|_{x_{(j)}=\mu} \mathsf{cov}[x_{(i)}, x_{(j)}]$. 

a) Show that these methods give the same answer for this function.



## Example Solution to math task

**Speedy Answer**

For the second method, we have partial derivatives in place of the coefficients $\alpha$. We can quickly see that the partial derivatives of a linear function do indeed return the coefficents. For example:

$\dfrac{\partial f}{\partial x_{(1)} }\bigg|_{x=\mu} = \dfrac{\partial}{\partial x_{(1)} }\left(\alpha_1 X_{(1)} + \alpha_2 X_{(2)} + \alpha_3 X_{(3)}\right) = \alpha_1$

This is sufficient to show that the two approaches are equivalent for a linear function.

Notice that the definition has a specifier that after taking the derivative, we evaluate the result at $x=\mu$. This is irrelevant for a linear function, because there will not be any $x$ terms after taking the derivatives.

**A bit more detail:**

For the first method note that the $i,j$ sums are cycling through pairs, so we get $3^2=9$ terms.

$\alpha_1=2, \alpha_2=3, \alpha_3=4$

$V[f] = \alpha_1 \alpha_1\, \mathsf{cov}(x_{(1)}, x_{(1)}) + \alpha_1 \alpha_2\, \mathsf{cov}(x_{(1)}, x_{(2)}) + \alpha_1 \alpha_3 \,\mathsf{cov}(x_{(1)}, x_{(3)}) +$

$\;\;\;\;\;\;\;\;\;\;\alpha_2 \alpha_1\, \mathsf{cov}(x_{(2)}, x_{(1)}) + \alpha_2 \alpha_2\, \mathsf{cov}(x_{(2)}, x_{(2)}) + \alpha_2 \alpha_3\, \mathsf{cov}(x_{(2)}, x_{(3)}) +$

$\;\;\;\;\;\;\;\;\;\;\alpha_3 \alpha_1\, \mathsf{cov}(x_{(3)}, x_{(1)}) + \alpha_3 \alpha_2\, \mathsf{cov}(x_{(3)}, x_{(2)}) + \alpha_3 \alpha_3\, \mathsf{cov}(x_{(3)}, x_{(3)})$

---

