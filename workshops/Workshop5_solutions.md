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

(workshop-5-solutions)=
# Workshop 5 Solutions: Normal Tests


### Task 1 Answers

1. **True** — $\mathsf{SEM} = \sigma_{true}/\sqrt{N}$, so it shrinks as $N$ grows.

2. **True** — The notes state the two distributions are "barely distinguishable on a linear scale" for $N \gtrapprox 30$.

3. **True** — The Z test statistic uses $\mathsf{SEM} = \sigma_{true}/\sqrt{N}$, which requires $\sigma_{true}$.

4. **False** — The p value is the probability of observing data *at least this extreme*, given that $H_0$ is true. It says nothing directly about the probability that $H_0$ itself is true.

5. **False** — The two-sample T test works for equal or unequal sample sizes; the pooled variance and multiplier $\psi$ handle this.

6. **True** — Within a group, all observations share the same group membership, so the group label cannot explain why individual values differ from their group mean.

7. **True** — $\nu_{within} = N_x + N_y - N_s = 15 + 15 - 2 = 28$.

8. **False** — The Z test requires the true variance to be known. The T test does not — it uses the estimated SEM instead.

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

### Task 2 Example Solution

```{code-cell} python
import numpy as np
from scipy.stats import t

# Assumes true_mu, true_sigma, B_data, S_data, G_data are already defined
# by running three_cities_data() from the Normal Tests notebook.

# --------------------------------------------------------
# (a) Test design
# --------------------------------------------------------

mu0    = true_mu  # H0: mean hours slept = 7.33 H
alpha  = 0.05
nsided = 2        # H1: mean != 7.33 H (two-sided)

# --------------------------------------------------------
# (b) Look at data for N=30
# --------------------------------------------------------

N = 30
data = G_data[:N]

mean    = np.mean(data)
sigma   = np.std(data, ddof=1)
est_sem = sigma / np.sqrt(N)

print(f"Glasgow N={N}: mean={mean:.3f} H, sigma={sigma:.3f} H")
print(f"Estimated SEM = {est_sem:.3f} H")

# --------------------------------------------------------
# (c) The T test is appropriate because we do not know sigma_true
# --------------------------------------------------------

# --------------------------------------------------------
# (d) T test statistic and p value
# --------------------------------------------------------

nu    = N - 1
Tstat = (mean - mu0) / est_sem

p_value = nsided * t.sf(abs(Tstat), df=nu)

print(f"\nT statistic = {Tstat:.3f}")
print(f"Degrees of freedom nu = {nu}")
print(f"P value = {p_value:.3f}")

if p_value > alpha:
    print("=> Test Negative: do not reject H0")
else:
    print("=> Test Positive: reject H0!")

# --------------------------------------------------------
# (e) Repeat for N=3
# --------------------------------------------------------

N_small = 3
data_small  = G_data[:N_small]
mean_small  = np.mean(data_small)
sigma_small = np.std(data_small, ddof=1)
est_sem_small = sigma_small / np.sqrt(N_small)

Tstat_small   = (mean_small - mu0) / est_sem_small
p_value_small = nsided * t.sf(abs(Tstat_small), df=N_small-1)

print(f"\nGlasgow N={N_small}: mean={mean_small:.3f} H, sigma={sigma_small:.3f} H")
print(f"Estimated SEM = {est_sem_small:.3f} H")
print(f"T statistic = {Tstat_small:.3f}")
print(f"Degrees of freedom nu = {N_small-1}")
print(f"P value = {p_value_small:.3f}")

if p_value_small > alpha:
    print("=> Test Negative: do not reject H0")
else:
    print("=> Test Positive: reject H0!")

print(f"\nFor N=3: the estimated SEM is large because sigma is estimated from only 3 points.")
print(f"The T distribution with nu=2 also has fat tails, making it harder to reject H0.")
print(f"Both effects mean we need stronger evidence to reach the same conclusion with tiny samples.")
```

---

### Task 3 Example Solution

```{code-cell} python
import numpy as np
from scipy.stats import t, ttest_ind

alpha  = 0.05
nsided = 2   # H1: means are different (two-sided)
             # H0: Stockport and Glasgow have the same mean

NS = 30
NG = 30

dataS = S_data[:NS]
dataG = G_data[:NG]

meanS = np.mean(dataS)
meanG = np.mean(dataG)
varS  = np.var(dataS, ddof=1)
varG  = np.var(dataG, ddof=1)
nuS   = NS - 1
nuG   = NG - 1

# (b) Check equal-variance condition: ratio of standard deviations should be 0.5 < r < 2
ratio = np.sqrt(varS) / np.sqrt(varG)
print(f"Sigma ratio sigma_S / sigma_G = {ratio:.2f}")
if 0.5 < ratio < 2:
    print("Equal-variance condition satisfied.")
else:
    print("Warning: variances differ too much for the standard two-sample T test.")

# (c) Pooled variance
var_p = (nuS * varS + nuG * varG) / (nuS + nuG)

# Multiplier psi
psi = np.sqrt(1/NS + 1/NG)

# Test statistic
T2 = (meanS - meanG) / (psi * np.sqrt(var_p))

# P value
p_value = nsided * t.sf(abs(T2), df=nuS + nuG)

print(f"\nStockport: mean={meanS:.3f} H, var={varS:.3f}")
print(f"Glasgow:   mean={meanG:.3f} H, var={varG:.3f}")
print(f"Pooled variance = {var_p:.3f}")
print(f"Psi = {psi:.3f}")
print(f"T2 statistic = {T2:.3f}")
print(f"P value = {p_value:.3f}")

if p_value > alpha:
    print("=> Test Negative: do not reject H0")
else:
    print("=> Test Positive: reject H0!")

# (d) Verify with scipy
T2_scipy, pval_scipy = ttest_ind(dataS, dataG)

print(f"\nScipy ttest_ind: T={T2_scipy:.3f}, p={pval_scipy:.3f}")
print("(Should match our manual calculation.)")
```

---

### Optional Extension Example Solution

Will be shared after Coursework Asssignment 2 has been marked.
