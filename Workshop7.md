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

(workshop-7)=
# Workshop 7: Numerical MLE with `scipy.optimize`

This workshop covers material from [Maximum Likelihood Estimation](#chapter:MaxLike): the Negative Log Likelihood ([](#sec:nll)), numerical minimisation, and uncertainty estimation from the curvature of the NLL.

You will work with two toy datasets:

- `DataW7.txt` — integer counts recorded at a surface particle detector (one column, one count per line)
- `DataW7b.txt` — continuous energy measurements in GeV (one column, one value per line)

Load them as follows:

```python
import numpy as np
counts = np.loadtxt('DataW7.txt', dtype=int)   # integer Poisson counts
energies = np.loadtxt('DataW7b.txt')            # float energy values
```

---

## Task 1: True or False

State whether each of the following is **True** or **False**, and give a one-sentence justification.

1. `scipy.optimize.minimize_scalar` can only find the minimum of a function of one variable.

2. If the NLL has a unique minimum at $\hat\lambda$, then doubling the dataset size moves the minimum but does not change its curvature.

3. The Hessian of the NLL at its minimum provides an estimate of the **variance** of the MLE.

4. `result.hess_inv` returned by `scipy.optimize.minimize` with method `'L-BFGS-B'` is an exact Hessian matrix.

5. For a Poisson distribution, the MLE of $\lambda$ equals the sample mean $\bar{x}$.

6. For a Gaussian distribution with **both** $\mu$ and $\sigma$ unknown, a 2D NLL surface is needed; the MLE cannot be found from two separate 1D minimisations.

---

## Task 2: Poisson MLE — one parameter, one method

The `counts` dataset contains Poisson-distributed integers. Recall from the notes that the Poisson NLL for a dataset $\{x_i\}$ is

$$
-\ln\mathcal{L}(\lambda) = -\sum_i \bigl[x_i \ln\lambda - \lambda - \ln(x_i!)\bigr],
$$

which simplifies (dropping constants in $\lambda$) to

$$
\text{NLL}(\lambda) = N\lambda - \bigl(\sum_i x_i\bigr)\ln\lambda.
$$

a) Write a Python function `nll_poisson(lam, data)` that returns $\text{NLL}(\lambda)$ for an array `data`. Use the full expression with `scipy.special.gammaln(data + 1)` (so the value is comparable across datasets).

b) Use `scipy.optimize.minimize_scalar` with `method='bounded'` and `bounds=(0.5*counts.mean(), 2*counts.mean())` to find $\hat\lambda$. Print the result and compare it to the sample mean `counts.mean()`.

c) Plot the NLL as a function of $\lambda$ over a sensible range. Mark $\hat\lambda$ with a vertical dashed line. Label the axes clearly.

d) Estimate the **uncertainty** on $\hat\lambda$ using the NLL offset method: find the two values of $\lambda$ where $\text{NLL}(\lambda) = \text{NLL}(\hat\lambda) + 0.5$. You may do this graphically or numerically (hint: `scipy.optimize.brentq`). What is $\hat\lambda \pm \delta\lambda$?

---

## Task 3: Gaussian MLE — two parameters via `minimize`

The `energies` dataset contains measurements drawn from a Gaussian distribution with unknown mean $\mu$ and standard deviation $\sigma$.

a) Write a function `nll_gaussian(params, data)` where `params = [mu, sigma]`. Use `scipy.stats.norm.logpdf` or write the Gaussian log-PDF explicitly. Return the negative sum of log-PDF values.

b) Minimise the 2D NLL using `scipy.optimize.minimize` with:
   - `method='L-BFGS-B'`
   - `x0 = [energies.mean(), energies.std()]`
   - `bounds = [(None, None), (0.01, None)]`  (keep $\sigma > 0$)

   Print the MLE values $\hat\mu$ and $\hat\sigma$.

c) Extract the estimated **standard errors** from the inverse Hessian:

   ```python
   hess_inv = result.hess_inv.todense()
   sigma_mu  = np.sqrt(hess_inv[0, 0])
   sigma_sig = np.sqrt(hess_inv[1, 1])
   ```

   Report $\hat\mu \pm \sigma_\mu$ and $\hat\sigma \pm \sigma_\sigma$. Do the uncertainties look reasonable for your dataset size?

d) Make a 2D filled contour plot of the NLL surface in $(\mu, \sigma)$ space around the minimum. Use `np.meshgrid` with about 40 points on each axis spanning ±3 standard errors from the MLE. Overplot the MLE as a white marker. Add a contour at $\text{NLL}_\min + 0.5$ (the 1$\sigma$ boundary) and at $\text{NLL}_\min + 2.0$ (the 2$\sigma$ boundary).

---

## Task 4: Model Comparison with the LLR

A colleague tells you the energies were measured using one of two instruments, each with a known systematic offset but the same $\sigma = 0.72$ GeV. The two competing models are:

- **Model A**: $\mu_A = 3.15$ GeV (calibrated instrument)
- **Model B**: $\mu_B = 3.50$ GeV (uncalibrated instrument)

Both models have $k = 0$ free parameters (the values are fixed — not fitted to the data).

a) Compute the log-likelihood $\ln\mathcal{L}_A$ and $\ln\mathcal{L}_B$ for each model. Which is larger?

b) Compute the Log Likelihood Ratio statistic:
   $$
   \Lambda = -2(\ln\mathcal{L}_B - \ln\mathcal{L}_A).
   $$
   A positive $\Lambda$ favours Model A. Is the difference substantial?

c) Compute $\Delta\text{AIC} = \text{AIC}_B - \text{AIC}_A$. Since both models have $k = 0$ free parameters, what is the relation between $\Delta\text{AIC}$ and $\Lambda$? Verify this numerically.

d) Based on your results, write a one-paragraph summary stating which model is preferred and how confident you are.

