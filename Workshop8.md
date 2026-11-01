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

(workshop-8)=
# Workshop 8: Linear Regression and Model Comparison

This workshop covers two related topics: fitting a linear model using `scipy.optimize.curve_fit` (Least Squares), and comparing nested and non-nested models using the log-likelihood ratio, Pearson $\chi^2$, and AIC.

Material is drawn from [Maximum Likelihood Estimation](#chapter:MaxLike) and [Least Squares](#chapter:LeastSquares). You should be comfortable with the Poisson NLL from Workshop 6 and 7 — this workshop extends that to model selection.

You will work with two toy datasets:

- `DataW8_regression.txt` — a three-column file with columns `x_cm`, `y_N`, `sigma_N` (position, force measurement, measurement uncertainty). Load with:
  ```python
  x, y, sigma = np.loadtxt('DataW8_regression.txt', skiprows=1, unpack=True)
  ```
- `DataW8_models.txt` — a four-column file with columns `bin_centre`, `D`, `M1`, `M2` (bin centres, observed counts, and two model predictions). Load with:
  ```python
  bc, D, M1, M2 = np.loadtxt('DataW8_models.txt', skiprows=1, unpack=True)
  D = D.astype(int)
  ```

---

## Task 1: True or False

State whether each of the following is **True** or **False**, and give a one-sentence justification.

1. `scipy.optimize.curve_fit` minimises the sum of squared residuals weighted by the measurement uncertainties $\sigma_i$.

2. The covariance matrix `pcov` returned by `curve_fit` has diagonal entries equal to the **variances** of the fitted parameters.

3. For Poisson-distributed counts, using the unweighted $\chi^2$ statistic (without $\sigma_i^2$ in the denominator) is the correct goodness-of-fit measure.

4. The Pearson $\chi^2$ and the LLR statistic always give the same numerical value for Poisson data.

5. A model with more parameters always has a lower AIC than a simpler model, because it fits the data better.

6. A $\Delta\text{AIC}$ of 2 between two models (with $k_A = k_B = 0$) is conventionally considered **substantial** evidence for the preferred model.

---

## Task 2: Linear Regression with `curve_fit`

The dataset `DataW8_regression.txt` contains measurements of a force $y$ (in Newtons) at positions $x$ (in cm), each with uncertainty $\sigma$.

a) Plot the data with error bars (`plt.errorbar`). Add a title and axis labels.

b) Define a linear model function `linear(x, m, c)` returning $mx + c$. Use `scipy.optimize.curve_fit` with the `sigma` column as the `sigma` argument and `absolute_sigma=True` to fit the model. Print the fitted slope $\hat m$ and intercept $\hat c$ with their uncertainties.

c) Overplot the best-fit line on your data plot. Extend the line slightly beyond the data range.

d) Compute the weighted **residuals** $r_i = (y_i - \hat y_i) / \sigma_i$ for each data point. Make a residual plot (residual vs $x$). Do the residuals look randomly distributed around zero, or is there a systematic pattern?

e) Compute the reduced chi-squared statistic:
   $$
   \chi^2_\nu = \frac{1}{N - k}\sum_i \left(\frac{y_i - \hat y_i}{\sigma_i}\right)^2,
   $$
   where $k = 2$ (slope and intercept). A value near 1 indicates a good fit. What do you find?

---

## Task 3: Model Comparison for Binned Count Data

The dataset `DataW8_models.txt` contains binned count data from a particle detector. The columns `D` (observed counts), `M1` and `M2` give the predicted counts under two competing models. Both models have $k = 0$ free parameters (no fitting to this data was performed — the predictions are fixed).

a) Plot `D`, `M1`, and `M2` as a function of `bin_centre` on the same axes. Use `plt.step` for the counts and `plt.plot` for the model curves. Label everything clearly.

b) Write a function `poisson_log_likelihood(D, M)` that computes
   $$
   \ln\mathcal{L} = \sum_i \bigl[D_i \ln M_i - M_i - \ln(D_i!)\bigr].
   $$
   Use `scipy.special.gammaln` for the factorial term. Compute $\ln\mathcal{L}_1$ and $\ln\mathcal{L}_2$. Which model fits better?

c) Compute the Log Likelihood Ratio:
   $$
   \Lambda = -2(\ln\mathcal{L}_2 - \ln\mathcal{L}_1).
   $$
   What does its sign tell you? If $k_1 = k_2 = 0$, what is $\Delta\text{AIC}$?

d) Compute the **Pearson $\chi^2$** statistic for each model:
   $$
   \chi^2 = \sum_i \frac{(D_i - M_i)^2}{M_i}.
   $$
   The expected value under a good fit is approximately equal to the number of bins. Do either or both models fit well?

e) Write a short summary (3–4 sentences) comparing the two models using all three statistics. Which model would you choose, and why?

---

## Task 4: Fitting a Quadratic Extension

Return to the regression data from Task 2. A colleague suggests the relationship might be quadratic: $y = ax^2 + bx + c$.

a) Define a `quadratic(x, a, b, c)` function and fit it using `curve_fit`. Print the fitted parameters and their uncertainties.

b) Compute the reduced $\chi^2_\nu$ for the quadratic fit. Compare it to the linear fit's $\chi^2_\nu$.

c) Compute $\Delta\text{AIC} = \text{AIC}_\text{quad} - \text{AIC}_\text{lin}$.

   Hint: for a Gaussian likelihood with known $\sigma_i$, the log-likelihood is $\ln\mathcal{L} = -\frac{1}{2}\sum_i \left(\frac{y_i - \hat y_i}{\sigma_i}\right)^2 + \text{const}$, so $\text{AIC} = 2k + \chi^2_\text{weighted}$ (up to constants independent of the model).

d) Based on $\Delta\text{AIC}$ and the visual evidence, is the extra complexity of the quadratic model justified?

