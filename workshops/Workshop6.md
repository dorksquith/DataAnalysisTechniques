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

(workshop-6)=
# Workshop 6: Likelihood and the Bootstrap Method

This workshop covers material from [Estimators](#chapter:ei): interval estimation ([](#sec:ei_intest)), the Negative Log Likelihood ([](#sec:ei_nll)), and the Bootstrap method ([](#sec:ei_res)).

---

## Task 1: True or False

1. The Likelihood $\mathcal{L}(\theta; x)$ is a probability distribution over the parameter $\theta$.

2. For a dataset of $N$ measurements, the Likelihood is a product of $N$ PDF (or PMF) values, one evaluated at each data point.

3. Taking the logarithm of the Likelihood changes the location of the Maximum Likelihood Estimate $\hat{\theta}_{_{\mathsf{MLE}}}$.

4. The Negative Log Likelihood (NLL) has a **minimum** at the MLE.

5. A 95% Confidence Interval means there is a 95% probability that the true parameter value lies within that specific computed interval.

6. The Bootstrap method requires knowledge of the true underlying distribution of the data.

7. In a bootstrap resample, each value from the original dataset can be selected more than once.

8. For a Gaussian dataset of size $N=100$, `np.argmin(nll_vals)` and `np.argmax(lnL_vals)` will return different indices.

---

## Task 2: Likelihood for the Scale Parameter

The code in [](#sec:ei_nllpy) plots the Likelihood, Log Likelihood, and NLL as functions of a **trial mean** $\mu$, holding $\sigma$ fixed. In this task you will adapt it to vary $\sigma$ instead.

a) Before modifying anything, run the provided `like_loglike_nll` function with the default parameters (`mu_true=5, sigma=1, N=100`). In your own words: what does each of the three curves show, and why do their maxima/minima all occur at the same trial value?

b) Adapt the code to plot the three likelihood curves ($\mathcal{L}$, $\ln\mathcal{L}$, $-\ln\mathcal{L}$) as functions of a **trial value of $\sigma$**, keeping $\mu$ fixed at its true value.

   A few things to think about:
   - What range of trial $\sigma$ values makes sense? (The true value is $\sigma = 1$.)
   - Which argument of `norm.pdf` do you now need to vary?
   - Update the x-axis label and the MLE label accordingly.

c) What value of $\hat{\sigma}_{_{\mathsf{MLE}}}$ do you find? How close is it to the true value $\sigma = 1$?

d) Increase $N$ from 100 to 1000. How does the NLL curve change? Explain briefly why.

---

## Task 3: Bootstrap Uncertainty by Hand

Use the `generate_data` and `LongBootstrap` functions from [](#sec:ei_res).

a) Generate a dataset using `data = generate_data(seed=42)` (default parameters: `mu=6.0, sigma=3.77, N=100`).

b) Compute the sample mean $\overline{x}$ and sample standard deviation $\hat{\sigma}_x$ directly from the data using `numpy`. What values do you get?

c) It can be shown (using the [delta method](#sec:mmb_deltam)) that for a normally distributed dataset, the standard error of the standard deviation is approximately:

   $$\hat{\sigma}(\hat{\sigma}_x) \approx \frac{\hat{\sigma}_x}{\sqrt{2(N-1)}}$$

   Using this result and your values from (b), compute the analytical SEV. Keep a note of this number.

d) Now run `LongBootstrap(data, seed=111)`. How does the bootstrap uncertainty compare to your analytical estimate from (c)?

e) The `LongBootstrap` function returns the list of 9999 bootstrap standard deviations. Modify it (or write additional code after calling it) to compute and print the **68.27% and 95% Confidence Intervals** using `np.percentile`. 

   Hint: the 68.27% CI runs from the 15.865th to the 84.135th percentile of the bootstrap distribution; the 95% CI runs from the 2.5th to the 97.5th.

---

## Task 4: Bootstrap with `scipy`

a) Repeat the bootstrap uncertainty estimate using `ScipyBootstrap(data)` from [](#sec:ei_res). Does the result agree with Task 3?

b) Repeat using `ParaBootstrap(data)` (the Parametric Bootstrap). Then call `PlotBootstraps` to compare all three bootstrap distributions on one plot. Comment on any differences.

c) The `scipy.stats.bootstrap` function accepts a `confidence_level` parameter. Modify `ScipyBootstrap` to also print the **95% Confidence Interval** and compare it to your result from Task 3e.

   You may notice that the scipy CI does not exactly match your percentile-based result from Task 3e, even when using the same seed (and therefore the same bootstrap distribution). Look up the `method` parameter of `scipy.stats.bootstrap`: what method does scipy use by default? How does it differ from a simple percentile interval, and why might it give more accurate coverage?

d) The `PlotBootstraps` function currently uses bins defined by the mean and standard deviation of the first distribution. Is this always appropriate? Suggest an improvement for cases where the three distributions have significantly different spreads.

---
