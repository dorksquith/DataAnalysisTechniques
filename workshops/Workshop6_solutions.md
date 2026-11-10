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

(workshop-6-solutions)=
# Workshop 6 Solutions: Likelihood & Bootstrapping

---

## Task 1: True or False

1. **False.** The Likelihood is a function of the parameter $\theta$ for a fixed dataset, not a probability distribution. It is not normalised and does not integrate to 1 over $\theta$.

2. **True.** $\mathcal{L}(\theta; x) = \prod_{i}^{N} f_X(x_i; \theta)$, a product of $N$ density (or probability) values.

3. **False.** The logarithm is a strictly monotonic (order-preserving) transformation. If $\mathcal{L}$ is maximised at some $\hat{\theta}$, then $\ln\mathcal{L}$ is also maximised at exactly the same $\hat{\theta}$.

4. **True.** The NLL is $-\ln\mathcal{L}$. Flipping the sign turns the maximum of $\ln\mathcal{L}$ into a minimum.

5. **False.** This is the most common misstatement of the frequentist CI. Once computed, the interval either contains the true value or it does not — there is no probability remaining. The correct statement is: *if we repeated the experiment many times and computed a CI each time, 95% of those intervals would contain the true value.*

6. **False.** The (non-parametric) Bootstrap is explicitly *nonparametric*: it makes no assumption about the true distribution and resamples directly from the data.

7. **True.** Sampling **with replacement** is essential. Without it, every resample would simply be a permutation of the original dataset, giving zero variance and a useless uncertainty estimate.

8. **False.** `np.argmin(nll_vals)` and `np.argmax(lnL_vals)` return the **same** index. The NLL is $-\ln\mathcal{L}$, so minimising the NLL is identical to maximising $\ln\mathcal{L}$.

---

## Task 2: Likelihood for the Scale Parameter

### a) Interpreting the three curves

All three curves — $\mathcal{L}(\mu)$, $\ln\mathcal{L}(\mu)$, and $-\ln\mathcal{L}(\mu)$ — are evaluated for the same dataset and the same set of trial $\mu$ values. Because $\ln$ is monotonic, the maximum of $\mathcal{L}$ and $\ln\mathcal{L}$ occur at the same trial value. Negating $\ln\mathcal{L}$ turns that maximum into a minimum, so all three agree on the MLE. The NLL is preferred in practice because minimising a positive quantity is numerically convenient, and standard `scipy` optimisers minimise by default.

### b) Adapting the code for $\sigma$

The key changes are: (i) introduce a `sigma_trial` array instead of `mu_trial`; (ii) fix `mu` and vary `sigma` in `norm.pdf`; (iii) update labels.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu_true    = 5    # fixed true mean
sigma_true = 1    # true sigma (what we are now estimating)
N          = 100

# range of trial sigma values
sigma_trial = np.linspace(0.5, 1.5, 200)

# reproducible data
data = norm.rvs(loc=mu_true, scale=sigma_true, size=N, random_state=42)

# likelihood functions — mu is now fixed; sigma varies
def likelihood(data, sigma):
    return np.prod(norm.pdf(data, mu_true, sigma))

def log_likelihood(data, sigma):
    return np.sum(np.log(norm.pdf(data, mu_true, sigma)))

L_vals   = [likelihood(data, s) for s in sigma_trial]
lnL_vals = [log_likelihood(data, s) for s in sigma_trial]
nll_vals = [-lnL for lnL in lnL_vals]

sigma_mle = sigma_trial[np.argmin(nll_vals)]
print(f"MLE estimate: sigma = {sigma_mle:.3f}  (true value: sigma = {sigma_true})")

plots = [
    (L_vals,   r'$\mathcal{L}(\sigma)$',       "Likelihood"),
    (lnL_vals, r'$\ln\,\mathcal{L}(\sigma)$',  "Log likelihood"),
    (nll_vals, r'$-\ln\,\mathcal{L}(\sigma)$', "Negative log likelihood"),
]

mle_label = r'$\hat{\sigma}$' + f' = {sigma_mle:.3f}'

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (values, curve_label, ylabel) in zip(axes, plots):
    ax.plot(sigma_trial, values, 'r--', label=curve_label)
    ax.axvline(sigma_mle, color='black', linestyle=':', linewidth=1, label=mle_label)
    ax.set_xlabel(r'Trial value of $\sigma$', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.show()
```

### c) MLE for $\sigma$

With `random_state=42` and `N=100`, the MLE should be close to but not exactly 1.0 (it is a sample-based estimate, so there is statistical scatter). The MLE for $\sigma$ from a Gaussian dataset is the sample standard deviation $\hat{\sigma} = \sqrt{\frac{1}{N}\sum_i (x_i - \overline{x})^2}$ (note: MLE uses $1/N$, not $1/(N-1)$).

### d) Effect of increasing $N$

As $N$ increases, the NLL curve becomes **narrower and deeper**. More data constrains the parameter more tightly, so the range of $\sigma$ values that are nearly as likely as the MLE shrinks. This is equivalent to saying the uncertainty on $\hat{\sigma}_{_{\mathsf{MLE}}}$ decreases with $N$.

---

## Task 3: Bootstrap Uncertainty by Hand

### a) Generate data

```{code-cell} python
import numpy as np

def generate_data(mu=6.0, sigma=3.77, N=100, seed=None):
    if seed:
        print(f"Seed set: {seed}")
    rng = np.random.default_rng(seed)
    from scipy.stats import norm
    return norm.rvs(loc=mu, scale=sigma, size=N, random_state=rng)

data = generate_data(seed=42)
```

### b) Sample statistics

```{code-cell} python
xbar   = np.mean(data)
sigma_x = np.std(data)
print(f"Sample mean:  {xbar:.3f}")
print(f"Sample sigma: {sigma_x:.3f}")
```

The true parameters are $\mu = 6.0$ and $\sigma = 3.77$; the sample values will be close but not identical.

### c) Analytical SEV for the standard deviation

Using the quoted result (derived via the delta method in [](#sec:mmb_deltam)):

$$\hat{\sigma}(\hat{\sigma}_x) \approx \frac{\hat{\sigma}_x}{\sqrt{2(N-1)}}$$

```{code-cell} python
N = len(data)
sev_sigma = sigma_x / np.sqrt(2 * (N - 1))
print(f"Analytical SEV on sigma: {sev_sigma:.3f}")
```

With $\hat{\sigma}_x \approx 2.9$ and $N=100$ this gives roughly $0.21$.

### d) Bootstrap uncertainty

```{code-cell} python
def LongBootstrap(data, seed=None):
    import numpy as np
    rng = np.random.default_rng(seed)
    print(f"sample sigma: {np.std(data):.3f}")
    sigmas = []
    for i in range(9999):
        rdata = rng.choice(data, replace=True, size=100)
        sigmas.append(np.std(rdata))
    print(f"bootstrap estimate for sigma: {np.mean(sigmas):.3f} +/- {np.std(sigmas):.3f}")
    return sigmas

byhand_sigmas = LongBootstrap(data, seed=111)
```

The bootstrap uncertainty should agree closely with the analytical SEV (~0.19–0.21). Both estimate the same quantity — the standard deviation of the estimator $\hat{\sigma}_x$ — by different routes.

### e) Confidence Intervals via percentiles

```{code-cell} python
sigmas = byhand_sigmas  # from above

# 68.27% CI
lo_68 = np.percentile(sigmas, (100 - 68.27) / 2)       # 15.865th percentile
hi_68 = np.percentile(sigmas, 100 - (100 - 68.27) / 2) # 84.135th percentile
print(f"68.27% CI: [{lo_68:.3f}, {hi_68:.3f}]")

# 95% CI
lo_95 = np.percentile(sigmas, 2.5)
hi_95 = np.percentile(sigmas, 97.5)
print(f"95% CI:    [{lo_95:.3f}, {hi_95:.3f}]")
```

The 95% CI is wider than the 68.27% CI — it corresponds to roughly $\pm 2\sigma$ rather than $\pm 1\sigma$.

---

## Task 4: Bootstrap with `scipy`

### a) `scipy` bootstrap

```{code-cell} python
def ScipyBootstrap(data, confidence_level=0.6827, seed=None):
    import numpy as np
    from scipy.stats import bootstrap
    rng = np.random.default_rng(seed)
    sbs = bootstrap((data,), np.std, rng=rng, confidence_level=confidence_level)
    sp_mean = np.mean(sbs.bootstrap_distribution)
    sp_ser  = sbs.standard_error
    sp_ci_lo, sp_ci_hi = sbs.confidence_interval
    print(f"scipy bootstrap: {sp_mean:.3f} +/- {sp_ser:.3f}")
    print(f"{confidence_level*100:.2f}% CI: [{sp_ci_lo:.3f}, {sp_ci_hi:.3f}]")
    return sbs.bootstrap_distribution

scipy_sigmas = ScipyBootstrap(data)
```

The scipy result should agree with the by-hand bootstrap to within the statistical noise of the resampling.

### b) Parametric bootstrap and comparison plot

```{code-cell} python
def ParaBootstrap(data, seed=None):
    import numpy as np
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    psigmas = []
    for i in range(9999):
        respin = norm.rvs(loc=np.mean(data), scale=np.std(data, ddof=1), size=100, random_state=rng)
        psigmas.append(np.std(respin))
    print(f"parametric bootstrap: {np.mean(psigmas):.3f} +/- {np.std(psigmas):.3f}")
    return psigmas

parametric_sigmas = ParaBootstrap(data)
```

```{code-cell} python
def PlotBootstraps(a, b, c):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sm = np.mean(a)
    ss = np.std(a)
    bins = np.linspace(sm - 3*ss, sm + 3*ss, 30)
    labs    = ['byhand', 'scipy', 'parametric']
    cols    = ['darkgreen', 'dodgerblue', 'grey']
    styles  = ['-', '--', ':']
    fills   = ['darkgreen', 'none', 'none']
    alphas  = [0.2, 1, 1]
    for i, d in enumerate([a, b, c]):
        ax.hist(d, fc=fills[i], alpha=alphas[i], lw=2, ls=styles[i],
                ec=cols[i], bins=bins,
                label=f'{labs[i]} {np.mean(d):.3f}+/-{np.std(d):.3f}')
    ax.set_xlabel(r"Bootstrap $\sigma$")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.2)
    plt.legend()
    plt.show()

PlotBootstraps(byhand_sigmas, scipy_sigmas, parametric_sigmas)
```

All three distributions should be centred near the same value and have similar widths. Small differences arise because the non-parametric methods resample from the data directly, while the parametric method draws from a fitted Gaussian — the two strategies agree only in the limit of large $N$ and a well-specified model.

### c) 95% CI via `scipy` and the BCa method

Pass `confidence_level=0.95` to `ScipyBootstrap`:

```{code-cell} python
scipy_sigmas_95 = ScipyBootstrap(data, confidence_level=0.95)
```

The scipy result will be close to, but not identical to, the percentile-based interval from Task 3e — even with the same seed producing the same bootstrap distribution. The reason is the CI **computation method**.

`scipy.stats.bootstrap` uses the **BCa method** (bias-corrected and accelerated) by default. It applies two corrections to the raw percentile interval:

- **Bias correction**: if the median of the bootstrap distribution does not sit exactly on the observed sample statistic, the interval is shifted to account for this.
- **Acceleration correction**: if the bootstrap distribution is skewed (as it is for $\hat{\sigma}$, since $\sigma$ cannot be negative), the interval is stretched asymmetrically to reflect that skewness.

The simple percentile method (Task 3e) just reads off the 15.865th and 84.135th quantiles of the bootstrap distribution and treats them as symmetric bounds. BCa adjusts those quantiles to give better **coverage** — meaning that in repeated experiments, the true value is captured closer to the nominal 68.27% or 95% of the time, especially for small $N$ or skewed estimators.

You can switch scipy to the plain percentile method with `method='percentile'`, which will then agree more closely with Task 3e.

### d) Improving `PlotBootstraps` for different spreads

The current implementation centres bins on the **first** distribution (`byhand`). If the three distributions have different means or spreads, some distributions may be clipped or compressed. A simple improvement is to define bins over the combined range of all three:

```{code-cell} python
import numpy as np

def PlotBootstrapsImproved(a, b, c):
    import matplotlib.pyplot as plt
    all_vals = np.concatenate([a, b, c])
    bins = np.linspace(np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5), 40)
    # ... rest of plotting code as before, using the new bins
```

Using the 0.5th–99.5th percentile range (rather than min/max) avoids extreme outliers from blowing out the x-axis.

---
