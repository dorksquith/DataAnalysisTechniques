---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

(chapter:python-methods)=
# Python Methods Reference

A reference table of all Python methods introduced across the module notes chapters.

```{note}
Plotting methods (`matplotlib.pyplot` and `seaborn`) are documented separately in [Plotting Tips and Tricks](#chapter:plotting-tips).
```

**Chapter key:** **T&C** = TerminologyAndConcepts · **PE** = ProbabilityEssentials · **DD** = DescribingData · **N** = Normal · **HT** = HypothesisTests · **NT** = NormalTests · **EI** = EstimatorsIntro

---

## `numpy`

| Method | Description | Chapter(s) |
|--------|-------------|------------|
| `np.sum(x)` | Sum of array elements | T&C |
| `np.min(x)` / `np.max(x)` | Min / max value | T&C |
| `np.mean(x)` | Arithmetic mean | T&C, NT, EI |
| `np.median(x)` | Median | T&C |
| `np.var(x)` / `np.var(x, ddof=1)` | Variance (biased / unbiased) | T&C, NT |
| `np.std(x)` / `np.std(x, ddof=1)` | Standard deviation (biased / unbiased) | T&C, NT, EI |
| `np.sqrt(x)` | Square root | T&C, NT |
| `np.percentile(data, q)` | q-th percentile | T&C |
| `np.quantile(data, q)` | q-th quantile (0–1 scale) | T&C |
| `np.linspace(start, stop, n)` | n evenly-spaced values | T&C, N, EI |
| `np.arange(start, stop)` | Integer sequence as array | PE |
| `np.array(x)` | Create NumPy array from list | DD |
| `np.stack((a, b))` | Stack arrays into 2D array | DD |
| `np.cov(xy)` | Covariance matrix | DD |
| `np.corrcoef(xy)` | Pearson correlation matrix | DD |
| `np.inf` | Floating-point infinity | HT |
| `np.random.default_rng(seed)` | Create random number generator (PCG64) | T&C, EI |
| `np.random.seed(n)` | Set global random seed (legacy) | NT |
| `rng.uniform(low, high, size)` | Uniform random variates | T&C |
| `rng.normal(loc, scale, size)` | Normal random variates | T&C |
| `rng.choice(data, replace, size)` | Random sample from array (bootstrap resampling) | EI |

---

## `scipy.stats` — Distributions

| Object / Method | Description | Chapter(s) |
|-----------------|-------------|------------|
| `uniform(loc, scale)` | Uniform distribution object | T&C, DD |
| `norm(loc, scale)` | Normal distribution object | T&C, N, HT, NT, EI |
| `randint(low, high)` | Discrete uniform distribution | PE |
| `t` | Student's t-distribution | NT |
| `f` | F-distribution | NT |
| `multivariate_normal(mu, cov)` | Multivariate normal distribution | N |
| `rv_continuous` | Base class for defining custom distributions | PE |
| `.rvs(size)` | Draw random variates from a distribution | T&C, DD, EI |
| `.pdf(x)` | Probability density at x | PE, N, HT |
| `.pmf(k)` | Probability mass at k (discrete distributions) | PE |
| `.cdf(x)` | Cumulative probability P(X ≤ x) | PE, N, HT, NT |
| `.sf(x)` | Survival function P(X > x) — used as p-value | HT, NT |
| `.ppf(q)` | Inverse CDF: value x for which P(X ≤ x) = q | HT |
| `.support()` | Lower and upper bounds of distribution | PE, DD |

---

## `scipy.stats` — Statistical Functions

| Method | Description | Chapter(s) |
|--------|-------------|------------|
| `mode(x)` | Mode of data | T&C |
| `pearsonr(x, y)` | Pearson linear correlation coefficient | DD |
| `spearmanr(x, y)` | Spearman rank correlation coefficient | DD |
| `ttest_1samp(data, popmean)` | One-sample t-test | NT |
| `ttest_ind(a, b)` | Two-sample independent t-test | NT |
| `f_oneway(...)` | One-way ANOVA F-test | NT |
| `bootstrap((data,), statistic)` | Bootstrap confidence intervals | EI |

---

## `pandas`

| Method | Description | Chapter(s) |
|--------|-------------|------------|
| `pd.DataFrame(data, columns)` | Create DataFrame from array or dict | N |
| `df.iloc[...]` | Integer-location based indexing | N |
| `df.mean()` / `df.max()` / `df.std()` | Summary statistics on DataFrame columns | N |

---

## `sympy`

| Method | Description | Chapter(s) |
|--------|-------------|------------|
| `sympy.Symbol(name)` | Create a symbolic variable | PE, DD |
| `sympy.init_printing()` | Enable pretty-printing of expressions | PE, DD |
| `sympy.integrate(expr, (x, a, b))` | Symbolic (definite or indefinite) integration | PE, DD |
| `sympy.exp(x)` | Symbolic exponential e^x | PE |

---

## Python built-ins

| Method | Description | Chapter(s) |
|--------|-------------|------------|
| `set()` / `range()` | Create a set / integer range | PE |
| `{... for ... in ...}` | Set / list comprehension | PE |
| `A.issubset(S)` / `A <= S` | Test whether A ⊆ S | PE |
| `A \| B` / `A & B` | Set union / intersection | PE |
| `len(x)` | Number of elements in sequence or array | T&C, PE |
| `abs(x)` | Absolute value | HT, NT |
| `range(n)` | Iterate n times (used in bootstrap loops) | EI |
| `sys.float_info.min` | Smallest representable float (motivates log-likelihood) | EI |
