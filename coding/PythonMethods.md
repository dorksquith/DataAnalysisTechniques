---
kernelspec:
  name: python3
  display_name: 'Python 3'
numbering:
  title: false
  headings: true
  equation: true
---

(chapter:python-methods)=
# Python Methods Reference

A reference table of all Python methods introduced across the module notes chapters.

```{note}
Plotting methods (`matplotlib.pyplot` and `seaborn`) are documented separately in [Plotting Tips and Tricks](#chapter:plotting-tips).
```

**Chapter key:** [**T&C**](#chapter:tandc) = TerminologyAndConcepts · [**PE**](#chapter:pe) = ProbabilityEssentials · [**DD**](#chapter:dd) = DescribingData · [**N**](#chapter:norm) = Normal · [**HT**](#chapter:hyp) = HypothesisTests · [**NT**](#chapter:normt) = NormalTests · [**EI**](#chapter:ei) = EstimatorsIntro · [**PMF**](#chapter:pmf) = PMFs · [**PDF**](#chapter:pdf) = PDFs · [**ML**](#chapter:mle) = MaxLike · [**LS**](#chapter:lse) = LeastSquares · [**BI**](#chapter:bay) = BayesianInference · [**MCMC**](#chapter:mcmc) = MCMC

---

## `numpy`

| Method | Description | Chapter(s) |
|--------|-------------|------------|
| `np.sum(x)` | Sum of array elements | [T&C](#chapter:tandc), [DD](#chapter:dd), [EI](#chapter:ei), [ML](#chapter:mle), [LS](#chapter:lse), [BI](#chapter:bay), [MCMC](#chapter:mcmc) |
| `np.min(x)` / `np.max(x)` | Min / max value | [T&C](#chapter:tandc) |
| `np.mean(x)` | Arithmetic mean | [T&C](#chapter:tandc), [DD](#chapter:dd), [NT](#chapter:normt), [EI](#chapter:ei), [LS](#chapter:lse), [BI](#chapter:bay), [MCMC](#chapter:mcmc) |
| `np.median(x)` | Median | [T&C](#chapter:tandc) |
| `np.var(x)` / `np.var(x, ddof=1)` | Variance (biased / unbiased) | [T&C](#chapter:tandc), [NT](#chapter:normt), [MCMC](#chapter:mcmc) |
| `np.std(x)` / `np.std(x, ddof=1)` | Standard deviation (biased / unbiased) | [T&C](#chapter:tandc), [NT](#chapter:normt), [EI](#chapter:ei) |
| `np.sqrt(x)` | Square root | [T&C](#chapter:tandc), [NT](#chapter:normt), [ML](#chapter:mle), [LS](#chapter:lse), [BI](#chapter:bay) |
| `np.exp(x)` / `np.log(x)` | Exponential $e^x$ / natural logarithm | [EI](#chapter:ei), [PMF](#chapter:pmf), [ML](#chapter:mle), [MCMC](#chapter:mcmc) |
| `np.prod(x)` | Product of array elements | [EI](#chapter:ei) |
| `np.round(x, n)` | Round to n decimal places | [BI](#chapter:bay) |
| `np.ceil(x)` | Round up to the next whole number | [MCMC](#chapter:mcmc) |
| `np.percentile(data, q)` | q-th percentile | [T&C](#chapter:tandc), [MCMC](#chapter:mcmc) |
| `np.quantile(data, q)` | q-th quantile (0–1 scale) | [T&C](#chapter:tandc) |
| `np.linspace(start, stop, n)` | n evenly-spaced values | [T&C](#chapter:tandc), [PE](#chapter:pe), [N](#chapter:norm), [EI](#chapter:ei), [ML](#chapter:mle), [LS](#chapter:lse), [BI](#chapter:bay), [MCMC](#chapter:mcmc) |
| `np.arange(start, stop, step)` | Evenly-spaced values with a given step | [PE](#chapter:pe), [MCMC](#chapter:mcmc) |
| `np.array(x)` | Create NumPy array from list | [DD](#chapter:dd), [ML](#chapter:mle), [LS](#chapter:lse), [MCMC](#chapter:mcmc) |
| `np.stack((a, b))` | Stack arrays into 2D array | [DD](#chapter:dd) |
| `np.cov(xy)` | Covariance matrix | [DD](#chapter:dd) |
| `np.corrcoef(xy)` | Pearson correlation matrix | [DD](#chapter:dd) |
| `np.diagonal(A)` | Diagonal of a matrix (e.g. the variances in a covariance matrix) | [LS](#chapter:lse) |
| `np.sort(x)` | Sorted copy of an array | [MCMC](#chapter:mcmc) |
| `np.argsort(x)` | Indices that would sort an array | [BI](#chapter:bay) |
| `np.argmin(x)` | Index of the smallest value | [EI](#chapter:ei), [MCMC](#chapter:mcmc) |
| `np.cumsum(x)` | Cumulative sum | [BI](#chapter:bay) |
| `np.diff(x)` | Differences between consecutive elements | [BI](#chapter:bay) |
| `np.searchsorted(a, v)` | Index where v would be inserted to keep a sorted | [BI](#chapter:bay) |
| `np.flatnonzero(x)` | Indices of the non-zero (`True`) elements | [BI](#chapter:bay) |
| `np.allclose(a, b)` | Check two arrays are equal within a tolerance | [EI](#chapter:ei), [PMF](#chapter:pmf) |
| `np.inf` / `np.e` | Floating-point infinity / Euler's number | [DD](#chapter:dd), [N](#chapter:norm), [HT](#chapter:hyp), [ML](#chapter:mle) |
| `np.random.default_rng(seed)` | Create random number generator (PCG64) | [T&C](#chapter:tandc), [DD](#chapter:dd), [EI](#chapter:ei), [LS](#chapter:lse), [MCMC](#chapter:mcmc) |
| `np.random.seed(n)` | Set global random seed (legacy) | [NT](#chapter:normt), [MCMC](#chapter:mcmc) |
| `np.random.choice(data)` / `np.random.rand()` | Random sample from an array / uniform random number in [0, 1) (legacy) | [NT](#chapter:normt), [MCMC](#chapter:mcmc) |
| `rng.uniform(low, high, size)` | Uniform random variates | [T&C](#chapter:tandc) |
| `rng.normal(loc, scale, size)` | Normal random variates | [T&C](#chapter:tandc), [DD](#chapter:dd) |
| `rng.standard_normal(size)` | Standard Normal random variates | [LS](#chapter:lse) |
| `rng.integers(low, high, size)` | Random integers | [LS](#chapter:lse) |
| `rng.poisson(lam, size)` | Poisson random variates | [LS](#chapter:lse), [MCMC](#chapter:mcmc) |
| `rng.choice(data, replace, size)` | Random sample from array (bootstrap resampling) | [EI](#chapter:ei) |

---

## `numpy.linalg`

| Function | Description | Chapter(s) |
|--------|-------------|------------|
| `np.linalg.inv(A)` | Matrix inverse $A^{-1}$; used to obtain the covariance matrix from the Hessian | [LS](#chapter:lse) |

---

## `scipy.stats` — Distributions

| Object / Method | Description | Chapter(s) |
|--------|-------------|------------|
| `uniform(loc, scale)` | Uniform distribution | [T&C](#chapter:tandc), [DD](#chapter:dd), [PDF](#chapter:pdf) |
| `norm(loc, scale)` | Normal distribution | [T&C](#chapter:tandc), [PE](#chapter:pe), [N](#chapter:norm), [HT](#chapter:hyp), [NT](#chapter:normt), [EI](#chapter:ei), [PDF](#chapter:pdf), [BI](#chapter:bay), [MCMC](#chapter:mcmc) |
| `randint(low, high)` | Discrete uniform distribution | [PE](#chapter:pe) |
| `bernoulli(p)` | Bernoulli distribution | [PMF](#chapter:pmf), [BI](#chapter:bay) |
| `binom(n, p)` | Binomial distribution | [PMF](#chapter:pmf), [ML](#chapter:mle), [BI](#chapter:bay) |
| `poisson(mu)` | Poisson distribution | [PMF](#chapter:pmf), [BI](#chapter:bay), [MCMC](#chapter:mcmc) |
| `expon(scale=1/lam)` | Exponential distribution | [PDF](#chapter:pdf), [BI](#chapter:bay) |
| `chi2(df)` | Chi-squared distribution | [PDF](#chapter:pdf), [ML](#chapter:mle) |
| `gamma(a, scale)` | Gamma distribution | [BI](#chapter:bay), [MCMC](#chapter:mcmc) |
| `beta(a, b)` | Beta distribution | [BI](#chapter:bay) |
| `nbinom(n, p)` | Negative Binomial distribution | [MCMC](#chapter:mcmc) |
| `t(df)` / `f(dfn, dfd)` | Student's t / F distribution | [NT](#chapter:normt) |
| `multivariate_normal(mu, cov)` | Multivariate Normal distribution | [N](#chapter:norm) |
| `rv_continuous` | Base class for defining custom distributions | [PE](#chapter:pe) |
| `.rvs(size)` | Draw random variates from a distribution | [T&C](#chapter:tandc), [DD](#chapter:dd), [N](#chapter:norm), [NT](#chapter:normt), [EI](#chapter:ei), [BI](#chapter:bay), [MCMC](#chapter:mcmc) |
| `.pdf(x)` | Probability density at x | [PE](#chapter:pe), [N](#chapter:norm), [HT](#chapter:hyp), [EI](#chapter:ei), [BI](#chapter:bay), [MCMC](#chapter:mcmc) |
| `.pmf(k)` | Probability mass at k (discrete distributions) | [PE](#chapter:pe), [PMF](#chapter:pmf), [MCMC](#chapter:mcmc) |
| `.logpmf(k)` | Natural log of the PMF (for log-likelihoods) | [MCMC](#chapter:mcmc) |
| `.cdf(x)` | Cumulative probability P(X ≤ x) | [PE](#chapter:pe), [N](#chapter:norm), [HT](#chapter:hyp), [NT](#chapter:normt), [EI](#chapter:ei), [PMF](#chapter:pmf), [PDF](#chapter:pdf) |
| `.sf(x)` | Survival function P(X > x) — used as p-value | [HT](#chapter:hyp), [NT](#chapter:normt), [ML](#chapter:mle) |
| `.ppf(q)` | Inverse CDF: value x for which P(X ≤ x) = q | [HT](#chapter:hyp), [PDF](#chapter:pdf), [BI](#chapter:bay) |
| `.isf(q)` | Inverse survival function: value x for which P(X > x) = q | [HT](#chapter:hyp) |
| `.interval(confidence)` | Equal-tailed interval containing the given probability | [BI](#chapter:bay) |
| `.stats(moments='mv')` | Mean and variance (`'mvsk'` adds skew and kurtosis) | [PE](#chapter:pe), [PMF](#chapter:pmf), [PDF](#chapter:pdf) |
| `.mean()` / `.var()` | Mean / variance of the distribution | [MCMC](#chapter:mcmc) |
| `.support()` | Lower and upper bounds of distribution | [PE](#chapter:pe), [DD](#chapter:dd), [PMF](#chapter:pmf), [PDF](#chapter:pdf) |

---

## `scipy.stats` — Statistical Functions

| Method | Description | Chapter(s) |
|--------|-------------|------------|
| `mode(x)` | Mode of data | [T&C](#chapter:tandc) |
| `pearsonr(x, y)` | Pearson linear correlation coefficient | [DD](#chapter:dd) |
| `spearmanr(x, y)` | Spearman rank correlation coefficient | [DD](#chapter:dd) |
| `ttest_1samp(data, popmean)` | One-sample t-test | [NT](#chapter:normt) |
| `ttest_ind(a, b)` | Two-sample independent t-test | [NT](#chapter:normt) |
| `f_oneway(...)` | One-way ANOVA F-test | [NT](#chapter:normt) |
| `bootstrap((data,), statistic)` | Bootstrap confidence intervals | [EI](#chapter:ei) |
| `fit(dist, data, bounds)` | Fit a distribution's parameters to data by maximum likelihood | [ML](#chapter:mle) |

---

## `scipy.optimize`

| Function | Description | Chapter(s) |
|--------|-------------|------------|
| `least_squares(fun, x0, args, loss, f_scale)` | Fit by minimising a sum of squares; supports robust losses (`'linear'`, `'soft_l1'`, `'huber'`, `'cauchy'`); returns Jacobian for covariance estimation | [LS](#chapter:lse) |
| `minimize(fun, x0, args, method, bounds)` | Minimise an arbitrary scalar function; use `method='BFGS'` to obtain `result.hess_inv` (approximate inverse Hessian / parameter covariance) | [ML](#chapter:mle) |
| `minimize_scalar(fun, args, bounds, method)` | Minimise a scalar function of **one** variable only | [ML](#chapter:mle), [BI](#chapter:bay) |
| `fsolve(func, x0, args)` | Find roots of a system $f(x)=0$ (e.g. score equations); returns array of root values | [ML](#chapter:mle) |

**Key attributes of `scipy.optimize` results:**

| Attribute | Description |
|-----------|-------------|
| `result.x` | Parameter estimates at the minimum |
| `result.fun` | Value of the function at the minimum (for `least_squares`: the residual vector) |
| `result.jac` | Jacobian matrix of residuals (`least_squares`; use `J.T @ J * (S/ν)` for covariance) |

---

## `scipy.special`

| Function | Description | Chapter(s) |
|--------|-------------|------------|
| `binom(n, k)` / `comb(n, k)` | Binomial coefficient $\binom{n}{k}$ | [PMF](#chapter:pmf), [ML](#chapter:mle) |
| `factorial(n)` | Factorial $n!$ | [PMF](#chapter:pmf) |
| `gamma(x)` | Gamma function $\Gamma(x)$ | [PMF](#chapter:pmf) |
| `gammaln(x)` | Natural log of the Gamma function $\ln\Gamma(x)$; use `gammaln(k+1)` in place of $\ln(k!)$ to avoid overflow for large $k$ | [PMF](#chapter:pmf) |

---

## `pandas`

| Method | Description | Chapter(s) |
|--------|-------------|------------|
| `pd.DataFrame(data, columns)` | Create DataFrame from array or dict | [N](#chapter:norm) |
| `df.iloc[...]` | Integer-location based indexing | [N](#chapter:norm) |
| `df.mean()` / `df.max()` / `df.std()` | Summary statistics on DataFrame columns | [N](#chapter:norm) |

---

## `sympy`

| Method | Description | Chapter(s) |
|--------|-------------|------------|
| `sympy.Symbol(name)` / `symbols('x y')` | Create symbolic variable(s) | [PE](#chapter:pe), [DD](#chapter:dd), [ML](#chapter:mle) |
| `sympy.init_printing()` | Enable pretty-printing of expressions | [PE](#chapter:pe), [DD](#chapter:dd) |
| `sympy.integrate(expr, (x, a, b))` | Symbolic (definite or indefinite) integration | [PE](#chapter:pe), [DD](#chapter:dd) |
| `sympy.exp(x)` / `sympy.log(x)` | Symbolic exponential / natural logarithm | [PE](#chapter:pe), [DD](#chapter:dd), [ML](#chapter:mle) |
| `diff(expr, x)` | Symbolic derivative | [ML](#chapter:mle) |
| `solve(expr, x)` | Solve an equation symbolically | [ML](#chapter:mle) |
| `expand_log(expr)` / `cancel(expr)` | Simplify: expand logarithms / cancel common factors | [ML](#chapter:mle) |
| `lambdify(x, expr)` | Turn a symbolic expression into a Python function | [ML](#chapter:mle) |

---

## `statsmodels`

| Function | Description | Chapter(s) |
|--------|-------------|------------|
| `plot_acf(x, lags)` | Plot the autocorrelation function of a chain or time series (`statsmodels.graphics.tsaplots`) | [MCMC](#chapter:mcmc) |

---

## `pymc` and `arviz`

| Method | Description | Chapter(s) |
|--------|-------------|------------|
| `pm.Model()` | Container for a probabilistic model (used with `with`) | [MCMC](#chapter:mcmc) |
| `pm.Gamma(...)` / `pm.Poisson(..., observed=data)` | Prior and likelihood distributions; `observed` attaches the data | [MCMC](#chapter:mcmc) |
| `pm.sample(draws, tune, chains)` | Draw posterior samples (NUTS by default) | [MCMC](#chapter:mcmc) |
| `idata.posterior["mu"]` | Posterior samples of a parameter, by chain and draw | [MCMC](#chapter:mcmc) |
| `az.summary(idata, ci_prob, ci_kind)` | Summary table: mean, sd, HDI, effective sample size, $\hat{R}$ | [MCMC](#chapter:mcmc) |

---

## Python built-ins and standard library

| Method | Description | Chapter(s) |
|--------|-------------|------------|
| `set()` / `range()` | Create a set / integer range | [PE](#chapter:pe) |
| `{... for ... in ...}` | Set / list comprehension | [PE](#chapter:pe) |
| `A.issubset(S)` / `A <= S` | Test whether A ⊆ S | [PE](#chapter:pe) |
| `A \| B` / `A & B` | Set union / intersection | [PE](#chapter:pe) |
| `len(x)` | Number of elements in sequence or array | [T&C](#chapter:tandc), [PE](#chapter:pe) |
| `abs(x)` | Absolute value | [HT](#chapter:hyp), [NT](#chapter:normt) |
| `range(n)` | Iterate n times (used in bootstrap and MCMC loops) | [EI](#chapter:ei), [MCMC](#chapter:mcmc) |
| `math.factorial(n)` | Factorial $n!$ | [PMF](#chapter:pmf) |
| `timeit(stmt, number)` | Time how long a piece of code takes (`timeit` module) | [PMF](#chapter:pmf) |
| `sys.float_info.min` | Smallest representable float (motivates log-likelihood) | [EI](#chapter:ei) |
