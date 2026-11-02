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

(workshop-2-solutions)=
# Workshop 2 Solutions: Summary Statistics

---

## Task: Alf's Rolo Obsession

### Setup

```{code-cell} python
import numpy as np

xA = np.array([52.010, 52.041, 52.105, 51.998, 51.981])
xL = np.array([52.05, 52.03, 52.07, 51.90, 51.94])

# Use the unbiased standard deviation (ddof=1) because these are small samples.
def get_stats(x):
    mean = np.mean(x)
    sigma = np.std(x, ddof=1)
    N = len(x)
    sem = sigma / np.sqrt(N)
    return mean, sem, N

meanA, semA, NA = get_stats(xA)
meanL, semL, NL = get_stats(xL)

print(f" Alf's data:   mean = {meanA:.3f} +/- {semA:.3f}")
print(f" Lily's data:  mean = {meanL:.3f} +/- {meanL:.3f}")
```

### Weighted mean

The weighted mean is

$$\bar{x}_w = \frac{\sum_i w_i \bar{x}_i}{\sum_i w_i}, \qquad w_i = \frac{1}{V[\bar{x}_i]} = \frac{N_i}{v_{x_i}},$$

with uncertainty

$$\sigma_{\bar{x}_w} = \sqrt{\frac{1}{\sum_i w_i}}.$$

```{code-cell} python
xAL = np.stack((xA, xL))

def weighted_mean(data2D):
    # axis=1: each dataset is a row
    m  = np.mean(data2D, axis=1)
    v  = np.var(data2D,  axis=1, ddof=1)
    N  = np.array([len(data2D[0]), len(data2D[1])])
    vm = v / N          # variances on the individual means
    w  = 1 / vm         # weights
    wm  = np.sum(m * w) / np.sum(w)
    sem = np.sqrt(1 / np.sum(w))
    return wm, sem

wm, wsem = weighted_mean(xAL)
print(f" Weighted mean: {wm:.3f} +/- {wsem:.3f}")
```

### Which dataset holds more weight, and why?

Alf's data have a smaller sample variance ($v_{x_A} < v_{x_L}$), which means a smaller SEM for the same number of measurements.
Because the weight of each dataset is the reciprocal of the variance on its mean, Alf's measurements receive the larger weight.

This is not because Alf recorded an extra decimal place — a higher number of significant figures in the raw data does not automatically reduce the spread.
The weights reflect *precision* (how tightly the data cluster around their mean), not *resolution* (how finely the measurements are reported).
