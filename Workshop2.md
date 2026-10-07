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
# Workshop 2: Summary Statistics

My good friend Alf has become obsessed with weighing packets of rolos (this happens to him from time to time - he will be fine).

This is his set of measurements for the weights of five different packs (in grams):

```
xA = [52.010, 52.041, 52.105, 51.998, 51.981]
```

I decide to help him and make these measurements:
```
xL =[52.05, 52.03, 52.07, 51.90, 51.94]
```

Calculate the mean and uncertainty for each dataset, and the weighted mean and uncertainty of the two datasets. Indicate which of them holds more weight and why that is.

The weighted mean is defined [](#weighted-mean).

The uncertainty on the mean is defined as the standard deviation $\sigma$ divided by the square root of the number of measurements N:

```{math}
:label: eq:sem1
SEM = \dfrac{\sigma}{\sqrt{N}}
```

We will cover the Standard Error on the Mean (SEM) this in [6. Normal Tests](https://dorksquith.github.io/DataAnalysisTechniques/normaltests/) but for now you can use [](#eq:sem1) without understanding it.

