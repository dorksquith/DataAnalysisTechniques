---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Terminology & Concepts


<!--Make sure you are familiar with these before embarking on the topics in this module.-->

## Hypotheses, Models, and Theories

A **Hypothesis** is an educated guess at the outcome of a specific event. 


> The **Null Hypothesis**, usually denoted $H_0$, is one that assumes there is no relationship between two variables, and the **Alternative Hypothesis**, usually denoted $H_1$, is one that assumes there is. 

For example:
- $H_0$: There is no correlation between how long we sleep and how long we live.
- $H_1$: People who sleep well live longer.




A **Model** is an object, a computer program, or a mathematical equation that can explain how something works, and/or be used to make predictions. 


:::{figure} /figures/models
:label: fig:models

A Scale Model of a Delorian, a Computer Model of Earth's temperature, and the Standard Model of particle physics.
:::


For example:
- a Scale Model of a delorian (a toy car based on time-travelling car in back to the future movies, which captures the visual appearance of the machine)  
- a Climate Model (a complex computer program that tells us how much trouble we are in existentially) 
- the Standard Model of particle physics (a set of mathematical equations describing how the universe works at a fundamental level)


> When statisticians talk about Toy Models (often shortened to "Toys"), they are referring to a mathematical model that has been simplified. 


A **Theory** is the description of a set of models, and the relationships between them. It is complete, and is a candidate explanation for the "Underlying Truth". The Underlying Truth is an abstract concept which cannot be known with certainty. We can test our models and theories, and those which we fail to prove wrong tend to grow in esteem. 

> It is not possible to prove that a hypotheses or theory is correct. We can only endeavour to prove them wrong. 


A theory that passes all the tests accessible by our imaginations and technological capability may well be wrong - we just haven't asked the right questions or gathered enough data to see it.

## Population and Sample

To test our models we must gather data as evidence. How much data do we need? No amount is ever enough!

:::{figure} /figures/HeartData.jpeg
:label: fig:heart-data

A robot who loves data.
:::

The **Population** refers to the maximum number of data points we can possibly gather in terms of the subject of our hypothesis or model. If we are modelling adult human heights in 2026, the population would be all human beings alive in 2026. If we are modelling the heights of women between the ages of 22-25, our population would be all women between the ages of 22 and 25.

A **Sample** is a subset of the population. If certain criteria are met, we can use the sample as a proxy for the population. The Representative Sample must be reasonably large and unbiased. Lots more on this to come.

> Even if we use the entire population at a given moment as our dataset, we must still recognise that our dataset is limited in size. 


## Random Variables, Parameters, and Statistics

An example of a **Variable** is height. It is a label for a property that can be measured. Any particular heights that we are able to measure are samples from the theoretical distribution of all possible heights. This underlying truth distribution has an infinite number of data points, and it lives in the theoretical universe. 


:::{figure} /figures/TheoreticalUniverse.png
:label: fig:theoretical-universe

An attempt at a visual explanation of the Underlying Truth distribution.
:::

We call variables such as height **Random Variables (RVs)** because they are in a sense randomly selected instances of height from that infinite-data distribution. The "random" refers to our understanding that the heights of everyone alive today are no more special or representative of the truth than the heights of everyone alive 100 years ago.

A **Parameter** is a number that describes some characteristic of the true underlying distribution, for example the mean height of all humans. 

A **Statistic** is a number that describes some characteristic of a sample, for example the mean height of all humans in Sussex Uni. 


## Discrete and Continuous Data

**Discrete RVs** are counts or rates that can only have certain values, rather than any values on the real number line ($\mathbb{R}$). 

> Discrete RVs don’t have to be integers, but they do have to be countable.

For example, the probability of dice rolls (any number of dice) **are not** integers but **are** Discrete. As I increase the number of dice and/or rolls, I can generate lots of probability values from the original set, but there will always be real numbers I cannot generate (gaps).


**Continuous RVs** are measurements such as height or temperature that can theoretically take any value on the real number line. 

Note that in the real world, measurements always have a finite precision, so the measurements x of a continuous RV will not strictly be continuous.

What is true in the real world is also in true in your computer. The precision with which python generates CRVs on my machine is ```float64```: precision 15.

> A continuous RV is still continuous even though its measurements cannot be.



## Summary Statistics 

**Summary statistics** are numbers that describe the properties of a whole sample (dataset), rather than a single data point. Examples of summary statistics we will use in this module are the sample Sum, Mean, Median, Mode, Minimum, Maximum, Variance, and Standard Deviation.

### Mean $\overline{x}$

The **Mean** $\overline{x}$ of $N$ measurements $x_i$ of a random variable $X$ is the sum of all measurements divided by the number of measurements:
$$
\label{eq:mean}
\overline{x}  = \frac{1}{N} \sum\limits_i^N  x_i
$$

```{code-cell} python
import numpy as np 
x = [5,10,12]
mean_x = np.mean(x)

# check
sum_x = np.sum(x)
mean_x_check = sum_x / len(x)

print(f" mean x : {mean_x}, check: {mean_x_check }")
```

> When we talk about the Theoretical Universe ("Truth") where Probability Distributions live, we often use the term **Expected Value** or **Expectation** rather than Mean. The Expected Value is a parameter, not a summary statistic, and is defined in terms of the Probabilities $p_i$ of the individual measurements rather than as a normalised sum over them: $E[X]  = \sum\limits_i^\infty  x_i p_i$. Lots more on this later.

### Variance $V[x]$ <a name="intro-variance"></a>

The sample **Variance** $V[x]$ is a measure of the spread of a dataset $x$ with respect to the mean.

$$
\label{eq:variance}
V[x]  = \frac{1}{N} \sum\limits_i^N  (x_i - \overline{x})^2
$$

The term $(x_i - \overline{x})^2$ is the variance of the $i^{th}$ measurement, $x_i$. The sum is over the $N$ measurements.

Sometimes we will see the **Unbiased Variance**:

$$
\label{eq:varianceU}
V[x]  = \frac{1}{N-1} \sum\limits_i^N  (x_i - \overline{x})^2
$$

It is better to use the unbiased form if you have a small dataset. I would probably use this form for $N<100$ as a rule of thumb. More on this later.

```{code-cell} python
import numpy as np 
x = [5,10,12]
var_x = np.var(x)

# check
mean_x = np.mean(x)
sq_diffs = [ (xi - mean_x)**2 for xi in x]
sum_sq_diffs = np.sum(sq_diffs)
var_x_check = sum_sq_diffs / len(x)

print(f" var x : {var_x}, check: {var_x_check }")

unbiased_var_x = np.var(x,ddof=1)

print(f" (unbiased variance: { unbiased_var_x})")

```

### Standard Deviation $\sigma_x$

The **Standard Deviation** $\sigma_x$ is the square root of the variance.

$$
\label{eq:std}
\sigma_x  = \sqrt{V[x]}
$$

```{code-cell} python
import numpy as np 
x = [5,10,12]
std_x = np.std(x)

# check
var_x = np.var(x)
std_x_check = np.sqrt(var_x)

print(f" std x : {std_x}, check: {std_x_check }")

unbiased_std_x = np.std(x,ddof=1)

print(f" (unbiased std x : {unbiased_std_x})")

```

## Test Statistics

**Test statistics** are numbers that describe the compatibility of a sample with a hypothesis, or with another sample. Examples are the p value, Z score, Chi squared. We will discuss these at length later.

## Random and Pseudorandom Numbers

A **Random Number** is a term used to describe a value that is produced by a random (unpredictable) process.

There are many things in the physical world that exhibit random behaviour (such as quantum mechanical processes), but the instruments we use to measure the behaviour often destroy/hide the randomness.

We can extract (a limited number of) truly random numbers from eg [](random.org), which uses atmospheric noise.

True Random Number Generation (TRNG) is an active area of research, and is rapidly changing. Whatever I write here will probably be out of date in two years. A recent exciting development was published in [Nature](https://www.nature.com/articles/s41586-025-09054-3) last summer.


A **Pseudorandom Number** (PRN) is a term used to describe a number in a sequence that appears random, but is produced by a deterministic process. Deterministic means that an outcome is caused by preceeding events.


In this module we will generate PRNs using ```numpy``` and ```scipy```.

```{code-cell} python


# Generate 10 numbers uniformly distributed between the values of 5 and 95

# using numpy:
import numpy as np
rng = np.random.default_rng()
x_uniform_numpy = rng.uniform(low=5,high=95,size=10)

# using scipy: 
from scipy.stats import uniform
x_uniform_scipy = uniform.rvs(loc=5,scale=90,size=10)

# generate 15 numbers from a normal distribution with mean 7 and standard deviation 4.1

# using numpy:
x_normal = rng.normal(loc=7,scale=4.1,size=15)

# using scipy:
from scipy.stats import norm
x_normal_scipy = norm.rvs(loc=7,scale=4.1,size=15)

```

:::{note}
Human beings tend to have fixed ideas of what random should look like, which is
amusing if you think about it. If you truly had a random shuffle on your music
playlist, you would get many repetitions. People don’t like that.
:::

:::{figure} /figures/dilbert2.jpg
:label: fig:dilbert

Credit: DILBERT © 2001 Scott Adams [All rights reserved].
:::



## Plots, axes, histograms, bins

* A **Plot** is a visual representation of data (a "graph")


```{code-cell} python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(x=[1.41, 1.52, 1.56, 1.61, 1.70 ], 
	    y=[60, 55, 65, 67, 80], 
	    color='skyblue', 
	    markerstyle='*',
	    linewidth=2,
	    label='legend entry' 
	    )

ax.set_title("Example of a plot")
ax.set_xlabel("Height (m)")
ax.set_ylabel("Weight (kg)")
plt.show()

```

* A **Histogram** is a bar plot indicating the counts of measurements within defined ranges of values. These ranges are known as **Bins**.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng()

heights = rng.normal(loc=165, scale=15, size=1000)

fig, ax = plt.subplots()

ax.hist(heights, 
	    bins=10,
	    histtype='stepfilled',
	    facecolor='skyblue', 
	    edgecolor='black', 
	    linewidth=2,
	    alpha=0.5,
	    density=False,
	    label='legend entry'
	    )

ax.set_title("Example of a histogram")
ax.set_xlabel("Height (m)")
ax.set_ylabel("Count/ bin")
plt.show()

```

* **Axes** define the space of the plot. A 1D plot has an x axis along the horizontal direction, indicating the values (or bins) for the measurement, and a y axis indicating the count, frequency, or density of those measured values.

To "plot on the same axes" means to draw two or more datasets on the same "graph pad".

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng()

heights = rng.normal(loc=165, scale=15, size=1000)

weights = rng.normal(loc=65, scale=10, size=1000)

fig, ax = plt.subplots()

ax.plot(heights[:50], weights[:50], 
	    color='black', 
	    markerstyle='*',
	    label='First 50')

ax.plot(heights[50:100], weights[50:100], 
	    color='red', 
	    markerstyle='o',
	    label='Second 50')

ax.set_title("Example of a histogram")
ax.set_xlabel("Height (m)")
ax.set_ylabel("Weight (kg)")
plt.show()

```




