# Terminology & Concepts


Make sure you are familiar with these before embarking on the topics in this module.

## Hypotheses, Models, and Theories

:::{note}
A Null Hypothesis is one that assumes there is no relationship between two variables, and an Alternative Hypothesis is one that assumes there is. 
:::

A hypothesis is an educated guess at the outcome of a specific event. Examples:
- There is no correlation between how long we sleep and how long we live.
- People who sleep well live longer.
- There is no correlation between the messiness of someone's office and that of their home.
- People with messy offices have tidy homes.


:::{note}
All models are wrong, but some are useful.
:::
A model is an object, a mathematical equation, or a computer program that can explain how something works, or be used to make predictions. Examples:
- a model of a delorian (a toy car based on time-travelling car in back to the future movies, which captures the visual appearance of the machine)  
- a Climate model (a complex computer program that tells us how much trouble we are in existentially) 
- the Standard Model of particle physics (a set of mathematical equations describing how the universe works at a fundamental level)

:::{note}
When statisticians talk about Toy Models (often shortened to "Toys"), they are referring to a mathematical model that has been simplified. 
:::

A theory is the description of a set of models, and the relationships between them. It is complete, and is a candidate explanation for the Underlying Truth, which cannot be known.


The Underlying Truth is an abstract concept which cannot be known. We can test our models and theories, and those which we fail to prove wrong can become highly esteemed. 

:::{note}
It is not possible to prove that a hypotheses, model, or theory is correct. We can only prove them wrong. 
:::

A theory that passes all the tests we have thought of, or have the technological capability to perform, may well be wrong - we just haven't asked the right questions or gathered enough data to know yet.

## Population and Sample

To test our models we must gather data as evidence. How much data do we need? No amount is ever enough!

The Population refers to the maximum number of data points we can possibly gather in terms of the subject of our hypothesis or model. If we are modelling adult human heights in 2026, the population would be all human beings alive in 2026. If we are modelling the heights of women between the ages of 22-25, our population would be all women between the ages of 22 and 25.

A Sample is a subset of the population. If certain criteria are met, we can use the sample as a proxy for the population. The Representative Sample must be reasonably large and unbiased. Lots more on this to come.

:::{note}
Even if we use the entire population at a given moment as our dataset, we must still recognise that our dataset is limited in size. 
:::

## Random Variables, Parameters, and Statistics

An example of a variable is height. It is a label for a property that can be measured. Any particular heights that we are able to measure are samples from the theoretical distribution of all possible heights. This theoretical distribution has an infinite number of data points, and it lives in the theoretical universe. We call variables such as height Random Variables (RVs) because they are in a sense randomly selected instances of height from that infinite-data distribution. The "random" refers to our understanding that the heights of everyone alive today are no more special or representative of the truth than the heights of everyone alive 100 years ago.

A Parameter is a number that describes some characteristic of the true underlying distribution, for example the mean height of all humans. 

A Statistic is a number that describes some characteristic of a sample, for example the mean height of all humans in Sussex Uni. 

## Summary Statistics and Test Statistics

Summary statistics are numbers that describe the properties of a whole sample (dataset), rather than a single data point. Examples are the sample Mean, Median, Mode, Minimum, Maximum, Variance, and Standard Deviation.

:::{math}
:name: eq:mean

\overline{x}  = \frac{1}{N} \sum\limits_i^N  x_i
:::


Test statistics are numbers that describe the compatibility of a sample with a hypothesis, or with another sample. Examples are the p value, Z score, Chi squared.

## Random and Pseudorandom Numbers

:::{figure} https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/dilbert2.jpg
:label: fig:dilbert
:::

A 


Mean, Median, and Mode
Variance and Standard Deviation
Plots, axes, histograms