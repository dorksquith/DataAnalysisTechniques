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

# Statistical Tests

## Hypotheses

A Hypothesis is an educated guess at the explanation for some occurrence. It must
be based on real observations and it must be testable.

For example, if I collect some data (and I do not know the underlying PDF) I may
observe that it looks like exponential decay when I plot it, but it has some lumps and bumps, so something more interesting could possibly be going on...

The Null Hypothesis H0 is the educated guess that the most obvious (boring?)
explanation for the data is the correct one. For the example above, the Null
Hypothesis could be that the data is indeed Expon(λ).

The Alternative Hypothesis H1 is an alternative to H0. For the example above, the
Alternative Hypothesis would be that the lumps and bumps are exciting resonances
that will win your research group a nobel prize and pay for a new espresso machine.

### Example: The Higgs

After waiting forever for the LHC to get going after some unfortunate stalls due to
bad soldering and a ferret, we started to see ‘Higgs-like’ signs in 2011.



:::{figure} 
:label: fig:higgs
:align: left
![](figures/higgs.png)

Higgs hunting, back in the day.
:::


## Bias

My tangible excitement at the idea of an
alternative hypothesis should raise a big red flag.
Anyone who has done Harvard’s excellent
Unconscious Bias tests will know that humans are
hopeless at being unbiased, even when we aren’t
aware that we have a bias.
This means that if we are going to do excellent
science, and not just come up with whatever result
means we get a new espresso machine, we have to
remove any chance of introducing bias.
One way of doing this is to set out very strict
criteria for the null and alternate hypotheses before
we look at our data.

:::{figure} 
:label: fig:espresso
:align: left
![](figures/espresso.jpg)

A beautiful espresso machine.
:::



## Critical Regions

A Critical Region W is a region of the Sample Space in which the probability of
finding the data, given the Null Hypothesis, is very small.
The region W can have any shape, making it suitable for tests involving multivariate
data. The simplest possible definition of W for univariate data would be to draw a
vertical line (a ‘cut’) and define W as being on one side of it.
A Positive Result means that we have found our data x in W , and so we reject the
Null Hypothesis.
One cannot prove a hypothesis. Hypotheses can only be rejected or not rejected.
In reality it is impossible to find a W that is not consistent with both H0 and H1. If
we could, we would not need to do any of this....


“Define a Critical Region W . This is a region of the Sample Space in which the
probability of finding the data, given the Null Hypothesis, is very small.”
What does very small mean?
A popular choice in the wide world (but not in my world) is to set a tolerance level of
5%. We define our region W such that there is a 5% or lower probability of finding
the data in that region, given the null hypothesis H0.
This is called the Significance Level, α.
P(x ∈W |H0) ≤α
Once we when we have decided on W and α, we can look at (‘unblind’) our data.


### Choosing a Critical Region is challenging


Is purity or efficiency more important to
you? High purity means small α(small
contamination) and lower power (you have
cut out a lot of your sample).
Are you suffering from a small dataset? If
so, you will probably have to maximise
power at the cost of purity.
Will your funding be withdrawn if you
don’t publish? This is not a valid reason
to make a scientific decision, but is
shockingly common (see later).

### Limitations of the Statistical Test

If we find our data satisfies our predefined statistical test P(x ∈W |H0) ≤α, then
we have some responsibility to publish.
With a Significance Level of α= 5%, this would give me major heeby jeebies.
The statistical test for rejecting H0 has a False Positive rate of up to 5%,
which to me seems enormous. A one in twenty chance that this is...wrong.
This is called a Type 1 Error.
If we find our data fails our predefined statistical test, then we have no reason to
reject H0.
This does not mean H0 is correct; a hypothesis cannot be shown to be correct.
It could be that we just got unlucky - the data ended up in some other critical
region that we did not choose for our predefined statistical test.
This is a False Negative or Type 2 Error

## The Confusion Matrix

## Significance: P values and Z values

The Significance of a measurement is expressed in terms of either a p-value or
z-value, both of which are calculated from the data.



## Confidence levels and limits