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
observe that it looks like exponential decay, when I plot it, but it has some lumps and bumps, so something more interesting could possibly be going on...

The **Null Hypothesis** $H_0$ is the educated guess that the most obvious (boring?)
explanation for the data is the correct one. For the example above, the Null
Hypothesis could be that the data is indeed $\sim$Expon($\lambda$)[^probs].

[^probs]: The notation $\sim$ Expon($\lambda$) means "follows an exponential distribution". The parameter of the exponential distribution is $\lambda$. We will meet many distributions later and refer to them in this way: $\sim$ DistributionName(parameters) or $\sim$ DistributionName(parameter values).

The **Alternative Hypothesis** $H_1$ is an alternative to $H_0$. For the example above, the Alternative Hypothesis would be that the lumps and bumps are exciting resonances that will win your research group a Nobel prize and pay for a new espresso machine.

### Example: The Higgs

After waiting forever for the LHC to get going after some unfortunate stalls due to
bad soldering and a ferret, we started to see ‘Higgs-like’ signs in 2011. The plot in [](#fig:higgs) shows the Null Hypothesis (no Higgs) as a red dashed line and the Alternative Hypothesis (Higgs) as a solid red line. In this case, we were not able to disprove the Alternative Hypothesis, and we have still not been able to disprove it. When a particle physicist talks about Discovery (the D-word) we mean we have failed to disprove the opposite ()[#discovery].



:::{figure} 
:label: fig:higgs
:align: left
![](figures/higgs.png)

Higgs hunting, back in the day.
:::


## Bias

My tangible excitement at the idea of an
Alternative Hypothesis should raise a big red flag.
Anyone who has done [Harvard’s excellent
Unconscious Bias tests](https://implicit.harvard.edu/implicit/takeatest.html) will know that humans are hopeless at being unbiased, and hopeless at being aware that we have a bias.
This means that if we are going to do excellent
science, and not just come up with whatever result
means we get a new espresso machine, **we have to
remove any chance of introducing bias**.
One way of doing this is to set out **very strict
criteria** for the null and alternate hypotheses before
we look at our data.

:::{figure} 
:label: fig:espresso
:align: right
:width: 40%

![](figures/espresso.jpg)

A beautiful espresso machine.
:::



## Critical Regions

A Critical Region $W$ is a region of the Sample Space $S$ in which the probability of finding the data, given the Null Hypothesis, is very small^[verysmall].

[^verysmall]: "very small" is a qualititive term, which will be defined quantatively as part of our test.

:::{figure} 
:label: fig:robot-critregion
:align: right
:width: 40%

![](figures/RobotCritRegion.png)

A Robot showing their friends a simple critical region.
:::

The region $W$ can have any shape, making it suitable for tests involving multivariate data. The simplest possible definition of W for univariate data would be to draw a vertical line (a ‘cut’) and define W as being on one side of it, as illustrated by our robot friends in [](#fig:robot-critregion).

A **Positive Result** means that we have found our data x in the critical region, W , and so we **reject the Null Hypothesis**.

> Reminder: One cannot prove a hypothesis. Hypotheses can only be rejected or not rejected.



We stated above that W is a region of the Sample Space in which the
probability of finding the data, given the Null Hypothesis, is **very small**.

What does very small mean?

A popular choice in the wide world (but not in my world) is to set a tolerance level of $5%$. We define our region W such that there is a 5% or lower probability of finding the data in that region, given the null hypothesis $H_0$.

This tolerance level is called the **Significance Level, $\alpha$**




$$\label{eq:testW} P(x \in W | H_0) \leq \alpha $$





Once we when we have decided on $W$ and $\alpha$, we can look at (‘unblind’) our data.

The two strict criteria for designing a statistical test that is not prone to bias:

1. Define a critical region, $W$
2. Choose a significance level, $\alpha$


### Choosing a Critical Region is challenging

In reality, it is impossible to find a $W$ that is not consistent with both $H_0$ and $H_1$. There will always be overlap between them, as I have attemped to illustrate in [](#fig:critregion). 


:::{figure} 
:label: fig:critregion
:align: right
:width: 40%

![](figures/CriticalRegion_5.png)

A cartoon showing the PDFs for the Null (top) and alternative (bottom) hypotheses.
:::

In this simplified example, the critical region that provides us with a significance level $\alpha=5%$ exludes more than half of the distribution for the alternative hypothesis. In life we may find that our critical region eliminates much larger fractions of the alternative hypothesis probability space. We need to think carefully about how we define our critical regions to ensure we mazimize our **Sensitivity** to the data. 

Defining a critical region can involve carefully defining a test statistic, which could be eg the output from a multivariate classifier, that minimises the overlap between $H_0$ and $H_1$. Minimizing this overlap maximizes our sensitivity. We must also choose our siginficance level $\alpha$ in parallel with defining W.

### The Hypotheses as Universes

Notice that [](#eq:testW) is a Conditional Probability. The Condition is $H_0$; that is, we are conditioning on the Truth, which cannot be known by us.

We previously noted that the [True Underlying Distribution](#fig:theoretical-universe) lives in the Theoretical Universe. In our real, physical universe all we can do is look at statistically-limited (finite) samples from it.

When we consider a hypothesis, we are considering a theoretical universe in which that hypothesis is correct. The null and alternative hypotheses cannot both be correct in the same universe. So, when we design a test that is conditional on $H_0$, we are simply testing the compatibility of our real universe data with one possible theoretical universe. The same goes for $H_1$ and any other Hypothetical Universe we care to think up. 

### Purity and Efficiency


There are several ways in which purity and efficiency can be usefully defined. 

Here is a simple definition of **Purity**:

$$
:label: eq:purity1
\mathsf{Purity} =  \dfrac{\mathsf{Number correctly classified as A}{\mathsf{ Total number classified as A}}
$$


Notice that the numerator uses the word "Correctly", which indicates we have knowledge of the Truth. 

Using a lower case symbol a for the classification and an upper case symbol A for the truth, we can write this definition of **Purity** more concisely:

$$
:label: eq:purity2
p =  \dfrac{\mathsf{ N(A \cap a) }{\mathsf{ N(a)}}
$$

Using the reasoning in [](#cond-prob), we can therefore write the purity as a [Conditional Probability](#eq:conp):

$$
:label: eq:purity3
p = P(A|a) 
$$


Here is a simple definition of **Efficiency**:

$$
:label: eq:efficiency1

\mathsf{Efficiency} = \dfrac{\mathsf{Number correctly classified as A}{\mathsf{ Total number that are truly A}}
$$

The more concise form is:

$$
:label: eq:efficiency2
e =  \dfrac{\mathsf{ N(A \cap a) }{\mathsf{ N(A)}}
$$

And as a conditional probability:

$$
:label: eq:efficiency3
e = P(a|A) 
$$


### Contingency Tables

A Contingency Table allows us to summarise the efficiency and purity of our tests in terms of the Truth: True/False and the Classification:Postive/Negative. 

:::{table} Contingency Table Example in terms of sets A (truth) and a (classification).
:widths: auto
:align: center

|      | a   | a'  | 
| ---  | --- | --- | 
| A    | a\capA  | a'\capA  | 
| A'   | a\capA' | a'\capA' | 
:::



Example: I have 100 robots, 41 of whom were designed by my friend Alf. The set A has 41 elements. I am attempting to predict which ones are Alf's design using their features. My classifier identifies 60 robots as being designed by Alf. The set a has 60 elements. Of these 60, 37 are in fact Alf's (I was right), and 23 are not (I was wrong). 

Summary:
* $N(a\cap A) = 37$
* $N(a\cap A') = 23$
* $N(a'\cap A) = 4$ (inferred, 41-37)
* $N(a'\cap A') = 36$ (inferred, 40-4)

:::{table} Contingency Table for Alf's robots.
:widths: auto
:align: center

|      | a   | a'  | Total |
| ---  | --- | --- | --- |
| A    | 37  | 4   | 41  |
| A'   | 23  | 36  | 59  |
| Total| 60  | 40  | 100 |
:::

Note that we are able to fill out the whole table using the information above and basic logic. But just because the logic is "basic", doesn't mean that it is easy to get your head round making and using contingency tables. We will practice this.

For this example, the purity is calculated in [](#purity-alf) and the efficiency in [](#efficiency-alf). 

$$
:label: purity-alf

p = P(A|a) = N( A\cap a) / N( a ) = 37/60 \approx 62\%

$$

The efficiency is:

$$
:label: efficiency-alf

e = P(a|A) = N( A\cap a) / N( A ) = 37/41  \approx 90\%

$$


I would conclude that this classifier as having good efficiency, as it misses only 10% of the Alf robots, but questionable purity, as it misclassifies a lot of non-Alfs as being Alf designs.



### The Purity versus Efficiency play-off

Choosing a small value of $\alpha$ in the construction of our [test](#eq:testW) results in very high **purity** for $H_0$, meaning you are very unlikely to find the data in W if $H_0$ is true. But, a smaller value of $\alpha$ is achieved by shifting the region such that we increase the overlap with  $H_1$, lowering the test's **efficiency** for the alternative hypothesis. The efficiency for $H_1$ is referred to as the **Power of the test**.


<!--
The purity of our critical region W, which is designed to exclude the null hypothesis $H_0$, is the proportion of the region not overlapping with $H_0$:
$$
:label: purity-null
p_W = P( H_0' | W )  = 1-\alpha
$$-->



**Efficiency ("power") is more important: choose larger $\alpha$**
If you are **Statistically Limited** (suffering from a small dataset) you will probably try to maximise efficiency at the cost of purity. But, choosing a larger significance level $\alphs$ means the findings of your test are less significant, so less likely to be published or considered important. As such, it is often a good choice to gather more data, such that a stronger (smaller) choice of $\alpha$ can be used.


**Purity (significance) is more important: choose smaller $\alpha$**
There may be motivation to make $\alpha$ very small in an effort to make the critical region "Ultra Pure". The alternative hypotheses (there is often more than one of these) must be considered carefully for this decision, as we don't usually want to define a region W that has been cleansed not only of $H_0$, but also most of $H_1$.







<!--
:::{figure} 
:label: fig:robot-critregion
:align: right
:width: 40%

![](figures/RobotCritRegion.png)

A Robot showing their friends the overlap between the null and alternative hypotheses.
:::
-->





<!--Is purity or efficiency more important to
you? High purity means small α(small
contamination) and lower power (you have
cut out a lot of your sample).
Are you suffering from a small dataset? If
so, you will probably have to maximise
power at the cost of purity.
Will your funding be withdrawn if you
don’t publish? This is not a valid reason
to make a scientific decision, but is
shockingly common (see later).-->




## Limitations of the Statistical Test

If we find our data satisfies our [predefined statistical test](#eq:testW), then this is a **Positive Result** and we have some responsibility to publish.

> It is very common in many fields of research to choose a **Significance Level** of $\alpha= 5%$, but this would give me major heeby jeebies. The statistical test for rejecting $H_0$ has a **False Positive Rate** of up to 5%, which to me seems enormous. A one in twenty chance that this is...wrong.

If we find a Positive Result and are wrong, this is a **False Positive**, and is called a **Type 1 Error**.

If we find our data fails our [predefined statistical test](#eq:testW), then this is a **Negative Result**: we have failed to reject $H_0$.

This does not mean $H_0$ is correct; a hypothesis cannot be shown to be correct. It could be that we just got unlucky - the data ended up in some other critical
region that we did not choose for our predefined statistical test.

If we find a Negative Result and are wrong, this is a **False Negative**, and is called a **Type 2 Error**.


Because **we can never know the truth**, a positive result is either exciting or very embarrassing, and we have no way of knowing which. We can reduce the probability of embarrassment by making $\alpha$ very small, but this is only possible if we have lots of data and can design our phase space such that discernment between $H_0$ and $H_1$ is possible with a very small value of $\alpha$.


### Probabilities are Conditional on the Unknown

:::{figure} 
:label: fig:falsepos
:align: right
:width: 40%

![](figures/CriticalRegion-Wrongness_5.png)

The probabilities of a False Positive (Type 1 Error) and False Negative (Type 2 Error) are set by our choice of $\alpha$ and W in the test design.
:::



**If the Null Hypothesis were true**, then the correct result is a True Negative. There are two possibilities for our measurement, and their Conditional Probabilities must add to 1.

$$P(Negative | H_0) + P(Positive | H_0) =1 $$

The Conditional Probability of a False Positive is labeled as $P(pos|H_0)=\alpha$ in [](#fig:falsepos). 

**If the Alternative Hypothesis were true**, then the correct result is a True Positive. There are two possibilities for our measurement, and their Conditional Probabilities must add to 1.

$$P(Negative | H_1) + P(Positive | H_1) =1 $$

The Conditional Probability of a False Negative is labeled as $P(neg|H_1)=\beta$ in [](#fig:falsepos). 

> The "power of a test" (efficiency for $H_1$ is 1-$\beta$)


:::{important} 
The values of $\alpha$ and $\beta$ are of no use whatsoever to our interpretation of the measurement, as they are conditional on something we can never know. They are only useful in terms of setting up our test.
:::


## The Confusion Matrix

The ways in which a test can play out are summarised in the so-called "Confusion Matrix" which I have tried to explain in the series of images in [](#fig:confusion-mp4).


:::{figure} 
:label: fig:confusion-mp4
:align: right
:width: 40%

![](figures/confusion-matrix.mp4)

The Confusion Matrix.
:::


## Significance: P values and Z values

The Significance of a measurement is expressed in terms of either a p value or
z value, both of which are calculated from the data.

The **p value is very often misinterpreted**. It is very hard to understand what it means, because it is calculated from an extrapolation between our real physical universe and a theoretical universe in which the null hypothesis is true. Thinking simultaneously about two universes, one of which is entirely inaccessible to us, is hard.

:::{figure} 
:label: fig:pval1

![](figures/CriticalRegionPvalue.png)

We measure the test statistic in our data. We compare this with the hypothetical distribution of $H_0$, and define the area intersected by our data and $H_0$ as the p value.
:::

Once we have made a measurement of our test statistic, indicated by the green line on [](#fig:pval1), we calculate the p value as **the area under the Null Hypothesis curve beyond our measured value**. If the p value is smaller than the chosen significance level $\alpha$, indicated by the pink dashed line in [](#fig:pval1) we have a Positive Result.

If we measure a test statistic that yields a p value of 3%, we can say that our data would occur at or beyond this value in 3% of experiments, **given an assumption that the null hypothesis is true**. 

This is a strange thing to say, if you think about it. We can never know if the null hypothesis is true or not, but we are quoting a result conditional on it being so.

> I would personally not interpret a p value of 3% as suggestive that we should reject $H_0$. In particle physics, we only claim "discovery" if we measure a p value of < 0.00006%, and would not even raise an eyebrow for p values above 1%.



## Confidence Levels and Limits





