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

# Probability Essentials


Probability is very important to human beings. We are subject to random
events in nature that can completely change the course of our lives,
leading to world domination or total annihilation.
It is not surprising that games of chance are so popular, and have
probably been around much longer than eg written communication.

## Two Philosophies

There are two ways of viewing probabilities: Frequentist and Bayesian.

**Frequentist**: everything is defined in terms of repeated experiments. The probability of something happening can only be defined in terms of how often it is observed to happen.

**Bayesian**: the probability of something happening is as per the frequentist definition, but we must "weight" this using our relevant prior knowledge.

Let's anthropomorphise these interpretations by introducting a Frequentist, Fernanda, and a Bayesian, Betty, imagined by Gemini in [](#fig:fernanda-and-betty).

:::{figure} /figures/FernandaAndBetty.jpeg
:label: fig:fernanda-and-betty

Fernanda and Betty.
:::


### Where are my keys?

Fernanda and Betty have each misplaced their keys. They will each adopt a different method for finding them based on their different philosophies of probability.

**Fernanda**: my keys are in an unknown location in my home. There is no "probability of them being by the sink", because they are either in a place or not. The only way to locate them is to look in every location methodically.

**Betty**: my keys are in an unknown location in my home. On the last ten occasions I lost them, they were by the sink three times, so there are most likely to be by the sink and I will start my search there.


### Is my coin fair?

**Fernanda**: I will toss the coin a million times and see how many times we get heads.

**Betty**: I will also toss the coin several times, but before I do so I will consider what we know about the coin. Did we get it from a joke shop, or a post office?

### Is a pregnancy test correct?

**Fernanda**: I will do the test a million times, and count how many times it is correct[^tests].

[^tests]: The question of the pregnancy test is an interesting one, because we have to define exactly what we are asking in order to define the denominator. The probability of a positive test being wrong is not the same as the probability of a wrong test if we are postive...

**Betty**: I will collect data similarly to Fernanda, but will also consider what we know about the person taking the test, for example: do they have a uterus?

 
> It may appear that the Bayesian approach is prone to bias, so I should note that the Bayesian does not "stick to their guns" on the prior knowledge aspect. If the data indicate their prior knowledge is unlikely to be correct, then the prior is updated. 



## Sets

Let's define some arbitrary example sets:

```{math}
:label: our-sets

\begin{aligned}
S &= \{1,2,3,4,5,6,7,8,9,10\} & \\
A &= \{2,4,6,8,10\}& \\
B &= \{1,2,3,4\}& \\
\end{aligned}
```

```{code-cell} python
import numpy as np

# Our sets

S = set(range(1,11)) 

A = {i for i in S if i % 2 == 0} 
B = {i for i in S if i <5} 


print(f"S= {S}")
print(f"A= {A}")
print(f"B= {B}")
```


We will use these to demonstrate some terminology numerically.




### Special Sets
:::{figure} /figures/SpecialSets.png
:label: fig:special-sets

Some special sets.
:::


### Sample Space & Subsets

The **Sample Space** S is the set of all possible outcomes of some experiment or operation.

A **Proper Subset** of S is denoted $A\subset S$. This means that A is a subset of S but A$\neq$S. If A is a subset of S *and* can be S, we use $A\subseteq S$

For [our sets](#our-sets), A and B are proper subsets of S: $A \subset S$ and $B \subset S$.

```{code-cell} python

A_subset_S = A.issubset(S)
print(f"A.issubset(S): {A_subset_S}") 

# equivalent method with different python operator
A_subset_S_alt = A <= S
print(f"A <= S: {A_subset_S_alt}" ) 

```



### Venn Diagrams

Venn diagrams are helpful for visualising relationships. Here $A\subset S$ and $B\subset S$.

::::{grid} 1 1 3 3

:::{image} /figures/S.png
:::

:::{image} /figures/A.png
:::

:::{image} /figures/B.png
:::

::::


### Complement $A'$

The **Complement** of a set is denoted by a prime, $A'$; this means 'not A'. Other notations commonly used for the complement (though not by me) include $A^{\complement}$ and $\overline{A}$.

::::{grid} 1 1 2 2

:::{image} /figures/NotA.png
:::

:::{image} /figures/NotB.png
:::

::::



### Union $A\cup B$

The **Union** of two sets is written $A\cup B$; this means 'either A, or B, or both'. For [our sets](#our-sets), $A\cup B = \{1,2,3,4,6,8,10\}$

```{code-cell} python
AUB = A|B 
print (f"Union AUB ={AUB}")
```


### Intersection $A\cap B$

The **Intersection** of two sets is written $A\cap B$; this means 'both A and B'. For [our sets](#our-sets), $A\cap B = \{2,4\}$


::::{grid} 1 1 2 2

:::{image} /figures/AorB.png
:::

:::{image} /figures/AandB.png
:::

::::

```{code-cell} python
AnB = A&B # {2,4}
print (f"Intersection AnB ={AnB}")
```

The **Complement of the Union** is $(A\cup B)'$ and means 'Not in A and not in B (and not in both)'.


The **Complement of the Intersection** is $(A\cap B)'$ and means 'Not in both A and B'.


::::{grid} 1 1 2 2

:::{image} /figures/NotAorB.png
:::

:::{image} /figures/NotAandB.png
:::

::::



### Independence and Mutual Exclusivity

Two sets are **Independent** if they are defined without reference to one another (changing one does not impact the other). We could construct **dependent** sets like this for example:
* $K = {\mathbb{N}}$
* $J = {K^2}$

[Our sets](#our-sets) are independent.

Two sets are **Mutually Exclusive** if there is no overlap between them. Another word for this is **Disjoint**.


## Frequentist Probability


The **Frequentist Probability** of A is written $P(A) = \dfrac{N_A}{N_S}$: the number of times A occurs in the sample, divided by the number of possible outcomes
in the sample.

Probabilities for [our sets](#our-sets):
* $P(S) = N_S/N_S =1$
* $P(A) = N_A / N_S  = 0.5$
* $P(B) = N_B / N_S  = 0.4$





```{code-cell} python

P_S = len(S)/len(S) 
P_A = len(A)/len(S)  
P_B = len(B)/len(S)  


print(f"P(S)= {P_S}")
print(f"P(A)= {P_A}")
print(f"P(B)= {P_B}")

```





### Probability of Intersection (Joint Probability)

The **Probability of Intersection** $P(A\cap B)$ is the probability that **both A and B** are true. This is also called the **Joint Probability** of A and B.

The Joint Probability is zero when A and B are **Disjoint** (no overlap: **Mutually Exclusive**):

$P(A\cap B) =0$ when $(A\cap B)'$

For [our sets](#our-sets):
* The intersection is $A\cap B = \{2,4\}$
* The Joint probability is $P(A\cap B) = 0.2$
* $A$ and $B$ are not mutually exclusive.


### Probability of Union

The **Probability of Union** $P(A\cup B)$ is the probability that either A or B or both are true. 

$$\label{eq:prob-union} P(A\cup B) =P(A)+P(B) - P(A\cap B)$$

Note that the subtraction of the intersection $P(A\cap B)$ in [](#eq:prob-union) is to remove the double counting of that intersection, as illustrated in [](#fig:venn_axiom3).

:::{figure} 
:label: fig:venn_axiom3
:align: left
![](/figures/Axiom3.png)
:::

For [our sets](#our-sets): $P(A\cup B) = 0.5 +0.4 - 0.2 = 0.7$


```{code-cell} python
P_AUB = len(AUB)/len(S) # 0.7
P_AnB = len(AnB)/len(S) # 0.2
print("P(AUB) = ",P_AUB )
print(f"P(A) +P(B) - P(AnB) = {P_A} + {P_B} - {P_AnB} = {P_A+P_B-P_AnB}")
```
(kolmogorov)=
### The Kolmogorov Axioms

> [Andrey Kolmogorov](https://en.wikipedia.org/wiki/Andrey_Kolmogorov) was a Russian mathematician. You may have heard of the 'KS test', which is used to eg check for overtraining by comparing ML classifier outputs for test and train samples. This is named for Kolomogorov and Nikolai Smirnov. See also 'KANs' - Kolmogorov Arnold Networks [arXiv:2404.19756](https://arxiv.org/abs/2404.19756).

The **Non-Negativity Axiom**
: The probability of any event A must be a real number greater than zero.
  $$\label{eq:kolmogorov1} P(A) \geq 0 \;\; \forall A $$

The **Normalisation Axiom** 
: The probability of the entire sample space is one.
  $$\label{eq:kolmogorov2} P(S) =1$$ 

The **Countable Additivity Axiom**
: If A and B are **mutually exclusive**, the probability of their union is the sum of their individual probabilities.
  $$\label{eq:kolmogorov3} P(A\cup B) =P(A)+P(B)$$ 

(cond-prob)=
### Conditional Probability $P(A | B)$

The **Conditional Probability** is most usefully written:

$$\label{eq:conp} P(A | B) = \dfrac{P(A\cap B)}{P(B)}$$

The notation $ P(A | B) $ means 'the probability of A, given that B is true'. 

> When we define a function of some variables and parameters as $f(x,y;\theta)$, the semicolon ";" is used to indicate the conditional, just as the pipe "|" is used to indicate the conditional in standard probability notation. The function has variables x and y, and is conditional on the parameter $\theta$.

We can understand where [](#eq:conp) comes from as follows:

1. The number of elements in the intersection is $N(A \cap B)$.

2. The proportion of A in B is $\dfrac{N(A\cap B)}{N(B)}$

3. Relative to the original Sample Space S this is: $\dfrac{N(A\cap B)/N(S) }{N(B)/N(S)}$

4. These are **probabilities**: $\dfrac{P(A\cap B)}{P(B)}$

The condition is that the element must exist in B; P(B) is our denominator. 


For [our sets](#our-sets):

$A\cap B = \{2,4\}$: The set of events in both A and B

$N(A \cap B) = 2$: The count of events in both A and B

$\dfrac{N(A\cap B)}{N(B)} = \dfrac{2}{4}$: The fraction of B that is also in A

$P(A | B)  = 0.5 $: The probability of A, given that B is true.


```{code-cell} python
P_AgivenB = P_AnB / P_B # 0.2/0.4 = 0.5

print("P(A|B) = ", P_AgivenB )

P_BgivenA = P_AnB / P_A 

print("P(B|A) = ", P_BgivenA )
```


**If A and B are Independent**, the conditional probability is $$\label{eq:conpInd}P(A | B)  = P(A)$$. This is because independence means that B has no effect on A and vice versa.

[Our sets](#our-sets) are independent: $P(A| B) = P(A) = 0.5$.


Conditional Probability looks harmless enough, but can have some counter-intuitive results illustrated very well in the **Monty Hall Problem**, which we will think about later.



### The Multiplication Rule

Rearranging [the conditional probability](#eq:conp): $P(A\cap B) = P(A | B) P(B)$ and noting that for **Independent** A and B, $P(A | B)=P(A)$, we get the very useful **Multiplication Rule**:

$$\label{eq:multrule} P(A \cap B) = P(A) P(B)$$


This states that **if A and B are independent**, the probability of the intersection of A and B is equal to the product of their individual probabilities: 



Example: I roll a dice twice. What is the probability I will get two sixes?

$P(six \cap six) = P(six) P(six) = \dfrac{1}{6} \dfrac{1}{6}  = \dfrac{1}{36}$


### Total Probability

In the below tryptich of Venn diagrams, I have divided the Sample Space S into four quadrants, each of which is a set $B_i$. There are N=4 disjoint sets $B_i$ intersecting with A. 

The **Total Probability** of A can be written as a sum over the four quadrants:

$$\label{eq:totp1} P(A) = \sum \limits_{i=1}^4 P(A\cap B_i)$$



:::{figure} 
:label: fig:venn_tot
:align: left
![](https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/AandB1.png)


:::

Because we know how to write the intersection in terms of the [conditional probability](#eq:conp), we can express the [total probability](#eq:totp1) in terms of the conditional:

$$\label{eq:totp} P(A) = \sum \limits_i^N P(A | B_i) P(B_i)$$

This may seem a bit contrived, but we will see soon that it is useful.

> The Total Probability P(A) is also referred to as the **Marginal Probability** of A. This is because we have "Marginalised Out" the probability of B, and are only considering A.

## Bayes' Theorem

To write down Bayes' theorem we need only the **Conditional Probability** [](#eq:conp), and to observe that:

1. The [conditional probability](#eq:conp) applies to any sets A and B, so we can switch A and B and remain true:
$$\label{eq:conpB} P(B | A) = \dfrac{P(B\cap A)}{P(A)}$$

2. The intersection is not directional, so $P(B\cap A) \equiv  P(A\cap B)$.

3. This equality gives us Bayes' theorem: $ P(A | B) P(B) = P(B | A) P(A)$.

This is more usually rearranged as:

```{card} Bayes Theorem

$$
\label{eq:bayes}
\begin{aligned}
P(A | B) = 
\dfrac{
P(B|A)P(A)
}
{
P(B)
}
\end{aligned}
$$
```

Injecting [our sets](#our-sets) into Bayes' theorem is a little anticlimactic:

* $P(A | B) = P(A)$ because A and B are independent
* Bayes theorem tells us that $P(A) = \dfrac{P(B) P(A)}{P(B)}$ which is just basic algebra.

Bayes' Theorem gets interesting when we consider that it applies to **any sets $A$ and $B$ that satisfy Kolmogorov's Axioms**, as we shall see later.

:::{important} Bayes' Theorem is not "Bayesian"
Using Bayes' theorem does not make one a Bayesian. It is used by Frequentists and Bayesians alike!
:::


## Probability Distributions

We can express probabilities as single numbers, for example the probability of getting six when I roll a die is $p_6 = \frac{1}{6}$, if the die is fair and has six sides. 

More helpfully, we can express probabilities in terms of the RV and some parameters. For example, when I roll a fair die with $n$ sides, the probability of getting a number $x$ is $p_x = \frac{1}{n}$. This is a **Probability Distribution**[^admit].

[^admit] This is admittedly a very boring probability distribution, because it is flat (Uniform), with every value having the same probability. 

There are two kinds of Probability Distributions:
1. Probability Mass Functions (PMFs), for discrete RVs
2. Probability Density Functions (PDFs), for continuous RVs


::::{tab-set}
:::{tab-item} PMFs

The score from rolling a die (or any number of dice) is a Discrete RV. The probability of measuring a given value $k$ for a Discrete RV $K$ is described by a **Probability Mass Function (PMF)** $p_K(k)$.

The **Support** S of a PMF is the set of all values with a non-zero probability of occurring.

A PMF must satisfy the [Kolmogorov Axioms](#kolmogorov), which we usually write down in a slightly different format when talking about PMFs:

$$\label{eq:kolmogorov1pmf} p_K(k) > 0\;\;\; \forall\;\; k \in S $$

$$\label{eq:kolmogorov2pmf} \sum\limits_{k\in S} p_K(k) =1 $$

$$\label{eq:kolmogorov3pmf} P(k \in A) = \sum\limits_{k\in A} p_K(k) $$

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np

# randint is a uniform distribution of integers
from scipy.stats import randint

lo=1
hi=6

# the upper limit of a range is the maximum value **plus 1**
k = np.arange(lo, hi+1 ) 

# theoretical discrete uniform distribution of dice roll probabilities
dist = randint(low=lo, high=hi+1) 

# we can ask the distribution for its support
support = dist.support() 

print(f"dist.support(): {dist.support()}") 

# we can ask the distribution for its PMF
pmf = dist.pmf(k) 

# Plot the score values on the x-axis and the corresponding PMF values on the y-axis
plt.stem(k, pmf)

plt.xlabel("Dice Score")
plt.ylabel("Probability")
plt.xlim(0,7)
plt.ylim(0)
plt.show()

```


:::
:::{tab-item} PDFs

The probability of measuring a given value $x$ for a Continuous RV $X$ is described by a **Probability Density Function (PDF)** $f_X(x)$.

As per a PMF, the **Support** S of a PDF is the set of all values with a non-zero probability of occurring.

 A PDF must also satisfy the [Kolmogorov Axioms](#kolmogorov), but we now write the second and third conditions as integrals:

$$\label{eq:kolmogorov1pdf} f_X(x) > 0\;\;\; \forall\;\; x \in S $$

$$\label{eq:kolmogorov2pdf} \int\limits_{S} f_X(x) dx =1 $$

$$\label{eq:kolmogorov3pdf} P(x \in A) = \int\limits_{A} f_X(x) dx $$

```{important} Density
Notice that we are not labelling the y-axis as **Density** rather than Probability for the PDF. This is because the probability of measuring any single value for "X" is zero. This can seem a bit odd; it is a consequence of Continuous RVs having an uncountable infinity of possible values, so the only way such an RV can satisfy [Kolmogorov 2](#eq:kolmogorov2pdf) is to demand the probability of any exact value is zero.
```

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np

# norm is a distribution describing a continuous RV
from scipy.stats import norm

# theoretical normal distribution of some continuous RV (this is "standard normal", with default parameters mean=0 and standard deviation=1)

dist = norm() 

# we can ask the distribution for its support
support = dist.support() 

# note that norm has a support of -infty, infty! We will come back to this
print(f"dist.support(): {dist.support()}") 

# set a range of x values for the plot. 100 values is enough for a smooth plot.
x = np.linspace(-5,5,100)

# we can ask the distribution for its PDF
pdf = dist.pdf(x) 

# Plot the RV on the x-axis and the corresponding PDF values on the y-axis
plt.plot(x, PDF)

plt.xlabel("X")
plt.ylabel("Density")
plt.xlim(-5,5)
plt.ylim(0)
plt.show()

```
:::
::::







## Learning Objectives Checklist

- [ ] Summarise the Frequentist and Bayesian interpretations of probability
- [ ] Understand and express relationships between sets ($\subset$, $\cap$, $\cup$, $A'$)
- [ ] Correctly interpret Venn diagrams
- [ ] Calculate the Union and Intersection of sets
- [ ] Describe the meaning of Independent and Mutually Exclusive (Disjoint) sets
- [ ] Use the formula for calculating the Conditional Probability
- [ ] Describe the terms in Bayes' Theorem

