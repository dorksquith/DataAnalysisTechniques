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


### Where are my keys?

A Frequentist, Fernanda, and a Bayesian, Betty, have each misplaced their keys. They will each adopt a different method for finding them based on their different philosophies of probability.

:::{figure} /figures/FernandaAndBetty.jpeg
:label: fig:fernanda-and-betty

Fernanda and Betty.
:::

**Fernanda**: my keys are in an unknown location in my home. There is no "probability of them being by the sink", because they are either in a place or not. The only way to locate them is to look in every location methodically.

**Betty**: my keys are in an unknown location in my home. On the last ten occasions I lost them, they were by the sink three times, so there are most likely to be by the sink and I will start my search there.


### Is my coin fair?

Fernanda: would toss the coin a million times and see how many times we get heads.

Betty: would additionally consider what we know about the coin. Did we get it from a joke shop, or a post office?

### Is a pregnancy test correct?

Fernanda would do the test a million times and see how many times were correct.

Betty would also consider what we know about the person taking the test, for example: does the person taking the test have a uterus?

 
> It may appear that the Bayesian approach is prone to bias, and less scientific, so I should note that the Bayesian does not "stick to their guns" on the prior knowledge aspect. If the data indicate their prior knowledge is unlikely to be correct, then the prior is updated. 



## Example Sets

Example Sets:

```{math}
:label: our-sets

\begin{aligned}
S &= \{1,2,3,4,5,6,7,8,9,10\} & \\
A &= \{2,4,6,8,10\}& \\
B &= \{1,2,3,4\}& \\
\end{aligned}
```


## Frequentist Probability


The **Frequentist Probability** of A is written $P(A) = \dfrac{N_A}{N_S}$: the number of times A occurs in the sample, divided by the number of possible outcomes
in the sample.

Probabilities for [our sets](#our-sets):
* $P(S) = N_S/N_S =1$
* $P(A) = N_A / N_S  = 0.5$
* $P(B) = N_B / N_S  = 0.4$


## Special Sets
:::{figure} /figures/SpecialSets.png
:label: fig:special-sets

Some special sets.
:::


## Sample Space & Subsets

The **Sample Space** S is the set of all possible outcomes of some experiment or operation.

A **Proper Subset** of S is denoted $A\subset S$. This means that A is a subset of S but A$\neq$S. If A is a subset of S *and* can be S, we use $A\subseteq S$

For [our sets](#our-sets), A and B are proper subsets of S.


## Venn Diagrams

Venn diagrams are helpful for visualising relationships. Here $A\subset S$ and $B\subset S$.

::::{grid} 1 1 3 3

:::{image} /figures/S.png
:::

:::{image} /figures/A.png
:::

:::{image} /figures/B.png
:::

::::


## Complement $A'$

The **Complement** of a set is denoted by a prime, $A'$; this means 'not A'. Other notations commonly used for the complement include $A^{\complement}$ and $\overline{A}$.

::::{grid} 1 1 2 2

:::{image} /figures/NotA.png
:::

:::{image} /figures/NotB.png
:::

::::



### Union $A\cup B$

The **Union** of two sets is written $A\cup B$; this means 'either A, or B, or both'. For [our sets](#our-sets), $A\cup B = \{1,2,3,4,6,8,10\}$

## Intersection $A\cap B$

The **Intersection** of two sets is written $A\cap B$; this means 'both A and B'. For our sets [](#our-sets), $A\cap B = \{2,4\}$


::::{grid} 1 1 2 2

:::{image} /figures/AorB.png
:::

:::{image} /figures/AandB.png
:::

::::


The **Complement of the Union** is $(A\cup B)'$ and means 'Not in A and not in B (and not in both)'.


The **Complement of the Intersection** is $(A\cap B)'$ and means 'Not in both A and B'.


::::{grid} 1 1 2 2

:::{image} /figures/NotAorB.png
:::

:::{image} /figures/NotAandB.png
:::

::::



## Independence and Mutual Exclusivity

Two sets are **Independent** if they are defined without reference to one another (changing one does not impact the other). We could construct **dependent** sets like this for example:
* $K = {\mathbb{N}}$
* $J = {K^2}$

Two sets are **Mutually Exclusive** if there is no overlap between them. Another word for this is **Disjoint**.


## Probability of Intersection (Joint Probability)

The **Probability of Intersection** $P(A\cap B)$ is the probability that **both A and B** are true. This is also called the **Joint Probability** of A and B.

The Joint Probability is zero when A and B are **Disjoint** (no overlap: **Mutually Exclusive**):

$P(A\cap B) =0$ when $(A\cap B)'$

For [our sets](#our-sets):
* The intersection is $A\cap B = \{2,4\}$
* The Joint probability is $P(A\cap B) = 0.2$
* $A$ and $B$ are not mutually exclusive.


## Probability of Union

The **Probability of Union** $P(A\cup B)$ is the probability that either A or B or both are true. 

$$\label{eq:prob-union} P(A\cup B) =P(A)+P(B) - P(A\cap B)$$

Note that the subtraction of the intersection $P(A\cap B)$ in [](#eq:prob-union) is to remove the double counting of that intersection, as illustrated in [](#fig:venn_axiom3).

:::{figure} 
:label: fig:venn_axiom3
:align: left
![](/figures/Axiom3.png)
:::

For [our sets](#our-sets): $P(A\cup B) = 0.5 +0.4 - 0.2 = 0.7$


## The Kolmogorov Axioms

> [Andrey Kolmogorov](https://en.wikipedia.org/wiki/Andrey_Kolmogorov) was a Russian mathematician. You may have heard of the 'KS test', which is used to eg check for overtraining by comparing ML classifier outputs for test and train samples. This is named for Kolomogorov and Nikolai Smirnov. See also 'KANs' - Kolmogorov Arnold Networks [arXiv:2404.19756](https://arxiv.org/abs/2404.19756).

1. The **Non-Negativity Axiom**
: for all events A , $P(A) \geq 0$: 'the probability of any event A must be a real number greater than zero.
2. The **Normalisation Axiom** $P(S) =1$: the probability of the entire sample space is one.
3. The **Countable Additivity Axiom**
: if A and B are **mutually exclusive**, $P(A\cup B) =P(A)+P(B)$: the probability of their union is the sum of their individual probabilities.


## Conditional Probability $P(A | B)$

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

* $A\cap B = \{2,4\}$: Events in both A and B
* $N(A \cap B) = 2$: Count of events in both A and B
* $\dfrac{N(A\cap B)}{N(B)} = \dfrac{2}{4}$: Fraction of B that is also in A
* $P(A | B)  = 0.5 $: Probability of A given that we demand B.


If A and B are **Independent**, the conditional probability is $P(A | B)  = P(A)$.

This is because independence means that B has no effect on A and vice
versa.

Our example sets are independent: $P(A| B) = P(A) = 0.5$.


Conditional Probability looks harmless enough, but can have some counter-intuitive results illustrated very well in the **Monty Hall Problem**.



## The Multiplication Rule

Rearranging [](eq:conp): $P(A\cap B) = P(A | B) P(B)$ and noting that for **Independent** A and B, $P(A | B)=P(A)$, we get the very useful **Multiplication Rule**:

**If A and B are independent**, the probability of the intersection of A and B is equal to the product of their individual probabilities: 

$$\label{eq:multrule} P(A \cap B) = P(A) P(B)$$


Example: I roll a dice twice. What is the probability I will get two sixes?

$P(six \cap six) = P(six) P(six) = \dfrac{1}{6} \dfrac{1}{6}  = \dfrac{1}{36}$


## Total Probability (Marginal Probability)

In [](#fig:venn_tot) I have divided S into four quadrants, each of which is a set $B_i$. There are N=4 disjoint sets $B_i$ intersecting with A. 

The **Total Probability** of A can be written $\label{eq:totp1} P(A) = \sum \limits_{i=1}^4 P(A\cap B_i)$.

The Total Probability P(A) is also referred to as the **Marginal Probability** of A. This is because we have "Marginalised Out" the probability of B, and are only considering A.

:::{figure} 
:label: fig:venn_tot
:align: left
![](https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/AandB1.png)
:::

Because we know how to write the intersection in terms of the conditional probability [](eq:conp), we can express the Total Probability of A as [](eq:totp).

$$\label{eq:totp} P(A) = \sum \limits_i^N P(A | B_i) P(B_i)$$.

This may seem a bit contrived, but we will see soon that it is
useful.


## Bayes' Theorem

To write down Bayes theorem we need only the **Conditional Probability** [](#eq:conp), and to observe that:

1. [](#eq:conp) applies to any sets A and B, so we can switch A and B and remain true:
$$\label{eq:conpB} P(B | A) = \dfrac{P(B\cap A)}{P(A)}$$

2. The intersection is not directional, so $P(B\cap A) \equiv  P(A\cap B)$: The LHS of [](#eq:conp) is equal to the LHS of [](#eq:conpB).

3. The RHS of [](#eq:conp) must equal the RHS of [](#eq:conpB), giving us Bayes' theorem: $ P(A | B) P(B) = P(B | A) P(A)$.

This is more usually rearranged as:

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

Injecting our independent example sets [](#our-sets) into Bayes' theorem is a little anticlimactic:

* $P(A | B) = P(A)$ because A and B are independent
* Bayes theorem tells us that $P(A) = \dfrac{P(B) P(A)}{P(B)}$ which is just basic algebra.

Bayes' Theorem gets interesting when we consider that it applies to **any sets $A$ and $B$ that satisfy Kolmogorov's Axioms**.



