# Probability 1


Probability is very important to human beings. We are subject to random
events in nature that can completely change the course of our lives,
leading to world domination or total annihilation.
It is not surprising that games of chance are so popular, and have
probably been around much longer than eg written communication.

## Two Philosophies

There are two ways of viewing probabilities: Frequentist and Bayesian. In nutshell, the Bayesian intepretation involves the introduction of prior knowledge, whereas the frequentist way is to rely purely on the data.

A Frequentist, Fernanda, and a Bayesian, Betty, have each misplaced their keys. They will each adopt a different method for finding them based on their different philosophies of probability.

:::{figure} https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/FernandaAndBetty.jpeg
:label: fig:fernanda-and-betty

Fernanda and Betty.
:::

Fernanda: my keys are in an unknown location in my home. There is no "probability of them being by the sink", because they are either in a place or not. The only way to locate them is to look in every location methodically.

Betty: my keys are in an unknown location in my home. On the last ten occasions I lost them, they were by the sink three times, so there are most likely to be by the sink and I will start my search there.


## Notation

Example Sets:

$$
:label: our-sets
\begin{flalign}
S = \{1,2,3,4,5,6,7,8,9,10\}
A = \{2,4,6,8,10\}
B = \{1,2,3,4\}
\end{flalign}
$$


### Frequentist Probability


The **Frequentist Probability** of A is written $P(A) = \dfrac{N_A}{N_S}$: the number of times A occurs in the sample, divided by the number of possible outcomes
in the sample.

Probabilities for our sets [](#our-sets):
* $P(S) = N_S/N_S =1$
* $P(A) = N_A / N_S  = 0.5$
* $P(B) = N_B / N_S  = 0.4$


### Special Sets
:::{figure} https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/SpecialSets.png
:label: fig:special-sets

Some special sets.
:::


### Sample Space & Subsets

The **Sample Space** S is the set of all possible outcomes of some experiment or operation.

A **Proper Subset** of S is denoted $A\subset S$. This means that A is a subset of S but A$\neq$S. If A is a subset of S *and* can be S, we use $A\subseteq S$

For [](#our-sets), A and B are proper subsets of S.


### Venn Diagrams

Venn diagrams are helpful for visualising relationships. Here $A\subset S$ and $B\subset S$.

:::{figure} https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/SAB.png
:label: fig:sab

Venn diagrams showing datasets S, A, and B.
:::

## Test Grid

::::{grid} 1 1 2 2

:::{image} /figures/A.png
:::

:::{image} /figures/B.png
:::

::::




## Test 1

:::{figure} :label: my-figure :class: grid grid-cols-2 items-end gap-4

![First image description](/figures/A.png)
![Second image description](/figures/B.png)

Main caption for both figures.
:::


## Test 2

:::{figure} :label: my-figure :class: grid grid-cols-2 items-end 

![First image description](/figures/A.png)
![Second image description](/figures/B.png)

Main caption for both figures.
:::



### Complement

The **Complement** of a set is denoted by a prime, $A'$
: this means 'not A'.\

Other notations commonly used for the complement are
:
$A^{\complement}$, $\overline{A}$, and several others.

![image](NotA.png) ![image](NotB.png)


:::{figure} fig:venn_complement
:align: center

```{image} https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/NotA.png
:width=45%
```
```{image} https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/NotB.png
:width=45%
```

Venn diagrams showing datasets The Complement of A and B.
:::



### Union

The **Union** of two sets is written:
$A\cup B$
: 'either A, or B, or both'.\


* $A = \{2,4,6,8,10\}$
* $B = \{1,2,3,4\}$
* $A\cup B = \{1,2,3,4,6,8,10\}$

:::{figure} 
:label: fig:venn_union
:align: left
![](https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/AorB.png)
:::


### Intersection

The **Intersection** of two sets is written $A\cap B$
: 'both A and B'.


* $A = \{2,4,6,8,10\}$
* $B = \{1,2,3,4\}$
* $A\cap B = \{2,4\}$

:::{figure} 
:label: fig:venn_intersection
:align: left

![](https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/AandB.png)

:::


## Axioms

* The **Probability of Intersection** $P(A\cap B)$ is the probability that **both A and B** are true. 
* This is zero when A and B are **Disjoint** (no overlap: **Mutually Exclusive**)
* $P(A\cap B) =0$ when $(A\cap B)'$

For our sets [](#our-sets):
* $A\cap B = \{2,4\}$
* $P(A\cap B) = 0.2$
* $A$ and $B$ are not mutually exclusive.

:::{figure} 
:label: fig:venn_notAandB
:align: left
![](https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/NotAandB.png)

Venn diagram illustrating the complement of both A and B.
:::


### The **Probability of Union** $P(A\cup B)$
:
the probability that either A or B or both are true. 

$$P(A\cup B) =P(A)+P(B) - P(A\cap B)$$

Our sets: $P(A\cup B) = 0.5 +0.4 - 0.2 = 0.7$

:::{figure} 
:label: fig:venn_notAandB
:align: left
![](https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/Axiom3.png)
:::

### The Kolmogorov Axioms

1. The **Non-Negativity Axiom**
: for all events A , $P(A) \geq 0$: 'the probability of any event A must be a real number greater than zero.\
2. The **Normalisation Axiom**
$P(S) =1$: the probability of the entire sample space is one.\
3. The **Countable Additivity Axiom**
: if A and B are mutually exclusive, $P(A\cup B) =P(A)+P(B)$: the
probability of their union is the sum of their individual
probabilities.


## Conditional Probability 

The **Conditional Probability**
$$\label{eq:conp} P(A | B) = \dfrac{P(A\cap B)}{P(B)}$$


Example:\
$A = \{2,4,6,8,10\}$
$B = \{1,2,3,4\}$
$A\cap B = \{2,4\}$

The number of elements in A and B is $N(A \cap B)$

The proportion of A in B is
$\dfrac{N(A\cap B)}{N(B)}$

Relative to the original Sample Space S this is:
$\dfrac{N(A\cap B)/N(S) }{N(B)/N(S)}$

These are **probabilities**:
$\dfrac{P(A\cap B)}{P(B)}$

Given that we demand the element must exist in B, this is the
probability of finding it in A.

If A and B are **Independent**, the conditional probability
$P(A | B)  = P(A)$.

This is because independence means that B has no effect on A and vice
versa.

Our example sets are independent. $P(A| B) = P(A) = 0.5$.


Using the Conditional Probability It is common to use conditional
probabilities when we have a set of outcomes (A) and a set of choices
(B).

The Conditional Probability then gives us an answers to the question
: If I choose B, what is the probability of the outcome A?

Conditional Probability looks harmless enough, but can have some
counter-intuitive results illustrated very well in the **Monty Hall
Problem**.


## The Monty Hall Problem

TBD


## The Multiplication Rule

Rearrange the **Conditional Probability**
: $P(A | B) = \dfrac{P(A\cap B)}{P(B)}\;\; \therefore\;\;  P(A\cap B) = P(A | B) P(B)$

**The Multiplication Rule**

**If A and B are independent**, the probability
of the intersection of A and B is equal to the product of their
individual probabilities: $P(A\cap B) = P(A) P(B)$.

:::{tip}
Example: I roll a dice twice and get two sixes:

$P(six \cap six) = P(six) P(six) = \dfrac{1}{6} \dfrac{1}{6}  = \dfrac{1}{36}$
:::

## The Total Probability

In [](#eq:totp1) there are N=4 disjoint sets $B_i$ intersecting with A, we can see that 

$$\label{eq:totp1} P(A) = \sum \limits_i^4 P(A\cap B_i)$$.


:::{figure} 
:label: fig:venn_tot
:align: left
![](https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/AandB1.png)
:::

Comparing [](#eq:totp1) with [](#eq:conp), we can write the **Law of Total Probability**:

$$\label{eq:totp} P(A) = \sum \limits_i^N P(A | B_i) P(B_i)$$.\

This may seem a bit contrived, but we will see soon that it is
useful.


# Bayes' Theorem

We have the **Conditional Probability**:\

$$P(A\cap B) = P(A | B) P(B)$$, therefore
$$P(B\cap A) = P(B | A) P(A)$$\

**Bayes' Theorem** simply combines these:\

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


On the left we have P of A given B, and on the right P of B given
A.


Bayes' Theorem for our sets

$$
\label{bayes-oursets}
\begin{aligned}
P(A | B) = 
\dfrac{
P(B|A)P(A)
}
{
P(B)
}
P(A| B) = P(A) = 0.5
P(B| A) = P(B) = 0.4
\end{aligned}
$$

:::{tip}
Bayes' Theorem gets interesting when we consider that it applies to
**any sets $A$ and $B$ that satisfy Kolmogorov's Axioms**.





## Refs
:::{note}
This is note.
:::

This is an equation:

:::{math}
:name: eq:book

x \times y = z
:::

Here is a picture: [](#fig:special-sets) 

:::{note} Click Me! 👈
:class: dropdown
👋 This could be a solution to a problem or contain other detailed explanations.
:::


```{code-cell} python
hello = "hello"
there = "there"
phrase = f"{hello}, {there}!"
print(phrase)
```


::::{grid}
:gutter: 2

:::{grid-item}
:outline:
A
:::
:::{grid-item}
:outline:
B
:::
:::{grid-item}
:outline:
C
:::
:::{grid-item}
:outline:
D
:::

::::

Cards:
<!-- https://jupyter-book.readthedocs.io/v1/content/components.html#content-cards -->

::::{grid}
:gutter: 3

:::{grid-item-card} One!
Here's the first card.
:::

:::{grid-item-card} Two!
Here's the second card.
:::

:::{grid-item-card} Three!
Here's the third card.
:::
::::


Tabs:

````{tab-set}
```{tab-item} Tab 1 title
My first tab
```

```{tab-item} Tab 2 title
My second tab with `some code`!
```
````