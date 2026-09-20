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

**Fernanda**: my keys are in an unknown location in my home. There is no "probability of them being by the sink", because they are either in a place or not. The only way to locate them is to look in every location methodically.

**Betty**: my keys are in an unknown location in my home. On the last ten occasions I lost them, they were by the sink three times, so there are most likely to be by the sink and I will start my search there.


## Example Sets

Example Sets:

$$
:label: our-sets
\begin{aligned}
S &= \{1,2,3,4,5,6,7,8,9,10\} & \\
A &= \{2,4,6,8,10\}& \\
B &= \{1,2,3,4\}& \\
\end{aligned}
$$


## Frequentist Probability


The **Frequentist Probability** of A is written $P(A) = \dfrac{N_A}{N_S}$: the number of times A occurs in the sample, divided by the number of possible outcomes
in the sample.

Probabilities for our sets [](#our-sets):
* $P(S) = N_S/N_S =1$
* $P(A) = N_A / N_S  = 0.5$
* $P(B) = N_B / N_S  = 0.4$


## Special Sets
:::{figure} https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/SpecialSets.png
:label: fig:special-sets

Some special sets.
:::


## Sample Space & Subsets

The **Sample Space** S is the set of all possible outcomes of some experiment or operation.

A **Proper Subset** of S is denoted $A\subset S$. This means that A is a subset of S but A$\neq$S. If A is a subset of S *and* can be S, we use $A\subseteq S$

For [](#our-sets), A and B are proper subsets of S.


## Venn Diagrams

Venn diagrams are helpful for visualising relationships. Here $A\subset S$ and $B\subset S$.

::::{grid} 1 1 3 3

:::{image} /figures/S.png
:::

:::{image} /figures/A.png
:::

:::{image} /figures/B.png
:::

Venn diagrams showing datasets S, A, and B.
::::


## Complement $A'$

The **Complement** of a set is denoted by a prime, $A'$; this means 'not A'. Other notations commonly used for the complement are $A^{\complement}$, $\overline{A}$, and others.

::::{grid} 1 1 2 2

:::{image} /figures/NotA.png
:::

:::{image} /figures/NotB.png
:::

Venn diagrams showing datasets The Complement of A and B.
::::



### Union $A\cup B$

The **Union** of two sets is written $A\cup B$; this means 'either A, or B, or both'. For our sets [](#our-sets), $A\cup B = \{1,2,3,4,6,8,10\}$

## Intersection $A\cap B$

The **Intersection** of two sets is written $A\cap B$; this means 'both A and B'. For our sets [](#our-sets), $A\cap B = \{2,4\}$


::::{grid} 1 1 2 2

:::{image} /figures/AorB.png
:::

:::{image} /figures/AandB.png
:::

Venn diagram illustrating left: the Union and right: the Intersection of the sets A and B.
::::

## Independence and Mutual Exclusivity

Two sets are **Independent** if they are defined without reference to one another. Changing one does not impact the other. We can construct **dependent** sets like this for example:
* $K = {\mathbb{N}}$
* $J = {K^2}$

Two sets are **Mutually Exclusive** if there is no overlap between them. Another word for this is **Disjoint**.


## Probability of Intersection

* The **Probability of Intersection** $P(A\cap B)$ is the probability that **both A and B** are true. 
* This is zero when A and B are **Disjoint** (no overlap: **Mutually Exclusive**)
* $P(A\cap B) =0$ when $(A\cap B)'$

For our sets [](#our-sets):
* $A\cap B = \{2,4\}$
* $P(A\cap B) = 0.2$
* $A$ and $B$ are not mutually exclusive.


::::{grid} 1 1 2 2

:::{image} /figures/NotAorB.png
:::

:::{image} /figures/NotAandB.png
:::

Venn diagram illustrating the complement of left: the Union and right: the Intersection of the sets A and B.
::::



## Probability of Union

The **Probability of Union** $P(A\cup B)$ is the probability that either A or B or both are true. 

$$\label{eq:prob-union} P(A\cup B) =P(A)+P(B) - P(A\cap B)$$

Note that the subtraction of the intersection $P(A\cap B)$ in [](#eq:prob-union) is to remove the double counting of that intersection, as illustrated in [](#fig:axiom3).

For our sets [](#our-sets): $P(A\cup B) = 0.5 +0.4 - 0.2 = 0.7$

:::{figure} 
:label: fig:venn_notAandB
:align: left
![](https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/Axiom3.png)
:::

## The Kolmogorov Axioms

> [Andrey Kolmogorov](https://en.wikipedia.org/wiki/Andrey_Kolmogorov) was a Russian mathematician. You may have heard of the 'KS test', which is used to eg check for overtraining by comparing ML classifier outputs for test and train samples. This is named for Kolomogorov and Nikolai Smirnov. Also 'KANs' - Kolmogorov Arnold Networks - see [arXiv:2404.19756](https://arxiv.org/abs/2404.19756).

1. The **Non-Negativity Axiom**
: for all events A , $P(A) \geq 0$: 'the probability of any event A must be a real number greater than zero.\
2. The **Normalisation Axiom**
$P(S) =1$: the probability of the entire sample space is one.\
3. The **Countable Additivity Axiom**
: if A and B are **mutually exclusive**, $P(A\cup B) =P(A)+P(B)$: the
probability of their union is the sum of their individual
probabilities.


## Conditional Probability $P(A | B)$

The **Conditional Probability** is most usefully written:

$$\label{eq:conp} P(A | B) = \dfrac{P(A\cap B)}{P(B)}$$

The notation $ P(A | B) $ means 'the probability of A, given that B is true'. The condition is that B is true.

> When we define a function of some variables and parameters as $f(x,y;\theta)$, the semicolon ; is used to indicate the conditional 'pipe' |. The function has variables x and y, and is conditional on the parameter $\theta$.

The number of elements in the intersection is $N(A \cap B)$.

The proportion of A in B is
$\dfrac{N(A\cap B)}{N(B)}$

Relative to the original Sample Space S this is:
$\dfrac{N(A\cap B)/N(S) }{N(B)/N(S)}$

These are **probabilities**:
$\dfrac{P(A\cap B)}{P(B)}$

Given that we demand the element must exist in B, this is the
probability of finding it in A.


For our sets, [](#our-sets):
* $S = \{1,2,3,4,5,6,7,8,9,10\}$
* $A = \{2,4,6,8,10\}$
* $B = \{1,2,3,4\}$
* $A\cap B = \{2,4\}$: Events in both A and B
* $N(A \cap B) = 2$: Count of events in both A and B
* $\dfrac{N(A\cap B)}{N(B)} = \dfrac{2}{4}$: Fraction of B that is also in A
* $P(A | B)  = 0.5 $: Probability of A given that we demand B.


If A and B are **Independent**, the conditional probability is $P(A | B)  = P(A)$.

This is because independence means that B has no effect on A and vice
versa.

Our example sets are independent. $P(A| B) = P(A) = 0.5$.


It is common to use conditional probabilities when we have a set of outcomes (A) and a set of choices (B). The **Conditional Probability** then gives us an answers to the question
> If I choose B, what is the probability of the outcome A?

Conditional Probability looks harmless enough, but can have some counter-intuitive results illustrated very well in the **Monty Hall Problem**.



## The Multiplication Rule

Rearranging [](eq:conp): $P(A\cap B) = P(A | B) P(B)$ and noting that for **Independent** A and B, $P(A | B)=P(A)$, we get the very useful **Multiplication Rule**:

**If A and B are independent**, the probability of the intersection of A and B is equal to the product of their individual probabilities: 

$$\label{eq:multrule} P(A \cap B) = P(A) P(B)$$


Example: I roll a dice twice. What is the probability I will get two sixes?

$P(six \cap six) = P(six) P(six) = \dfrac{1}{6} \dfrac{1}{6}  = \dfrac{1}{36}$


## Total Probability

In [](#fig:venn_tot) I have divided S into four quadrants, each of which is a set $B_i$. There are N=4 disjoint sets $B_i, i=1,2,3,4$ intersecting with A. 

The Total Probability of A can be written $\label{eq:totp1} P(A) = \sum \limits_{i=1}^4 P(A\cap B_i)$.


:::{figure} 
:label: fig:venn_tot
:align: left
![](https://github.com/dorksquith/DataAnalysisTechniques/blob/main/figures/AandB1.png)
:::

Because we know how to write the intersection in terms of the conditional probability ([](eq:conp)), we can express the Total Probability of A as [](eq:totp).

$$\label{eq:totp} P(A) = \sum \limits_i^N P(A | B_i) P(B_i)$$.\

This may seem a bit contrived, but we will see soon that it is
useful.


## Bayes' Theorem

We have the **Conditional Probability** $P(A\cap B) = P(A | B) P(B)$, and this applies to any sets A and B, so we can also write $P(B\cap A) = P(B | A) P(A)$.

The intersection is not directional, so $P(B\cap A) \equiv  P(A\cap B)$, and if the left hand sides of the two forms are equivalent, the right hand sides are also equivalent:

$ P(A | B) P(B) = P(B | A) P(A)$

This is Bayes' theorem, more usually rearranged as:

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



