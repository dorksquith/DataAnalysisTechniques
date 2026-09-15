---
author:
- Dr Lily Asquith (Lily, she/her)
subtitle: Introduction
title: Probability & Statistics
---

::: frame
:::

::: frame
Introduction

- People

- What are Probability & Statistics?\

- Why are they useful?\

- What will we learn here?\
:::

::: frame
People

I am Dr Lily Asquith, please call me Lily.\
I am an Experimental Particle Physicist working on Neutrinos (NOvA and
DUNE experiments).\
Who are you? **[pollev.com/ilovephysics]{style="color: red"}**\
![image](pollev1)
:::

# Probability & Statistics  {#probability-statistics .unnumbered}

::: frame
Names of this Module This module covers Probability (maths) and
Statistics (philosophy? art?) with the goal of Data Analysis.\
It has two names (ooh, it is Binomial!):\

- Statistical Analysis & Probability, as delivered to MSc AI cohort.\

- Data Analysis Techniques, as delivered to everyone else.\

Other than the title, they are exactly the same.\
:::

::: frame
Side note on the term Statistics

Statistic(s) is a heavily overloaded term. What I mean when I say these
words:\
[A **statistic** is a value of a property of the data, and a
**statistic** is also the function that provides the value. For example
the mean of set of measurements.]{style="color: dblu"}\
**Statistics** is a set of methods for analysing data, often with the
aim of providing probabilities within a philosophical framework.\
[Lower case **statistics**: plural of statistic.]{style="color: dblu"}\
:::

::: frame
Statistics in this module We use models to make predictions. Our models
must accurately describe real data if they are to work effectively.\
To build a working model based on data, we need to Estimate the form of
the Underlying Distribution describing our data.\
:::

::: frame
Uncertainties Measurements are meaningless without uncertainties.\
:::

::: frame
Probability

Probability is very important to human beings. We are subject to random
events in nature that can completely change the course of our lives,
leading to world domination or total annihilation.\
It is not surprising that games of chance are so popular, and have
probably been around much longer than eg written communication.\
:::

::: frame
Example Example: I am sitting in a neolithic bar, tossing a fair coin.\
I toss the coin three times and get H H H.\
I bet you ten eggs that the next toss will be tails. How many eggs will
you put up against my bet?\
Understanding the essentials of probability allows the Neolithic Lily to
make money off people who do not.\
Make your bets! **[pollev.com/ilovephysics]{style="color: red"}**\
:::

# Usefulness {#usefulness .unnumbered}

:::: frame
Why are Prob & Stats useful?

::: multicols
2 1. Win games[\*]{style="color: red"}\
2. Interpret Data $\rightarrow$ Understand Universe better.\
**Note that**:\
All Scientific results are statistics (usually frequentist, but eg
particle physics also uses Bayesian).\
Machine Learning = Statistics.\
![image](ml-is-stats)
:::

[\*]{style="color: red"} Requires an absence of knowledge in
opponent(s). Not applicable to casinos.
::::

:::: frame
Tools for harm?

::: multicols
2

Statistics and Probability are frequently used as tools for harm by
characters such as Neolithic Lily.\
The only solution to this is good education, but that is notoriously
difficult.\
Perceived failures to achieve this are harshly punished by the community
(eg [this mean book
review](https://cerncourier.com/a/statistical-data-analysis-for-the-physical-sciences/)
) but as far as I can tell, these punishments have not resulted in an
obvious improvement in teaching materials generally.\
![image](roast-crop)
:::
::::

::: frame
Issues we will face

The main barriers to decent teaching, from my very fresh perspective,
and **in my humble opinion**, are:\

- Aesthetics (consistent notation and readability are not friends)\

- Boredom ('shortcut' errors everywhere, including textbooks)\

- Ignorance (lots of this stuff is counterintuitive, and 'long')\

- Fear (to avoid errors and scathing reviews, people are vague)\

- Cognitive bias (assumption of highly specific prior knowledge makes
  the good stuff inaccessible to almost everyone)\
:::

# Topics {#topics .unnumbered}

:::: frame
What will we learn here?

We will start with Probability. Two philosophies: Frequentist and
Bayesian.\

::: center
![image](orloff-bloom-probstat)

[pic credit: Orloof &
Bloom](https://math.mit.edu/~dav/05.dir/class17-prep-a.pdf)
:::
::::

::: frame
Prerequisites

The topics we cover all involve math and I use python (marimo notebooks)
to make them less long, boring, and confusing, and more useful. Please
see canvas Resources & Prerequisites.\
In a nutshell, we need partial derivatives, chain and product rule,
integration by u-sub and parts, matrix algebra, Taylor series, log
rules.\
I will sneak a proof or two into lectures now and again, but you will
not find rigour in these slides.\
:::

::: frame
Tools

The tools needed to (efficiently) construct (and test) Estimators are
covered in weeks 2-7:\

- Random Variables\

- PDFs and likelihoods, transformations\

- Specific PDFs\

- Correlations & Uncertainties\

- Useful statistics\

- Random numbers and Monte Carlo\
:::

::: frame
Estimators Once we have a strong grasp of the tools, which are useful in
their own right, in weeks 8-11 we explore making estimates:\

- Maximum Likelihood & Least Squares (Frequentist)\

- Testing our Estimators\

- Bayesian Inference\

- Resampling\
:::

::: frame
Teaching & Assessment We will have 2 lectures and 1
workshop[\*]{style="color: red"} each week.\
*Unless there are substantial elements of revision for you, it is going
to be a lot of work as the weeks go on. Don't stress over it, learning
anything new takes time.*\
The 3 problem sets will be available 2 weeks before their due dates\
**Please read the canvas pages Module Information and the Assignment &
Guidance** (re-read the latter before you prepare each submission).\
The software exercise is to be completed between the last week of term
and the first week of the exams period.\
[\* No workshop in week 1 or week 11]{style="color: red"}.
:::

::: frame
Python

Python is beautiful. It allows us to do complicated things without
hurting (exercising?) our brains.\
I will provide a [marimo](https://docs.marimo.io/getting_started/)
notebook for each week with python examples of the concepts we are
wrapping our heads around. Please Install marimo as per instructions on
the canvas Resources & Prerequisites page:

``` {.bash bgcolor="LightGray"}
pip install marimo
```

It is totally fine if you want to bring your laptop to lectures and do
the python while we go through the concepts.\
If you are prone to going down coding rabbit holes it is also fine to
leave the python until later so that you can pay attention.\
:::

::: frame
Thanks For Your Time Fin.
:::
