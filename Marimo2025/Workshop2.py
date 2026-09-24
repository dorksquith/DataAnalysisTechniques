import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Data Analysis Techniques: Workshop Exercises

    ## Week 3

    ### **Exercise 3.1: Discrete Joint PMF and Likelihood [Easy]**

    The measurements of two discrete RVs $X,Y$  correspond to the values (heads,tails =1,0) from two coin tosses.

    a) Write down the Joint PMF for the pair of coin tosses, and calculate the probability of observing both coins as heads.

    b) In this example you will have used your prior knowledge of the probability of heads being 0.5 (a fair coin). What is the Likelihood that the coin is fair?

    c) Now assume the coin is not fair, and has a 0.8 probability of heads. Recalculate the Likelihood with this new hypothesis.

    d) Is it more likely that the coin is fair or unfair, given the data?

    ### **Exercise 3.2: Continuous CDF with regions of validity [Less easy]**

    The Uniform PDF has flat probability in a region between two bounds $x: [a,b]$ and zero probability outside that range:

    \begin{equation}
    f_X(x) =
    \begin{cases}
         0\; & x < a\\[1ex]
      \dfrac{1}{b-a}\; & a\leq x \leq b \\[2ex]
      0\; & x>b \\[2ex]
    \end{cases}
    \end{equation}

    **a) Evaluate the CDF $F_X$ for the three ranges in (1) above.**

    *Note that you will need a separate integral for each range, because $f_X(x)$ is not continuous at the boundaries $x=a$ and $x=b$. Also note that the CDF is cumulative, so you will have sums in all but the first region. Format here so you don't waste your precious time:*

    \[
    F_X(x) = P(X<x) =
    \begin{cases}
         \int\limits_{-\infty}^{a}\,f_X\,dx\; & x < a\\[1ex]
      \int\limits_{-\infty}^{a}\,f_X\,dx + \int\limits_{a}^{x}\,f_X\,dx  & a\leq x \leq b \\[2ex]
      \int\limits_{-\infty}^{a}\,f_X\,dx + \int\limits_{a}^{b}\,f_X\,dx + \int\limits_{b}^{x}\,f_X\,dx \; & x>b \\[2ex]
    \end{cases}
    \]

    **b) Show that the Uniform PDF described by (1) above is a valid PDF for a continuous RV $X$ in the range $-1\leq x \leq 1$.**

    *Valid $\equiv$ Normalised $\equiv$ Integral is one.*

    **c) Write down the PDF $f_X(x)$ and CDF $F_X(x)$ for $-1\leq x \leq 1$.**

    *Your answer will be a number for the PDF, and a function of $x$ for the CDF.*

    ### **Exercise 3.3: Change of Variables [Optional Extension]**

    Consider a new RV, $Y(X)=X^2$.

    **a) What is the range of values for $y$ with non-zero probability?** Is this the *support* or *sample space*?

    **b) Write down the CDF and PDF for $Y$.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Solution: 3.1. a)

    We assume a fair coin, with probability of heads 0.5.

    Discrete $X$ and $Y$,  $p_X = 0.5$ and $p_Y = 0.5$

    Joint PMF $p_{XY} = p_X p_Y =  0.5\cdot 0.5 = 0.25$ *because $X$ and $Y$ are mutually independent*.

    For discrete RV, the PMF gives the probability (unlike for continuous RV, where the PDF value is meaningless and we need the CDF).

    The probability of two heads, given that the coin is fair, is 0.25: $\mathsf{P(HH | fair)} = 0.25$.

    ### Solution: 3.1. b)

    Likelihood: no need to call the two tosses separate RVs, just let them be two measurements of the same RV.

    $L = \prod\limits_i^N \dfrac{1}{2} \dfrac{1}{2} = \dfrac{1}{4}$

    The Likelihood of a fair coin, given the data, is 0.25: $\mathsf{L(fair | HH)} = 0.25$.

    ### Solution: 3.1. c)

    Likelihood $\mathsf{L(p=0.8 | HH)} = 0.64$

    ### Solution: 3.1. d)

    Data suggests the coin is more likely to be biased to p=0.8 for heads than it is to be fair. Note that we have a very small dataset of $N=2$ measurements!
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Solution: 3.2 a)


    **For the region $x<a$**:$\;\;\;$ $F_X(x) = \int\limits_{-\infty}^{a}\,f_X\,dx = 0\;\;$ because (1) tells us that $f_X=0$ in this range.

    **For the region $a\leq x \leq b$**:$\;\;\;$ $F_X(x)=\int\limits_{-\infty}^{a}\,f_X\,dx + \int\limits_{a}^{x}\,f_X\,dx \;=\; 0 + \int\limits_{a}^{x} \dfrac{1}{b-a}\, dx = \dfrac{x}{b-a}\bigg|_{a}^{x} = \dfrac{x-a}{b-a}\;\;$


    **For the region $x > b$**:$\;\;\;$ $F_X(x) = \int\limits_{-\infty}^{a}\,f_X\,dx + \int\limits_{a}^{b}\,f_X\,dx+ \int\limits_{b}^{x}\,f_X\,dx\;=\; 0 \;+\; \dfrac{b-a}{b-a} \;+\; 0\; =\; 1$

    \[
    F_X(x) = P(X<x) =
    \begin{cases}
         0 & x < a\\[1ex]
       \dfrac{x-a}{b-a} & a\leq x \leq b \\[2ex]
      1 & x>b \\[2ex]
    \end{cases}
    \]



    ### Solution: 3.2 b)

    For $f_X$ to be valid in the specified range for which it is non-zero, we must show that it is normalised: $\int\limits_{a}^{b} f_X\, dx= 1$ for $a=-1$, $b=1$.

    $\int\limits_{a}^{b} \dfrac{1}{b-a}\, dx = \dfrac{x}{b-a}\bigg|_{a}^{b} = \dfrac{b}{b-a} - \dfrac{a}{b-a} = \dfrac{1}{1-(-1)} - \dfrac{-1}{1-(-1)}= \dfrac{1}{2} - \dfrac{-1}{2} = 1\;\;$ This PDF is valid.


    ### Solution: 3.2 c)

    $f_X(x) =  \dfrac{1}{b-a} = \dfrac{1}{1-(-1)} = \dfrac{1}{2}$

    $F_X(x) = \int\limits_{a=-1}^{x} f_X\,dx\; = \dfrac{x-a}{b-a} = \dfrac{x-(-1)}{1-(-1)} = \dfrac{x+1}{2}$


    ### Solution: 3.3)

    The range of non-zero probabilities for $X$ is $-1 \leq x \leq 1$, so the range for $Y=X^2\;$ is $\;0 \leq y \leq 1$, and $x = \pm \sqrt{y}$.

    This is the support. The sample space is the set of all values, including zero probability ones.

    **Easier to find the CDF first imho**

    $F_Y(y) = P(Y\leq y) = P(X^2 \leq y) = P(-\sqrt{y} \leq X \leq \sqrt{y}) = \int\limits_{-\sqrt{y}}^{\sqrt{y}} f_X(x) dx = \int\limits_{-\sqrt{y}}^{\sqrt{y}} \dfrac{1}{2} dx\,= \int\limits_{0}^{\sqrt{y}} dx = x\bigg|_{0}^{\sqrt{y}} = \sqrt{y}$

    Then the PDF, (which is the derivative of the CDF because the CDF is the integral of the PDF):

    $f_Y(y) = \dfrac{d}{dy} F_Y(y) = \dfrac{d}{dy} y^{\frac{1}{2}} = \dfrac{1}{2} y^{-\frac{1}{2}} = \dfrac{1}{2\sqrt{y}}$

    Normalisation check: $\int\limits_{0}^{1} f_Y(y)\, dy\,=\int\limits_{0}^{1} \dfrac{1}{2} y^{-\frac{1}{2}}\, dy = \dfrac{1}{4} y^{\frac{1}{2}}\bigg|_{0}^{1} = \dfrac{1}{4}$ : Need a renormalisation factor of 4 on transformed PDF.

    Renormalised $f_Y(y) = \dfrac{2}{\sqrt{y}}$



    **More general method (slightly more faff?) gives same result**

    The PDF $f_Y(y) = \dfrac{dx}{dy} f_X(x)$.

    For  $X(Y) = + Y^{\frac{1}{2}}$:  $\;\;\;\dfrac{dx}{dy} = \dfrac{1}{2}y^{-\frac{1}{2}}$

    For  $X(Y) = - Y^{\frac{1}{2}}$:  $\;\;\;\dfrac{dx}{dy} = -\dfrac{1}{2}y^{-\frac{1}{2}}$

    $f_Y(y) = \left| \dfrac{dx}{dy}  \right| \cdot f_X(x) = \dfrac{1}{2}y^{-\frac{1}{2}} \cdot \dfrac{1}{2} = \dfrac{1}{4\sqrt{y}}$.

    Normalisation check: $\int\limits_{0}^1 \dfrac{1}{4}y^{-\frac{1}{2}}  dy = \dfrac{1}{8}y^{\frac{1}{2}}\bigg|_0^1 = \dfrac{1}{8}$ : Need a renormalisation factor of 8 on transformed PDF.

    Renormalised $f_Y(y) = \dfrac{2}{\sqrt{y}}$ as before.

    $F_Y(y) = \int\limits_{0}^y f_Y(y) dy  = \int\limits_{0}^y 2y^{-\frac{1}{2}}\,dy\, =  y^{\frac{1}{2}}\bigg|_0^y= \sqrt{y}$ as before.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
