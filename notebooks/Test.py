import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Data Analysis Techniques: Workshop Exercises

    ## Week 2

    ### **Exercise 2.1: Bayes Theorem**

    You are given the following information:

    1. In random testing, someone tests positive for a disease.
    2. In 5% of cases where the subject does not have the disease, this test shows positive anyway: i.e. there is a 5% false positive rate
    3. There are no false negatives.
    4.  In the population at large, one person in a thousand has the disease.

    What is the probability that the person tested actually has the disease?

    **Solution can be pen and paper or python.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Exercise 2.2: Function Normalisation, Expected Value and Variance of Random Variable**
    A function of a Random Variable X is $f(X) = Ax^{2}$  in the range $x: [0,1]$.

    In order for this function to be used as a probability density, it must have integral=1 (total probability=1).

    a) Find the coefficient $A$ that normalises  $f_X$ in the range $x:[0,1]$

    b) Calculate the Expected Value E[X] using the normalised $f_X$

    c) Calculate the True Variance V[X]

    **Please present you solution using pen and paper, then check your solution using python.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Exercise 2.3: Weighted Mean**
    My good friend Alfred has become obsessed with weighing packets of rolos (this happens to him from time to time - he will be fine).

    This is his set of measurements for the weights of five different packs (in grams)

    $x_{A} = [52.010, 52.041, 52.105, 51.998, 51.981]$

    I decide to help him and make these measurements:

    $x_{L} =[52.05, 52.03, 52.07, 51.90, 51.94]$

    Calculate the weighted mean of these two datasets, and indicate which of them holds more weight and why that is.

    **I recommend you use python for this one**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Solutions
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Solution to Exercise 2.1

    $P(X|H_i) = P(+|sick) = 1\;\;$   (**there are no false negatives**)

    $P(H_i) = P(sick) =1/1000\;\;$   (**the prevalence of the disease is our prior**)

    $\sum\limits_i^N  P(X|H_i) P(H_i) = P(+|sick) P(sick) + P(+|not\_sick) P(not\_sick) = (1)(1/1000) + (0.05)(999/1000)$

    For the sum in the denominator we have two terms for the two possible hypotheses: sick or not sick. The alternate hypothesis term uses:

    - $P(+|not\_sick)=0.05\;\;$ (**the false positive rate is 5%)**
    - $P(not\_sick) = 1-P(sick)= 999/1000\;\;$ (**Total Probability is 1, so**  $P(H_i') = 1-P(H_i)\;$**)**

    Putting the numbers into $\textcolor{blue}{[1]}$, we have:
        $P(H_i|X) =\dfrac{(1)(1/1000) }{ (1)(1/1000) + (0.05)(999/1000) } =0.02 = 2\%$

    #### Insights

    1. The low posterior of 2% arises becuase the prior is 1/1000 = 0.001 = 0.1%

    2. There has to be a non-zero false positive rate for the posterior to not be 1

    3. **A Frequentist approach** to this question would be different: they would say the probability of you being positive if you test positive is 100%, the true positive rate. Frequentists reject the notion of priors with the argument that a hypothesis can only be right or wrong, not somewhere imbetween!

    4. The $P(X|H_i)$ is the **Likelihood** of the data given the hypothesis. It is **not a Probability** because Kolmogorov's axiom tells us that the total probability must be 1. That is not the case when we consider X: data and H: Hypothesis. This can be confirmed with a table of data versus hypothesis:

    |              | Postive | Negative | $\textcolor{darkgreen}{\sum P(H)}$
    |              |---------| ---------|---------|
    | **Sick**     | true postive=1 | false negative =0 | **$\textcolor{darkgreen}{1}$**
    | **Not sick** | false positive=0.05 | true negative=0.95 | **$\textcolor{darkgreen}{1}$**
    | **$\textcolor{red}{\sum P(X)}$**          | **$\textcolor{red}{1.05}$** | **$\textcolor{red}{0.95}$**  |

    The $P(H)$ (rows) sum to 1, but the $P(X)$ (columns) do not.
    """)
    return


@app.cell
def _():
    # W2.1 python solution
    def BayesPosterior( px_h1, px_h2, prior_h1):
        P_H1_X =  ( px_h1*prior_h1) / ( px_h1*prior_h1 + px_h2*(1-prior_h1) )
        return P_H1_X

    true_positive_rate = 1.0
    false_positive_rate =0.05
    prior = 1/1000.
    prob_sick_if_pos =BayesPosterior( true_positive_rate, false_positive_rate, prior)
    print("probability of being sick if you get a positive test: {:.3f}".format( prob_sick_if_pos) )
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Solutions to Exercise 2.2

    a) $\;\int \limits_0^1 f(x) dx =  \int \limits_0^1 Ax^2 dx
    =\left[\dfrac{A}{3}x^3\right]_0^1
    = \dfrac{A}{3}=1\;\;\mathsf{when}\;$ **$A=3$**.

    $f_X = 3x^2$ is a valid PDF.


    b) $\;E[X]
          = \int \limits_0^1 x\, (3x^2)\,dx
          = \left[\dfrac{3}{4}x^4\right]_0^1
          =\dfrac{3}{4}$

    **The expected value $E[X] = \dfrac{3}{4} = 7.50\times10^{-1}\;$**

    I have chosen 3SF as a sensible-seeming number of sig figs, and I am expressing in scientific notation to make it super-easy to cross-check with python - below.

    c) $\;V[X]  = \textcolor{darkgreen}{E[X^2]} - \textcolor{darkblue}{E^2[X]}$

    $\textcolor{darkgreen}{E[X^2]  = \int \limits_0^1 x^2\, (3x^2)\,dx
          = \left[\dfrac{3}{5}x^5\right]_0^1
          =\dfrac{3}{5}}$

    $\textcolor{darkblue}{E^2[X]  =\left(\dfrac{3}{4}\right)^2 = \dfrac{9}{16}}$

    **The variance $V[X] = \dfrac{3}{5} - \dfrac{9}{16} =3.75\times10^{-2}\;$**
    """)
    return


@app.cell
def _():
    #### Python solution to 2.2

    # Turn f_X into a PDF using the rv_continuous class from scipy.stats import rv_continuous
    from scipy.stats import rv_continuous

    class fx_pdf(rv_continuous):
        def _pdf(self,x):
            return 3*x**2  

    normed_fx = fx_pdf(a=0, b=1)
    # check the normalisation
    Prob1 = normed_fx.cdf(1)
    Prob0 = normed_fx.cdf(0)
    print("Normalisation check: CDF(x=1)-CDF(x=0 ) ={}".format(Prob1-Prob0) )
    expect = normed_fx.expect()                        
    var = normed_fx.var()
    print("Expected Value E[X]={:1.2e}".format( expect) )
    print("Variance V[X]={:1.2e}".format(var) )
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### **Intro**

    Bayes theorem is
        $P(H|X) = \dfrac{P(X|H) P(H)}{ P(X)}$

    - H is our hypothesis, in this case 'sick'
    - X is our data: the test results

    $P(X|H)$ is the true positive rate: probability of a positive test given you are sick. This is the definition of a true positive - they get a bunch of people who they know have the illness and test them for the illness.

    $P(H)$ is the prevalence of the illness. This is our **prior**.

    $P(X)$ is the total probability of the data, which we write:
    $P(X)=\sum\limits_i^N P(X|H_i) P(H_i)$,
    where the sum is over all possible hypotheses $H_i$.

    This gives us the more useful form of Bayes:
    $P(H_i|X) = \dfrac{P(X|H_i)P(H_i)}{\sum\limits_i^N P(X|H_i)P(H_i)}$  $\;\;\;\textcolor{blue}{[1]}$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The weighted mean is:

    $\overline{x}_w = \dfrac{\sum\limits_j^Z \overline{x}_{(j)} w_j}{\sum\limits_j^Z w_j }$,  $\;\;\;\textcolor{blue}{[2]}$

    where the weights are the reciprocal of the variance $w_j = \dfrac{1}{V[x_{(j)}]}$ and the sum is over the number of datasets $Z$.

    For this problem we calculate the mean $\overline{x}_{(j)}$ and variance $V[x_{(j)}]$ for each dataset.

    We then plug the values into $\textcolor{blue}{[2]}$ to find $\overline{x}_w = 52.02\,\mathsf{g}$.
    """)
    return


@app.cell(hide_code=True)
def _():
    #### Python solution to 2.3
    import numpy as np

    # The two datasets as numpy arrays
    xA = np.array([52.010, 52.041, 52.105, 51.998, 51.981])
    xL = np.array([52.05, 52.03, 52.07, 51.90, 51.94])

    # 2D array of both datasets
    xAL = np.stack((xA, xL))

    # axis=1: dataset is row
    means_xAL = np.mean( xAL, axis=1)
    vars_xAL = np.var( xAL, axis=1)
    wmean_xAL = np.sum(means_xAL/vars_xAL)/np.sum(1/vars_xAL)
    print("Weighted mean of a pack of rolos from Alf and Lily's joint datasets {:.2f} g".format(wmean_xAL) )
    print("The variance on Alf's measurements is V[x_A] = {:1.2e} g and on Lily's is V[x_L] = {:1.2e} g".format(vars_xAL[0], vars_xAL[1]) )
    # Alf's measurements have smaller variance, which means smaller statistical uncertainty
    # Alf's measurements will have a larger weight for this reason. Not because they have an additional sig fig.
    return


if __name__ == "__main__":
    app.run()
