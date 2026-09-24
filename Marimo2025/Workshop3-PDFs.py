import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Data Analysis Techniques: Workshop Exercises

    ## Week 4

    #### **Exercise 4.1) When we have independent trials with known individual probabilities**

    The probability of me rolling a dice and getting a six is $p=\frac{1}{6}$. Last week I was feeling lucky, so I rolled a dice 100 times and got a six 21 times.

    a) Identify the probability distribution you would use in this case, stating why.

    b) Use it to calculate the probability of this occurring "naturally".

    c) Plot the distribution, indicating this outcome on the plot.

    #### **Exercise 4.2) When we have an expectation for events/time**

    On average, I receive 100 emails between the hours of 08:00 and 18:00 UK time, or ten emails per hour over this period. Last week I got the heeby jeebies after receiving zero emails for just over 2 hours, and wondered if the email server was down. It turns out I was just lucky!

    a) Identify the probability distribution you would use in this case, stating why.

    b) Use it to calculate the probability of this occurring "naturally".

    c) Plot the distribution, indicating this outcome on the plot.

    #### **Exercise 4.3) When we measure X~Expon($\lambda$) but want probabilities for Y=f(X)**

    **Parts of this exercise is looking ahead to things we will cover in the next week, but all of you should be able to do some of it and some of you will be able to do all of it**

    Let X ~ Expon(2) and Y=2+3X

    a) Find P(X>2)

    b) Find E[Y] and Var[Y]
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ##Solution 4.1##

    The probability of me rolling a dice and getting a six is $p=\frac{1}{6}$. Last week I was feeling lucky, so I rolled a dice 100 times and got a six 21 times.

    a) Identify the probability distribution you would use in this case, stating why.

    **I will use the Binomial distribution $K\sim$ Binom($n=100,p=\frac{1}{6}$) for this problem, as it is a set of 100 independent Bernoulli trials.**

    **The Binomial PMF is** $p_k = C_{k}^np^k(1-p)^{n-k}$**, where the Binomial Coefficient **$C_k^n = \dfrac{n!}{k!(n-k)!}$ **gives me the number of ways I can get $k$ successes from $n$ trials.**

    b) Use it to calculate the probability of this occurring "naturally".

    **The DRV is K. This is the number of successes, where in this case a success is rolling a six, and failure is `not a six'.**

    First we find $C_{21}^{100} = \dfrac{100!}{21!(100-21)!}$
    """)
    return


@app.cell
def _():
    def BinomCoeff(n,k,p):
        # number of ways (combinations) to get k successes from n trials
        # n!/k!(n-k)!
        import math
        return math.comb(100,21)

    BinomCoeff(100,21,1/6)
    print("There are {:e} ways to get {} sixes in {} rolls of the dice.".format(BinomCoeff(100,21,1/6), 21, 100))
    return (BinomCoeff,)


@app.cell
def _(mo):
    mo.md(r"""
    This is a huge number, which makes sense when you think about what we are asking: "How many ways can I write out a sequence of 100 outcomes that has 21 sixes in it, in any positions?"\"

    We can find the probability of an exact value for $K$ using the PMF because $K$ is discrete.

    $p_{21} = C_{21}^{100}\; \frac{1}{6}^{(21)}\;\frac{5}{6}^{(100-21)}$
    """)
    return


@app.cell
def _(BinomCoeff):
    p21 = BinomCoeff(100,21,1/6) * (1/6)**(21) * (5/6)**(100-21)
    print("Probability of {} sixes from {} rolls: {} ".format(21,100,p21) )
    return


@app.cell
def _(np):
    # We have found the probability of getting 21 sixes is 5.17%

    # We can reassure ourselves that we haven't done something mad in python by going back to the math.
    # Expected value and variance for Binom are E[K] = np and V[K] = np(1-p)
    expected_value = 100*(1/6)
    var = 100*(1/6)*(5/6)
    print("Expected number of sixes in {:.2f} rolls is: {:.2f}".format(100, expected_value ) )
    print("The variance on that is: {:.2f}".format(var ) )
    print("So the true mean +/- standard deviation is: {:.2f}+/-{:.2f}".format(expected_value, np.sqrt(var) ) )

    # This is informative because I can immediately see that 21 is higher than I would expect, but not outrageously high. It is more than 1sigma (one standard deviation) from the mean, but less than 2sigma.
    # Let's see how many sigmas:
    sigmas_from_mean = (21-expected_value)/np.sqrt(var)
    print("21 sixes is {:.2f} sigmas away from the expected value ".format(sigmas_from_mean))
    return


@app.cell
def _(mo):
    mo.md(r"""
    c) Plot the distribution, indicating this outcome on the plot.
    """)
    return


@app.cell
def _(np):
    # matplotlib pyplot for making plots
    import matplotlib.pyplot as plt
    # matplotlib.pylab to make the axis lables and titles nice and big. The bigger the better, within reason.
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 16,
        'axes.labelsize': 16,
        'axes.titlesize':16,
        'xtick.labelsize':14,
        'ytick.labelsize':14,
        'figure.constrained_layout.use':True}
    pylab.rcParams.update(params)


    def PlotBinomPMF(n,p):
        from scipy.stats import binom

        fig, ax1 = plt.subplots(1, 1)
        # set up values for the x-axis 

        # first pass, plot the full support of the RV 
        kvals = np.arange(0, n) 
        pmf=binom.pmf(k=kvals, n=n, p=p, loc=0)
        cdf=binom.cdf(k=kvals, n=n, p=p, loc=0)
        ax1.bar(kvals,pmf, alpha=0.4, color='lightsteelblue',edgecolor='black',linewidth=1)

        color = 'tab:blue'
        ax1.set_ylabel('PMF', color=color, fontsize=16)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='y', labelcolor=color)

        # I am going to zoom  on the lower part of the support where the action is.
        ax1.set_xlim(0,n/2)

        ax2 = ax1.twinx()  

        ax2.plot(kvals, cdf, 'r-', linewidth=2)
        color = 'tab:red'
        ax2.set_ylabel('CDF', color=color, fontsize=16)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0)

        # Add titles and labels
        xtext="Number of heads $k$ from n={} tosses".format(n)
        ax1.set_xlabel(xtext, fontsize=16)

        titletext="Binomial(n={}, p={:.2f})".format(n,p)
        plt.title(titletext, fontsize=16)

        # draw a vertical line at k=21
        ax1.axvline(x=21, color="green", linestyle="--",linewidth=2)


        return plt.show()

    PlotBinomPMF(100,1/6)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### **Solution 4.2) When we have an expectation for events/time**

    On average, I receive 100 emails between the hours of 08:00 and 18:00 UK time, or ten emails per hour over this period. Last week I got the heeby jeebies after receiving zero emails for just over 2 hours, and wondered if the email server was down. It turns out I was just lucky!

    a) Identify the probability distribution you would use in this case, stating why.

    **I see counts per time, and I think $K~$Poisson($\lambda$).**

    **We are given an expected count for a fixed time period of two hours,** $\lambda=20$, **and know the PMF** $p_k = e^{-\lambda} \dfrac{\lambda^k}{k!}$, so are all set to find out the probability of $k=0$.


    b) Use it to calculate the probability of this occurring "naturally".
    """)
    return


@app.cell
def _(np):
    # we are using mu for our rate lambda
    def PoissonProbCalc(mu, k):
        import math
        prob = np.exp(-mu) * (mu**k / math.factorial(k) )
        return prob

    print( "Probability of {} emails in 2 hours based on usually getting {} in 2 hours is {} ".format(0,20,PoissonProbCalc(20, 0) ) )

    return


@app.cell
def _(np):
    # This is a VERY small probability. I will just quickly check how many sigmas that is...
    def sigma_check_email(mu,k):
        # expected value for poisson is E[k] = mu and V[k] is the same.
        expec = mu
        var = mu
        sigma = np.sqrt(var)
        print("True mean [+/-] standard deviation is: {:.2f}[+/-]{:.2f}".format(expec, sigma ) )
        # Let's see how many sigmas - BUT note, this will give us a negative value which doesn't make sense    
        # sigmas_from_mean = (k-expec)/sigma
        # we will square it and then take the square root, as that will give us a positive value
        sigmas_from_mean = np.sqrt( (k-expec)**2 ) /sigma
        return sigmas_from_mean

    print("Zero emails in 2 hours is a {:.2f} sigma event ".format( sigma_check_email(20,0) ))
    return


@app.cell
def _(mo):
    mo.md(r"""
    c) Plot the distribution, indicating this outcome on the plot.
    - please see the complete example in SpecialPDFs.py on the canvas page for week 4.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### **Solution 4.3) When we measure X~Expon($\lambda$) but want probabilities for Y=f(X)**

    Let X ~ Expon(2) and Y=2+3X
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now we have a Continuous RV $X~$Expon($\lambda$), so we have to use CDF to calculate probabilities.

    a) Find P(X>2)

    $P(X>2) = 1 - P(X\leq 2) = 1 - F(2)$

    CDF $F(2) = \int\limits_{0}^{2} f_X(x)\, dx = \int\limits_{0}^{2} \lambda e^{-\lambda x}\, dx =\left[-\dfrac{\lambda}{\lambda} e^{-\lambda x} \right]_0^2 =-\left[ e^{-2\lambda} -1 \right]=1-e^{-2\lambda}$

    $P(X>2) =1-F(2) = e^{-2\lambda}$

    Final step, we are given $\lambda=2$ in the question: "Let X ~ Expon(2)":

    $P(X>2) =e^{-4} = 0.01831563888$

    Now let's check this in python.
    """)
    return


@app.cell
def _():
    def prob_leq_expon(rate, x):
        from scipy.stats import expon
        cdf = expon.cdf(x, loc=0, scale=1/rate)
        return cdf

    #print("P(X<=2) = {} ".format( prob_leq_expon(2, 2) ) )
    print("P(X>2) = {} ".format( 1-prob_leq_expon(2, 2) ) )    
    return


@app.cell
def _(mo):
    mo.md(r"""
    b) Find E[Y] and Var[Y]

    We are given Y = 2+3X: a monotonic f(X)

    The formula for transforming a single-varieable PDF is: $f_Y(y) = \left|\dfrac{dx}{dy}\right| f_X(x)$

    To differentiate X wrt Y we write $X = \dfrac{1}{3}(Y-2)$ such that:

    $\dfrac{dx}{dy} = \dfrac{1}{3}$

    $f_Y(y) = \dfrac{1}{3} \lambda e^{-\lambda (\frac{1}{3} (y-2) )} = \dfrac{1}{3} [\lambda e^{-\lambda y/3} e^{2\lambda/3} ]=\dfrac{2e^{4/3}}{3} e^{-2y/3}$

    We know that for X~Expon($\lambda$) we can write $E[X] = \dfrac{1}{\lambda}$ and $V[X] = \dfrac{1}{\lambda^2}$ (week 4 lectures)

    Expectation algebra tells us that $E[aX+b] = E[aX] + E[b] = aE[X] +b$

    $E[Y] = E[2+3X] = 2 + 3E[X] = 2 + \dfrac{3}{\lambda}$

    For the variance, (we are looking at this in week 5) the algebra is $V[aX+b] = a^2 V[X]$

    $V[Y] = V[2+3X] = 3^2 V[X] = 9\dfrac{1}{\lambda^2}$

    So for X~Expon(2) we have $E[Y] = \dfrac{14}{4}$ and $V[Y] = \dfrac{9}{4}$.

    Let's check with python
    """)
    return


@app.cell
def _():
    def expvar_expon(rate):
        from scipy.stats import expon

        lb,ub = expon.support(scale=1/rate)
        print("Support of X~Expon[rate={}] is {}<=X<={} ".format(rate,lb,ub) )

        # grab the expected value and variance of X real quick
        mean, var = expon.stats(moments='mv',scale=1/rate)
        print("E[X]={}, V[X]={} ".format(mean,var) )

        # scipy has neat way to get the E[f(X)] by passing the function directly to expect()
        # lambda x: 2+3*x means "x, where x=2+3x"    
        EY = expon.expect(lambda x: 2+3*x, lb=lb, ub=ub, scale=1/rate)   
        print("E[Y] = {}".format(EY) )

        # to get E[Y^2] we just change the function to lambda x: (2+3*x)**2, which means  "x, where x=(2+3x)^2"
        EY2 = expon.expect(lambda x: (2+3*x)**2, lb=lb, ub=ub, scale=1/rate)  

        # Variance is V[Y] = E[Y^2] - E^2[Y]
        VY = EY2 - EY**2
        print("V[Y] = {}".format(VY) )

    expvar_expon(2)    

    return


if __name__ == "__main__":
    app.run()
