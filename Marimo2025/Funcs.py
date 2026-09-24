import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    # Only a few snippets for this week. Next week will have lots.
    return (np,)


@app.cell
def _(np):
    # Normalise a function over some range so we can use it as a PDF
    from scipy.stats import rv_continuous

    class RV_dist(rv_continuous):
        def _pdf(self,x):
            p = x**3 
            s = np.sum(p)
            return p/s

    # instantiation of fX in the range 0,1 where we know it to be normalised
    f_X = RV_dist(name = 'x^3', a=0, b=1)
    return (f_X,)


@app.cell
def _(np):
    # Effect of varying the scale parameter
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    mu = 0
    lines = ['c--', 'k-', 'm:']
    sigmas = [0.5,1,2]
    xvals = np.linspace(-5,5, 100)
    for l,sigma in zip(lines,sigmas):             
            y = norm.pdf(xvals, loc=mu, scale=sigma)
            plt.plot(xvals, y, l)
    plt.show()
    return (norm,)


@app.cell
def _(f_X, norm, np):
    # CDF: scipy norm
    lb, ub = norm.support()# default -infty , infty
    print("Lower and upper bounds of scipy's default norm (Gaussian): {},{}".format(lb,ub))
    print("P(x<=infty) F(infty)=", norm.cdf(ub) )
    print("P(x<=-infty) F(-infty)=", norm.cdf(lb) )
    print("P(x<=1) = F(1)= {:.4f}".format( norm.cdf(1) ) )
    print("P(0.999<x<=1.000) = F(1)-F(0.999)= {:.4f}".format( norm.cdf(1) - norm.cdf(0.999) ) )

    # CDF: our toy PDF
    toy_lb, toy_ub = f_X.support()
    print("\nLower and upper bounds of our toy PDF (f_X = x^3): {},{}".format(toy_lb,toy_ub))
    print("Sanity check! P(x>1) = F(infty)-F(1)= {:.4f}".format( f_X.cdf(np.infty)-f_X.cdf(1) ) )

    print("P(x<0.1) = F(0.1)= {:.4f}".format( f_X.cdf(0.1) ) )
    print("P(0.45<x<=0.55) = F(0.55)-F(0.45)= {:.4f}".format( f_X.cdf(0.55) - f_X.cdf(0.45) ) )
    print("P(x>=0.95) = F(1)-F(0.95)= {:.4f}".format( f_X.cdf(1) - f_X.cdf(0.95) ) )

    return


if __name__ == "__main__":
    app.run()
