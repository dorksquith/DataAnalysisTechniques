import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # general imports and settings
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 16}
    rc('font', **font)
    rc('text', usetex=True)
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###**Exercise 6.1 CLT: Binomial Means are Gaussian Distributed**###

    Gaussian-And-Stats.py has the python for the example of uniform means forming a Gaussian distribution, as we saw in the lectures.

    Use this to write a python script to extract values from a Binomial distribution (you choose the parameters), calculate the mean, then repeat such that you have a large set of means.

    a) Plot the histogram of binomial means, and fit a gaussian to it.

    b) How many means do you need to calculate to get a feasible (by eye) fit from the Gaussian?

    ###**Exercise 6.2 Z-scores**###

    A dog show features 200 female golden retrievers whose weights follow a Gaussian distribution with a mean 27.2kg  and standard deviation 2.58 kg.

    My friend Geraldine who is also a golden retriever shows up to check out the competition. She weighs 33kg without her shoes on.

    a) What is Geraldine's z-value? Give three significant figures.

    b) What is Geraldine's p-value? Give three significant figures.

    c) How many of the other dogs are heavier than her?

    ###**Exercise 6.3 Bivariate Gaussian contours**##

    Gaussian-And-Stats.py has the python for plotting a bivariate Gaussian distribution with two choices of covariance matrix.

    a) Can you choose any values for the covariance matrix elements? What are the limitations?

    b) The contours in the example I provided are automatically selected. See if you can set them to colour the contours according to the 1,2,3,.. sigma probabilities. You will need to refer back to the lecture slides to set up an array of PDF values at different sigmas, and use the seaborn plot_joint levels flag to set the levels manually to this array.

    c) Extension: can you make a 3D plot of the bivariate Gaussian, with the z-axis being the PDF value?
    """)
    return


@app.cell
def _(np, plt):
    ###**Solution 6.1 CLT: Binomial**###

    def clt_binom():
        fig, ax1 = plt.subplots(1,1)
        from scipy.stats import binom,norm
        # parameters of binomial
        param_n = 10 # this is the number of times we toss the coin, can be anything
        param_p = 0.5 # the probability of success, can be anything
        true_mean = param_n*param_p
        true_sigma = np.sqrt(param_n*param_p*(1-param_p))
        print("The parameters I have chosen correspond to Binomial with true mean {:.3f} and standard deviation {:.3f} ".format(true_mean, true_sigma) )
        kDist = binom(param_n, param_p) # this is the distribution of K~Binom(n,p)

        # pull N random values of k from the distribution
        # This numbe N is used to calculate the means, and it must be large for the distribution to approach Normal
        N=10_000
        rvs = kDist.rvs(size=N) 
        # the rvs will have a mean value
        mean_1 = np.mean(rvs)
        sigma_1 = np.std(rvs)
        print("The mean of my single distribution with {} data points is {:.3f} +/- {:.3f}".format(N,mean_1, sigma_1))

        # we want to build a distribution of the means, so need a whole bunch of them
        Z = 1000  # to fill out the histogram
        Zmeans=[]
        for Ni in range(Z):
            rvs = kDist.rvs(size=N)
            mean_i = np.mean(rvs)
            Zmeans.append(mean_i)

        # plot the binomial means
        ax1.hist(Zmeans, density=True, alpha=0.6, color='lightsteelblue', edgecolor='black',label="Binomial means")

        # Gaussian fit
        mu_f, sigma_f = norm.fit(Zmeans)
        print("Gaussian fit to Binomial means distribution has mu: {:.3f}, sigma: {:.3f}".format(mu_f,sigma_f))
        sigma_N = true_sigma/np.sqrt(N)
        print("Note that the Gaussian sigma is {:.3f} and sigma_binomial/sqrt(N) =  {:.3f}".format(sigma_f, sigma_N) )
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        x_gaus = np.linspace(xmin, xmax, 100) # this just sets the range and number of x-points at which to draw the gaussian pdf
        pdf = norm.pdf(x_gaus, mu_f, sigma_f)

        ax1.plot(x_gaus, pdf, 'r--', linewidth=2)
        plt.show()

    clt_binom()
    return


@app.cell
def _():
    ###**Solution 6.2 Z-scores**###
    # a) z-value
    def zscore(x, mu, sigma):
        return (x-mu)/sigma

    x=33
    mu=27.2
    sigma=2.58
    z = zscore(x, mu, sigma)
    print("z-value: {:.3f}".format(z) )
    return (z,)


@app.cell
def _(z):
    # b) p-value
    def pvalue(z):
        from scipy.stats import norm
        p = 1-norm.cdf(z)
        return p

    p = pvalue(z)
    print("p-value: {:.3f}".format(p) )
    return


@app.cell
def _(np, z):
    # c) how many dogs are heavier?
    def count_heavier_dogs(z,n):
        from scipy.stats import norm
        p_lighter = norm.cdf(z)
        n_lighter = p_lighter*n
        return n-n_lighter

    n=200
    n_h = count_heavier_dogs(z,n)
    print("Number of dogs heavier than Geraldine: {}".format(np.floor(n_h)) )
    return


@app.cell
def _(np, plt):
    ###**Solution 6.3 Bivariate Gaussian Contours**###
    def Gaussian_SigmaContours():
        from scipy.stats import multivariate_normal
        import seaborn as sns
        import pandas as pd

        # set the means and the covariance matrix
        mu = [0, 0]
        cov = [[10,5],[5,10]]
        sigma=np.sqrt(cov[0][0])

        # From lectures, the max value of the PDF is 
        smax = 1./(np.sqrt(2*np.pi)*sigma)
        # and the value of the PDF at 1sigma is
        s1 = smax / np.sqrt(np.e)
        levels=[]
        for i in [7,6,5,4,3,2,1]:
            levels.append( s1/i )
        levels.append(smax)
        levels.append(1)
        print(levels)

        mustring='-'.join(str(m) for m in mu)
        covstring = '-'.join('-'.join(str(x) for x in y) for y in cov)
        figname="BivariateGaussian-mu-{}-cov-{}.png".format(mustring,covstring)

        XYdist = multivariate_normal(mu, cov)
        rvs= XYdist.rvs(size=100_000) # N rows, Z columns: need transpose for JointGrid- dataframe fixes this
        df = pd.DataFrame(rvs, columns=["X1", "X2"])

        g = sns.JointGrid(x=df["X1"], y=df["X2"])
        g.plot_joint(sns.kdeplot,fill=False,levels=levels)
        g.plot_marginals(sns.histplot,fill=True)

        g.ax_marg_x.set_xlim(-10,10)
        g.ax_marg_y.set_ylim(-10,10)
        g.ax_joint.text(-9, 9, r"$\mu$:{}, $\Sigma$:{}".format(mu,cov), fontsize=16)

        g.ax_joint.set_xlabel(r'X$_{(1)}$',fontsize=16)
        g.ax_joint.set_ylabel(r'X$_{(2)}$',fontsize=16)
        plt.show()
    Gaussian_SigmaContours()
    return


if __name__ == "__main__":
    app.run()
