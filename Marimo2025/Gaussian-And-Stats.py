import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
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
    return np, plt


@app.cell
def _(plt):
    # Bivariate Gaussian Example
    def bivariate_gaus(means,cov):
        from scipy.stats import multivariate_normal
        import seaborn as sns
        import pandas as pd
        # two means for two marginals
        mu = means  

        # bivariate gaussian
        XYdist = multivariate_normal(mu, cov)
        rvs= XYdist.rvs(size=100_000) # N rows, Z columns: need transpose for JointGrid- dataframe fixes this
        df = pd.DataFrame(rvs, columns=["X1", "X2"])

        # seaborn plotting
        g = sns.JointGrid(x=df["X1"], y=df["X2"])
        g.plot_joint(sns.kdeplot,fill=True)
        g.plot_marginals(sns.histplot,fill=True)

        # example of limiting axes
        g.ax_marg_x.set_xlim(-10,10)
        g.ax_marg_y.set_ylim(-10,10)

        # print mu and cov on plot
        g.ax_joint.text(-9, 9, r"$\mu$:{}, $\Sigma$:{}".format(mu,cov), fontsize=16)

        # axis labels
        g.ax_joint.set_xlabel(r'X$_{(1)}$',fontsize=16)
        g.ax_joint.set_ylabel(r'X$_{(2)}$',fontsize=16)

        plt.show()
        # example of converting arrays to strings
        #mustring='-'.join(str(m) for m in mu)
        #covstring = '-'.join('-'.join(str(x) for x in y) for y in cov)
        # set the figure name 
        #figname="BivariateGaussian-mu-{}-cov-{}.png".format(mustring,covstring)
        #plt.savefig(figname, bbox_inches='tight')
        #plt.clf()

    means = [0, 0]
    # zero covariance version
    cov = [[10,0],[0,5]]
    bivariate_gaus(means,cov)
    # nonzero covariance version
    cov = [[10,5],[5,5]]
    bivariate_gaus(means,cov)


    return


@app.cell
def _(np, plt):
    # CLT visual example
    # we will make six histograms for different numbers of dice rolls
    def clt_example():
        from scipy.stats import norm
        np.random.seed(1)
        a = 1 # lowest score possible
        b = 6 # highest score possible
        Z = 1000 # number of data points (to fill out the histograms!)
        # number of measurements used in mean calculation
        N = [10,100,1000] 
        # loop over the different N and save the means
        allmeans = []
        for Ni in N:
            means_zi=[]
            for zi in range(Z):
                rvs = np.random.randint(a, a+b, Ni)
                mean = np.mean(rvs)
                means_zi.append(mean)
            allmeans.append(means_zi)
        
        # loop over the different N and means, make the plots
        for means, Ni in zip(allmeans, N):
            fig, ax1 = plt.subplots(1,1)
            datalabel="Fake data, N={}".format(Ni)

            ax1.hist(means, density=True, alpha=0.6, color='lightsteelblue', edgecolor='black',label=datalabel)

            # fit gaussian to the data (returns mu and sigma)
            mu_f, sigma_f = norm.fit(means)
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            x_gaus = np.linspace(xmin, xmax, 100)
            pdf = norm.pdf(x_gaus, mu_f, sigma_f)
            mlab=r"$\overline{x}$"
            ax1.plot(x_gaus, pdf, 'r--', linewidth=2, label=r"{}$\sim Norm(\mu=${:.2f}$,\sigma=${:.2f}$)$".format(mlab,mu_f,sigma_f))

            # label faffing
            ax1.set_xlabel(r"Mean score $\overline{x}$")
            ax1.set_ylabel("Density")
            ax1.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc='upper right')
            plt.show()
            #figname="CLT-demo-N{}-Z{}-fit{}.png".format(Ni,Z,plotgaus)
            #plt.tight_layout()       
            #plt.savefig(figname, bbox_inches='tight')
            #plt.clf()

    clt_example()
    return


@app.cell
def _(np):
    # CLT usage example for RVs from any PDF
    def UseCLT(nx, meanx, varx, plow=0, phigh=np.inf):
        from scipy.stats import norm
        # nx is the number of iid RVs, meanx is their mean, varx is their variance
        # CLT: Z = ( Y-E[Y] ) / sqrt( V[Y] ) ~ Norm(0,1)
        # Let Y = sum(X)
        # EY = N E[X], VY = N V[X]  (because X are iid)
        EY = nx*meanx
        VY = nx*varx
        lower_bound = (plow-EY)/np.sqrt(VY)
        upper_bound = (phigh-EY)/np.sqrt(VY)
        prob = norm.cdf(upper_bound)-norm.cdf(lower_bound)
        if phigh==np.inf:
            print("P(Y > {} min ) = {}".format(plow,prob) )
        elif plow==0:
            print("P(Y < {} min ) = {}".format(phigh,prob) )
        else:
            print("P({} min < Y < {} min) = {}".format(plow,phigh, prob) )

    # Alf serves 17 customers, taking a mean time of 2 mins for each one with a variance of 1.5 minutes. What is the probability that the total time he is at work for will exceed 30 mins?
    UseCLT(17, 2, 1.5, 30)
    # What is the probability that he will work for less than 45 minutes?
    UseCLT(17, 2, 1.5, 0, 45)
    # What is the probability he will work for between 33 and 35 minutes?
    UseCLT(17, 2, 1.5, 33,35)
    return


@app.cell
def _():
    # p-values and z-values
    def PrintZvalues(pvalues):
        from scipy.stats import norm
        import pandas as pd
        data=[]
        for p in pvalues:
            prob = 100*(1-p)
            z = norm.ppf(1-p)
            d=[p,prob,z]
            data.append(d)

        df = pd.DataFrame(data, columns=["p-val", "Prob (%)","z-val (sigma)"])
        print ("\nprobability table:\n", df)

    PrintZvalues([0.01,0.02,0.03,0.04,0.05])

    def PrintPvalues(zvalues):
        from scipy.stats import norm
        import pandas as pd
        data=[]
        for z in zvalues:
            prob = 100*norm.cdf(z)         
            p = 1-norm.cdf(z)
            d=[z,prob,p]
            data.append(d)

        df = pd.DataFrame(data, columns=["z-val (sigma)", "Prob (%)","p-val "])
        print ("\n probability table:\n", df)

    PrintPvalues([1,2,3,4,5])
    return


if __name__ == "__main__":
    app.run()
