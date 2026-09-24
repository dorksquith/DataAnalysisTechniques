import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    # GENERAL IMPORTS
    import marimo as mo
    import numpy as np

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
    return np, plt, pylab


@app.cell
def _(plt):
    def plot_bernoulli():
        fig, ax1 = plt.subplots(1, 1)
        from scipy.stats import bernoulli

        # set up the parameter p and the RVs k
        p=0.6
        k = list(range(2))

        # plot the PMF
        pmf=bernoulli.pmf(k, p)


        ax1.bar(k,pmf, alpha=0.4, color='lightsteelblue',edgecolor='black',linewidth=1,width=0.2)
        # plot the CDF on shared x-axis and new y-axis
        cdf=bernoulli.cdf(k, p)

        ax2 = ax1.twinx()  
        ax2.plot(k, cdf, 'ro')

        # sort out the PMF y axis
        ax1.set_ylabel('PMF', color='tab:blue', fontsize=16)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # annotate PMF plot
        ax1.annotate("", (0,pmf[0]), (-1, pmf[0]),arrowprops={'arrowstyle':'<|-','color':'tab:blue', 'linestyle':'--'})
        ax1.annotate("", (1,pmf[1]), (-1, pmf[1]),arrowprops={'arrowstyle':'<|-','color':'tab:blue', 'linestyle':'--'})

        # sort out the CDF y axis
        color = 'tab:red'
        ax2.set_ylabel('CDF', color=color, fontsize=16)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0,top=1.2)

        # annotate CDF plot
        ax2.annotate("", (0,cdf[0]), (2, cdf[0]),arrowprops={'arrowstyle':'<|-','color':'red', 'linestyle':':'})
        ax2.annotate("", (1,cdf[1]), (2, cdf[1]),arrowprops={'arrowstyle':'<|-','color':'red', 'linestyle':':'})

        # Shared x-axis for PMF and CDF
        ax1.set_xlim(-1,2)
        ax1.set_xlabel("k", fontsize=16)

        # Save figure
        plt.title("K~Bernoulli(p=0.6)", fontsize=16)

        return plt.show()

    plot_bernoulli()
    return


@app.cell
def _(np, plt):
    # BINOMIAL PMF
    def plot_binomial():
        from scipy.stats import binom

        fig, ax1 = plt.subplots(1, 1)

        n=50
        p=0.5

        print("Expected value: ", binom(n,p).expect() )
        print("Variance: ", binom(n,p).var() )

        k = np.arange(0, n) 

        pmf=binom.pmf(k=k, n=n, p=p, loc=0)
        cdf=binom.cdf(k=k, n=n, p=p, loc=0)

        ax1.bar(k,pmf, alpha=0.4, color='lightsteelblue',edgecolor='black',linewidth=1)

        color = 'tab:blue'
        ax1.set_ylabel('PMF', color=color, fontsize=16)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlim(0,n)

        ax2 = ax1.twinx()  

        ax2.plot(k, cdf, 'r-', linewidth=2)
        color = 'tab:red'
        ax2.set_ylabel('CDF', color=color, fontsize=16)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0)

        # Add titles and labels
        xtext="k: Number of heads from n={} tosses".format(n)
        ax1.set_xlabel(xtext, fontsize=16)

        titletext="K$\sim$ Binomial(n=50, p=0.5)"
        plt.title(titletext, fontsize=16)
        return plt.show()

    plot_binomial()
    return


@app.cell
def _(np, plt):
    def plot_poisson():
        from scipy.stats import poisson
        fig, ax1 = plt.subplots(1, 1)

        n=25
        mu=10
        k = np.arange(0, n) 
        pmf=poisson.pmf(k=k, mu=mu, loc=0)


        expect = poisson(mu).expect()
        var = poisson(mu).var()



        cdf=poisson.cdf(k=k, mu=mu,  loc=0)

        ax1.bar(k,pmf, alpha=0.4, color='lightsteelblue',edgecolor='black',linewidth=1)

        color = 'tab:blue'
        ax1.set_ylabel('PMF', color=color, fontsize=16)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlim(0,n)

        ax2 = ax1.twinx()  

        ax2.plot(k, cdf, 'r-', linewidth=2)
        color = 'tab:red'
        ax2.set_ylabel('CDF', color=color, fontsize=16)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0)


        # Add titles and labels
        xtext="k: N emails received in one hour".format(n)
        ax1.set_xlabel(xtext, fontsize=16)

        titletext=r"K$\sim$Poisson($\lambda=${})".format(mu)
        plt.title(titletext, fontsize=16)
        plt.show()
    plot_poisson()
    return


@app.cell
def _(np, plt):
    def plot_uniform():
        from scipy.stats import uniform
        fig, ax1 = plt.subplots(1, 1)
        a=-2
        b=4
        x = np.arange(-5, 5, 0.001)

        pdf=uniform.pdf(x, loc=a, scale=b)
        cdf=uniform.cdf(x, loc=a, scale=b)

        ax1.plot(x, pdf, '-',linewidth=1)
        ax1.fill(x, pdf, 'lightsteelblue')

        color = 'tab:blue'
        ax1.set_ylabel('PDF', color=color, fontsize=16)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlim(-5,5)

        ax2 = ax1.twinx()  

        ax2.plot(x, cdf, 'r-', linewidth=2)
        color = 'tab:red'
        ax2.set_ylabel('CDF', color=color, fontsize=16)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0)


        # Add titles and labels
        xtext="x"
        ax1.set_xlabel(xtext, fontsize=16)


        titletext="X~Uniform(a={},b={})".format(a,b)
        plt.title(titletext, fontsize=16)
        plt.show()
    plot_uniform()
    return


@app.cell
def _(np, plt):
    def plot_expon():
        from scipy.stats import expon
        fig, ax1 = plt.subplots(1, 1)
        rate =1.3
        scale=1/rate
        t = np.linspace(0,5_000,50_000)

        pdf=expon.pdf(t, loc=0, scale=scale)
        cdf=expon.cdf(t, loc=0, scale=scale)
        ax1.plot(t, pdf, '-',linewidth=1)
        ax1.fill_between(t, pdf, interpolate=True, color='lightsteelblue')

        color = 'tab:blue'
        ax1.set_ylabel('PDF', color=color, fontsize=16)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlim(0,5)

        ax2 = ax1.twinx()  

        ax2.plot(t, cdf, 'r-', linewidth=2)
        color = 'tab:red'
        ax2.set_ylabel('CDF', color=color, fontsize=16)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0)


        # Add titles and labels
        xtext="t"
        ax1.set_xlabel(xtext, fontsize=16)
        titletext=r"T$\sim$ Expon($\lambda$={})".format(rate)
        plt.title(titletext, fontsize=16)
        plt.ylim(bottom=0)

        plt.show()
    plot_expon()
    return


@app.cell
def _(np, plt):
    def plot_beta():
        from scipy.stats import beta
        fig, ax1 = plt.subplots(1, 1)
        a=4
        b=9
        x = np.linspace(0,1,100)

        pdf=beta.pdf(x, a=a, b=b)
        cdf=beta.cdf(x, a=a, b=b)

        ax1.plot(x, pdf, '-',linewidth=1)

        d = np.zeros(len(pdf))
        ax1.fill_between(x, pdf, where=pdf>=d, interpolate=True, color='lightsteelblue')

        color = 'tab:blue'
        ax1.set_ylabel('PDF', color=color, fontsize=16)
        ax1.set_ylim(bottom=0)
        ax1.tick_params(axis='y', labelcolor=color)
        #ax1.set_xlim(0,5)

        ax2 = ax1.twinx()  

        ax2.plot(x, cdf, 'r-', linewidth=2)
        color = 'tab:red'
        ax2.set_ylabel('CDF', color=color, fontsize=16)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0)


        # Add titles and labels
        xtext="x"
        ax1.set_xlabel(xtext, fontsize=16)

        titletext="X~Beta({},{})".format(a,b)
        plt.title(titletext, fontsize=16)
        return plt.show()
    plot_beta()
    return


@app.cell
def _(np, plt, pylab):


    def plot_norm():
        from scipy.stats import norm
        import matplotlib.ticker as mticker
        params = {'legend.fontsize': 10,
                 'axes.labelsize': 10,
                 'axes.titlesize':10,
                 'xtick.labelsize':10,
                 'ytick.labelsize':10,
                 'figure.constrained_layout.use':True}
        pylab.rcParams.update(params)
        fig=plt.figure()
        mu=0
        sigma=1
        x = np.linspace(norm.ppf(1e-10), norm.ppf(1-1e-10), 10000)
        # x = np.linspace(norm.ppf(1e-10), norm.ppf(1-1e-10), 100)
        pdf=norm.pdf(x, loc=mu, scale=sigma)
        cdf=norm.cdf(x, loc=mu, scale=sigma)


        for i in range(1,7):
            ax = fig.add_subplot(2,3, i)
            p = norm.cdf(mu+i*sigma)-norm.cdf(mu-i*sigma)
            ax.semilogy(x, pdf, '-',linewidth=1)
            ax.fill_between(x, pdf, where=(x>mu-i*sigma) & (x<mu+i*sigma), color='lightsteelblue',edgecolor='black',alpha=0.6)
            ax.text(-7, .6, "{:.8f}%".format(100*p), color='black')
            ax.text(-1.8, .002, r"$\pm${}$\sigma$".format(i*sigma), color='blue')
            color = 'tab:blue'
            ax.set_ylabel('PDF', color=color)
            ax.set_ylim(bottom=10e-10,top=10)
            ax.tick_params(axis='y', labelcolor=color)
            ax.set_xlim(-10,10)

            titletext=r"X~Norm($\mu,\sigma$)"
            plt.title(titletext)


        #plt.tight_layout()
        return plt.show()

    #for i in range(1,6):
    plot_norm()
    return


if __name__ == "__main__":
    app.run()
