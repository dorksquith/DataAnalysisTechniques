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
    ###**Exercise 7.1 Nonlinear Least Squares**###

    EstimatorsIntro.py has the python for generating fake data points and doing a linear least squares fit. You will modify this to a nonlinear function of x for this exercise.

    a) Make a plot of the data points following an $x^2$ distribution, showing the residuals and the Least Squares Fit.


    ###**Exercise 7.2 Likelihood for Normal Sigma**###

    EstimatorsIntro.py has the python for plotting the Likelihood, Log Likelihood, and Negative Log Likelihood for the loc parameter (the expected value, $\mu$) from a normal distribution.

    a) Adjust the provided code to make the three Likelihood plots for the scale parameter ($\sigma$, the standard deviation) of the normal distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###**Solution 7.1 Nonlinear Least Squares**###

    Python example below - there is no limit to the ways in which you can do this! Please see EstimatorsIntro.py for properly commented example.
    """)
    return


@app.cell
def _(np, plt):
    def LeastSquaresNonLinear():
        from scipy.optimize import least_squares
        from numpy.random import default_rng

        # coefficient for x^2
        a = 1.0     

        # range of x
        x_min = -1
        x_max = 1
        n_points = 20

        # add noise to data points. We are generating fake data, so it will be too perfect if we don't add noise!
        noise = 0.1

        #figname='LinearLeastSquaresExample_c{}_m{}_noise{}_outliers{}.png'.format(c,m,noise,n_outliers)
        fig, ax1 = plt.subplots(1,1)

        # Function to generate some (N=x.size()) fake data points using the parameters defined above.
        def gen_data(x, a, noise=0.):
        	# set the seed inside here so we get the same data points every time we call gen_data
        	rng = default_rng(1)
        	# a quadratic function of x
        	y_central_vals = a*x**2 
        	y_uncertainties = noise * rng.standard_normal(x.size) 
        	return y_central_vals + y_uncertainties 

        # fixed values for our x-axis
        x_vals = np.linspace(x_min, x_max, n_points-1)    
        # fake data points
        y_vals = gen_data(x_vals, a, noise)

        # residuals
        def residuals(p, x, y):
        	return p[0] * x**2 - y 

        # initial estimate for parameter value
        p0 = np.array([1.0])

        # least squares 
        res_lsq = least_squares( residuals, p0, args=(x_vals, y_vals) )

        fit_result = r"y = {:.2f} x$^2$ ".format(res_lsq.x[0] ) 

        # x-values for function: ten times the number of axis points as for "data"
        x_test = np.linspace(x_min, x_max, n_points*10)

        y_lsq = gen_data(x_test, *res_lsq.x)

        ax1.plot(x_vals, y_vals, color='black',marker='o',linestyle='',label='data')
        ax1.plot(x_test, y_lsq, linestyle='--', color='r',label='LS: {}'.format(fit_result) )

        y_min = np.minimum(y_vals, y_lsq[0::11])
        y_max = np.maximum(y_vals, y_lsq[0::11])

        # draw a vertical line indicating the residual: difference between data and fit
        ax1.vlines(x=x_vals, ymin=y_min, ymax=y_max,color="blue", linestyle=":",linewidth=2,label='residuals')

        ax1.set_xlabel("x",fontsize=20)
        ax1.set_ylabel("y")

        plt.show()

    LeastSquaresNonLinear()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ###**Solution 7.2 Likelihood for Normal Sigma**###

    Python example below - there is no limit to the ways in which you can do this! Please see EstimatorsIntro.py for properly commented example.
    """)
    return


@app.cell
def _(np, plt):
    # Likelihood Example

    def PlotLikelihoodAndFriends():
        from scipy.stats import norm
        import scipy.optimize as opt

        # true params
        mu=1
        sigma=5
        # 100 values of sigma for our x-axis
        sigmas=np.linspace(4,6,100) 
        # fake data
        data = norm.rvs(loc=mu, scale=sigma, size=100)

        # likelihood for a given sigma is the product (over measurements) of PDF evaluated with that mu
        L = lambda x, sigma : np.prod( norm.pdf(x,mu,sigma) )

        # calculate likelihood for every value of mean (returns array of 100 vals)
        L_sigmas = [L(data,sigma) for sigma in sigmas]

        # log likelihood is the sum of the logs of the PDF
        lnL = lambda x, sigma : np.sum( np.log( norm.pdf(x,mu,sigma) ) )
        # identical to lnL = lambda x, mu : np.log( np.prod( norm.pdf(x,mu,sigma) ) )

        # calculate log likelihood for every value of mean (returns array of 100 vals)
        lnL_sigmas = [lnL(data,sigma) for sigma in sigmas]
        # negative log likelihood
        neglnL_sigmas = [-1*lnL(data,sigma) for sigma in sigmas]

        lab =r'$\hat{\theta}$'

        # Plot the likelihood as a function of sigma, with mu fixed
        fig, ax1= plt.subplots() 
        ax1.plot(sigmas, L_sigmas,'r--', label=r'$ \mathcal{L}_{\theta}$')
        ax1.set_ylabel("Likelihood of Data",fontsize=18)
        ax1.set_xlabel(r'$\theta$',fontsize=18)    
        ax1.set_ylim(0)
        ax1.legend(loc="best", prop={'size': 18})
        plt.show()


        # Now the log likelihood
        fig, ax2= plt.subplots() 
        ax2.plot(sigmas, lnL_sigmas,'r--', label=r'$\ln\, \mathcal{L}_{\theta}$')
        ax2.set_ylabel("Log (Likelihood of Data) ",fontsize=18)
        ax2.set_xlabel(r'$\theta$',fontsize=18)    
        ax2.legend(loc="best", prop={'size': 18})
        plt.show()


        # Now the negative log likelihood
        fig, ax3= plt.subplots() 
        ax3.plot(sigmas, neglnL_sigmas,'r--', label=r'$-\ln\, \mathcal{L}_{\theta}$')
        ax3.set_ylabel("Negative Log (Likelihood of Data)",fontsize=18)
        ax3.set_xlabel(r'$\theta$',fontsize=18)

        ax3.legend(loc="best", prop={'size': 18})

        plt.show()

    PlotLikelihoodAndFriends()
    return


if __name__ == "__main__":
    app.run()
