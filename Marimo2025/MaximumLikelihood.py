import marimo

__generated_with = "0.17.6"
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
def _(np, plt):
    # Least Squares Example

    def LeastSquaresLinear():
        from scipy.optimize import least_squares
        from numpy.random import default_rng
        # parameters for y = mx+c = theta_1 x + theta0
        c = 1.0 # theta_0
        m = -5.2 # theta_1
    
        # range of x
        x_min = 0
        x_max = 1
        n_points = 11
    
        # add noise to data points. We are generating fake data, so it will be too perfect if we don't add noise!
        noise = 2.4
        # for even more realistic data, add some outlier points with outlier_noise_scalefactor times the noise of normal!
        n_outliers = 2
        outlier_noise_sf = 2.
     
        #figname='LinearLeastSquaresExample_c{}_m{}_noise{}_outliers{}.png'.format(c,m,noise,n_outliers)
        fig, ax1 = plt.subplots(1,1)
    
        # Function to generate some (N=x.size()) fake data points using the parameters defined above.
        def gen_data(x, c, m, noise=0., n_outliers=0, outlier_noise_sf=1, seed=314):
    
        	# set the seed inside here so we get the same data points every time we call gen_data
        	rng = default_rng(seed)
    
        	# a straight line, y = mx+c 
        	y_central_vals = m * x +c 
    
        	# rng.standard normal returns random values from a standard normal distribution (mu=0, sigma=1)
        	# we scale the values up by noise to make the level of uncertainty more realistic
        	y_uncertainties = noise * rng.standard_normal(x.size) 
    
        	# outliers: thes are data points with a larger uncertainty than standard (to reflect the reality of a lab!)
        	# randomly select n_outliers data points from the set of data points
        	outliers = rng.integers(0, x.size, n_outliers) 
        	# rescale the uncertainties for these data points
        	y_uncertainties[outliers] *= outlier_noise_sf
    
        	return y_central_vals + y_uncertainties 
    
    
        # fixed values for our x-axis
        x_vals = np.linspace(x_min, x_max, n_points-1)
    
        # fake data points
        y_vals = gen_data(x_vals, c, m, noise, n_outliers, outlier_noise_sf)
    
    
        # now we do the Least Squares fit
        # This function returns an array of the residuals yi - E[y]
        # the function arguments are the parameter(s) p (an array of size n_parameters), and the arrays of x and y values
        def residuals(p, x, y):
        	# for a straight line p[0] is the intercept, p[1] is the gradient
        	# we subtract y from mx + c to get the residual
        	# if the data points had no uncertainties, the residuals would be zero.
        	return p[0] + p[1] * x - y 
    
        # initial estimate for parameter values [c,m]
        p0 = np.array([1.0, 1.0])
    
        # Least Squares algorithm provided by scipy:
        # note that in the least_squares agrument we don't call residuals like (residuals(p,x,y)) but like: 
        # (residuals, p, args=(x, y)) - we are passing the arguments separately from the functional form
        # and specifying x,y as the "args".
        res_lsq = least_squares( residuals, p0, args=(x_vals, y_vals) )
    
        # res_lsq.x : parameter estimate(s)
        #c_est=r"$\widehat{\theta_0}$"
        #m_est=r"$\widehat{\theta_1}$"
        fit_result = r"y = {:.2f} x + {:.2f}".format(res_lsq.x[1],res_lsq.x[0] ) 
        #print(fit_result)
    
        # x-values for function: ten times the number of axis points as for "data"
        x_test = np.linspace(x_min, x_max, n_points * 10)
    
        # this is the true function used to generate the data - we are not using it but in case you want it.
        # y_true = gen_data(x_test, c, m)
    
        # generate y values (10* as many as for data above) using the least squares estimate for the parameters
        # res_lsq.x = c,m . The pointer * means "pack the returned values in a tuple: (c,m)"
        # no point in adding noise or outliers in this case, because we are not generating fake data
        y_lsq = gen_data(x_test, *res_lsq.x)
    
    
        # plot the fake data points
        #ax1.errorbar(x_vals, y_vals, yerr=noise, color='black',marker='o',linestyle='')
        ax1.plot(x_vals, y_vals, color='black',marker='o',linestyle='',label='data')
    
        # plot the function used to generate the fake data points
        #plt.plot(x_test, y_true, 'k', linewidth=2, label='true')
        # plot the Least squares fir
        ax1.plot(x_test, y_lsq, linestyle='--', color='r',label='LS: {}'.format(fit_result) )
    
        # draw the residuals
        # find the min and max y-value between the data and the fit
        # this y_lsq[0::n_points+1] means "give me every Nth value of y_lsq" (N=npoints+1)
        # this is needed because y_lsq has 10* more data points than y_vals
        y_min = np.minimum(y_vals, y_lsq[0::n_points+1])
        y_max = np.maximum(y_vals, y_lsq[0::n_points+1])
        # draw a vertical line indicating the residual: difference between data and fit
        ax1.vlines(x=x_vals, ymin=y_min, ymax=y_max,color="blue", linestyle=":",linewidth=2,label='residuals')
    
        #ax1.set_title('y = {} x + {}'.format(m,c))
        ax1.set_xlabel("x",fontsize=20)
        ax1.set_ylabel("y")
        ax1.set_ylim(-5,5)
        plt.legend()
    
        #plt.savefig(figname, bbox_inches='tight')
        #plt.clf()
        plt.show()

    LeastSquaresLinear()
    return


@app.cell
def _(np, plt):
    # Likelihood Example

    def PlotLikelihoodAndFriends():
        from scipy.stats import norm
        import scipy.optimize as opt

        # true params
        mu=5
        sigma=1
        # 100 values of the mean for our x-axis
        means=np.linspace(4,6,100) 
        # fake data
        data = norm.rvs(loc=mu, scale=sigma, size=100)
    
        # likelihood for a given mu is the product (over measurements) of PDF evaluated with that mu
        L = lambda x, mu : np.prod( norm.pdf(x,mu,sigma) )
    
        # calculate likelihood for every value of mean (returns array of 100 vals)
        L_means = [L(data,mu) for mu in means]
       
        # log likelihood is the sum of the logs of the PDF
        lnL = lambda x, mu : np.sum( np.log( norm.pdf(x,mu,sigma) ) )
        # identical to lnL = lambda x, mu : np.log( np.prod( norm.pdf(x,mu,sigma) ) )
    
        # calculate log likelihood for every value of mean (returns array of 100 vals)
        lnL_means = [lnL(data,mu) for mu in means]
        # negative log likelihood
        neglnL_means = [-1*lnL(data,mu) for mu in means]
    
    
        lab =r'$\hat{\theta}$'
    
        # Plot the likelihood as a function of mu, with sigma fixed
        fig, ax1= plt.subplots() 
        ax1.plot(means, L_means,'r--', label=r'$ \mathcal{L}_{\theta}$')
        ax1.set_ylabel("Likelihood of Data",fontsize=18)
        ax1.set_xlabel(r'$\theta$',fontsize=18)    
        ax1.set_ylim(0)
        #ax1.axvline(x=estimate_mu, color="black", linestyle=":",linewidth=1,label='{}={:.2f}'.format(lab,estimate_mu) )
        ax1.legend(loc="best", prop={'size': 18})
        plt.show()
        #plt.savefig("PLOTDAT6-LikevMU.png", bbox_inches='tight')
        #plt.clf()
    
        # Now the log likelihood
        fig, ax2= plt.subplots() 
        ax2.plot(means, lnL_means,'r--', label=r'$\ln\, \mathcal{L}_{\theta}$')
        ax2.set_ylabel("Log (Likelihood of Data) ",fontsize=18)
        ax2.set_xlabel(r'$\theta$',fontsize=18)    
        #ax2.axvline(x=estimate_mu, color="black", linestyle=":",linewidth=1,label='{}={:.2f}'.format(lab,estimate_mu) )
        ax2.legend(loc="best", prop={'size': 18})
        plt.show()
        #plt.savefig("PLOTDAT6-LogLikevMU.png", bbox_inches='tight')
        #plt.clf()
    
        # Now the negative log likelihood
        fig, ax3= plt.subplots() 
        ax3.plot(means, neglnL_means,'r--', label=r'$-\ln\, \mathcal{L}_{\theta}$')
        ax3.set_ylabel("Negative Log (Likelihood of Data)",fontsize=18)
        ax3.set_xlabel(r'$\theta$',fontsize=18)
        #ax3.axvline(x=estimate_mu, color="black", linestyle=":",linewidth=1,label='{}={:.2f}'.format(lab,estimate_mu) )
    
        ax3.legend(loc="best", prop={'size': 18})
        #plt.savefig("PLOTDAT6-NegLogLikevMU.png", bbox_inches='tight')
        #plt.clf()
        plt.show()

    PlotLikelihoodAndFriends()
    return


if __name__ == "__main__":
    app.run()
