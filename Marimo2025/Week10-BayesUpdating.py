import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    # Example of Bayesian updating with:
    # Poisson data (fake, of course!)
    # Gamma Prior (because Gamma is conjugate for Poisson)
    # Negative Binomial Predictive Posterior (because integral of poisson*gamma = negative binomial)

    # usual imports
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import gamma, poisson, nbinom
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 16,
             'axes.labelsize': 16,
             'axes.titlesize':16,
             'xtick.labelsize':14,
             'ytick.labelsize':14}
    pylab.rcParams.update(params)
    return gamma, nbinom, np, plt, poisson


@app.cell
def _(np):
    #=============================================
    # USER PARAMETERS - TO BE PLAYED WITH
    #=============================================
    # SIZE OF CREDIBLE INTERVAL (0.95 recommended, can try 0.9 - 0.99 )
    ci = 0.95
    #print(f"{100*ci:.2f} % Credible Intervals")

    # FAKE DATA SCALE PARAMETER, Data ~ Poisson(mu) 
    # Note that we use lambda for parameter in lectures.
    # We cant use lambda in python because it is reserved 
    mu=15

    # PRIOR HYPER-PARAMETERS, Prior ~ Gamma(a,b)
    a=2 # 15 is the best guess, as it is the true value of mu
    b=1

    # WHICH VALES OF N DO WE WANT TO PLOT
    # We make a plot for n=1 data point, n=2, n=3 etcetera, with max n=100
    # You can change this, but will get python complaints if you have too many
    #n_for_plots = [1,2,3,4,5,10,20,50,100]
    n_for_plots = [1,2,3,4,5,50]

    # HOW MANY LOOPS DO WE WANT?
    # If the max is less than the largest n in n_for_plots, you won't get that plot
    ncalcs=np.arange(1, 1_000)
    return a, b, ci, mu, n_for_plots, ncalcs


@app.cell
def _(a, b, gamma, mu, np, poisson):
    #=============================================
    # SETUPS
    #=============================================

    # ----------------------- #
    # FAKE DATA ~ Poisson(mu)
    # ----------------------- #  
    pois  = poisson( mu=mu )
    p_rvs = pois.rvs( size=5_000 )

    # FAKE DATA HISTOGRAM BINS
    ybins=np.arange( 0, 2*mu+10 )

    # ----------------------- #
    # PRIOR  ~ Gamma(a,b)
    # ----------------------- # 
    # b in math Gamma (our version in lectures) is a rate hyperparameter
    # scipy wants a scale, so we invert: scale=1/rate.
    gam = gamma(a=a, scale=1/b)

    # Y VALUES FOR PRIOR
    ymin = min(ybins)
    ymax = max(ybins-10)
    y = np.linspace(ymin, ymax, 5_000)

    # PRIOR PDF
    prior = gam.pdf(y)
    prior_legend=f'Prior : Gamma({a}, {b})'
    return p_rvs, pois, prior, prior_legend, y, ybins


@app.cell
def _(
    a,
    b,
    ci,
    gamma,
    mu,
    n_for_plots,
    nbinom,
    ncalcs,
    np,
    p_rvs,
    plt,
    pois,
    prior,
    prior_legend,
    y,
    ybins,
):
    #=============================================
    # THE LOOP
    # make an empty container to store our data
    dat=[]
    #=============================================
    for n in ncalcs:
    
        # each time round the loop add one data point
        # we take one value from the p_rvs we made above
        r = np.random.choice(p_rvs, size=1, replace=False)
    
        # put the data point in the data container	
        dat.append(r[0])
        #print("dat ",dat)
    
        # get the mean value of our data points
        mean_y = np.mean(dat)
    
        # we don't want to make a plot for every single data point
        # just want plots if n is in our n_for_plots list
        # if it is not, our work is done and we go to next loop iteration
        if ( n not in n_for_plots ): 
            continue
    
        # PLOT SETUP 
        # We are having two subplots (2,1: two rows, 1 column)
        fig, axs = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': [1,1]})
    
    
        #=============================================
        # PLOT 1: Data, Prior, Posterior on same plot
        #=============================================
        ax1=axs[0]
    
        # figure name specifies the a,b, and the n (number of data points)
        figname=f"W10-PG-a{a}-b{b}-n{n}.png"
    
        # ---------------- #
        # PRIOR Gamma(a,b)
        # ---------------- #
    
        # Plot the prior (which is the same every time!)
        ax1.plot(y, prior, color='blue',linewidth=2,linestyle=':', label=prior_legend)
    
        # ---------- #
        # DATA
        # ---------- #
    
        # Plot the data histogram, which will have n entries
        ax1.hist(dat, bins=ybins, density=True,color='darkgreen', linewidth=2,linestyle='-',alpha=0.3,label=f'Data, n={n}')
    
        # ------------------------------ #
        # POSTERIOR Gamma(a_post, b_post)
        # ------------------------------ #
    
        # PARAMETERS 
        a_post = a + ( n * mean_y )
        b_post = b + n
    
        # POSTERIOR PDF
        posterior = gamma.pdf(y, a=a_post, scale=1/b_post )
    
        # PLOT
        ax1.plot(y, posterior, color='mediumvioletred',linewidth=2,linestyle='-', label='Posterior')
    
        # ------------------------------ ------------------------------ #
        # PRINT MAXIMUM A POSTERIORI ESTIMATE AND CREDIBLE INTERVAL
        # ------------------------------ ------------------------------ #
    
        # The mode for a Gamma(a,b) distribution is (a-1) / b
        map_est = ( a_post - 1 ) / ( b_post )
    
        lower_edge_ci = 0.5 * ( 1-ci )
        upper_edge_ci = 1 - 0.5 * (1-ci)
    
        # ppf(q, a, loc=0, scale=1): ppf(probability, shape param, 0, 1/rate_param)
        a1=gamma.ppf (lower_edge_ci , a_post, 0, 1 / b_post ) 
        a2=gamma.ppf (upper_edge_ci, a_post, 0, 1 / b_post )
    
        print(f"n= {n}, MAP: {map_est:.2f} , {100*ci:.0f} % CI: [{a1:.2f}, {a2:.2f}]")
    
        # ---------- #
        # STYLE
        # ---------- #
        # AXIS LABELS ETC
        ax1.set_xlabel(r"$\theta$", fontsize=14)
        ax1.set_ylabel("Density", fontsize=14)
    
        # LINE TO INDICATE TRUE PARAMETER VALUE USED FOR FAKE DATA
        lam=r"$\theta$"
        true_label=f"True {lam}={mu}"
        ax1.axvline(x=mu, color="green", linestyle="--",linewidth=2,label=true_label)
    
        # AXIS LIMITS
        ax1.set_ylim(0)
        ax1.set_xlim(0,40)
    
        # LEGEND
        ax1.legend(loc='upper right', frameon=False,fontsize=14 )
    
        #====================================================
        # PLOT 2: Truth and Predictive Posterior on same plot
        #====================================================
        ax2=axs[1]
    
        # ------------------------------- #
        # PREDICTIVE POSTERIOR NBinom(n,p)
        # ------------------------------- #
        # Predictive Posterior distribution is Negative Binomial (n,p)
    
        # PARAMETERS
        nbn_n = a + ( n * mean_y )
        nbn_p = ( b + n ) / ( b + n + 1 )
    
        nbin = nbinom(nbn_n, nbn_p)
  
        # PREDICTIVE PDF
        prediction = nbin.pmf(ybins)
    
        # PLOT
        ax2.plot(ybins, prediction, '*', color='mediumvioletred',label="Predictive")
    
        # ------------------------------- #
        # TRUE DISTRIBUTION: POISSON
        # ------------------------------- #
        pmf = pois.pmf(ybins)
        ax2.plot(ybins, pmf, 'darkgreen', linestyle='--', label="True")
    
        # ---------- #
        # STYLE
        # ---------- #
    
        # AXIS LABELS
        ax2.set_xlabel(r'$\tilde{y}$')
        ax2.set_ylabel("Probability", fontsize=14)
    
        # AXIS LIMITS
        ax2.set_xlim(0,40)
        ax2.set_ylim(0)
    
        # LEGEND
        ax2.legend(loc='upper right', frameon=False,fontsize=14 )
    
        # SAVE FIGURE
        plt.tight_layout()
        #plt.savefig(figname, bbox_inches='tight')
        #plt.clf()
        plt.show()
    return


if __name__ == "__main__":
    app.run()
