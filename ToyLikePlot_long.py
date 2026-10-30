# plots of single data point and PMF to illustrate Likelihood in toy example

# Imports at the top so functions can be called interactively or from a notebook
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

def main():

    # trial parameter values
    theta_a = 10
    theta_b = 8
    # fake data measurements
    k1 = 4
    k2 = 14
    # for N=21 fake data poisson
    theta_fake_data=9.3


    # plot a single data point, k1 (if kj=1, else k2), with two PMFs
    toy_likelihood_example(theta_a=theta_a, theta_b=theta_b, kj=1)
    
    # plot a single data point, k2, with two PMFs
    toy_likelihood_example(theta_a=theta_a, theta_b=theta_b, kj=2)
    
    # plot two data points, k1 and k2, with one PMF (theta_a)
    toy_likelihood_example(theta_a=theta_a, two_data=True)

    # plot two data points, k1 and k2, with one PMF (theta_b)
    toy_likelihood_example(theta_b=theta_b, two_data=True)


    # two data points as above, but with Likelihood calc on plot.

    plot_multi_data_likelihood(
        [k1,k2], theta=theta_a,
        data_label=r"Data: $k_1,k_2$, $N=2$",
        figname=f"ToyLike_TwoData_withLike_theta{theta_a}.png"
    )
    
    # same thing for theta_b
    c_b = 'orange'
    plot_multi_data_likelihood(
        [k1,k2], theta=theta_b,
        data_label=r"Data: $k_1,k_2$, $N=2$",
        figname=f"ToyLike_TwoData_withLike_theta{theta_b}.png",
        color=c_b
    )
    
    # what if we measure 21 different values of k, all once?
    # this is uniform data: flat probability
    uniform_data = np.arange(1, 22)   # k = 1, 2, ..., 21

    # plot the uniform dataset as green markers, and the PMF(theta_a)
    # calculate the total Likelihood (Joint probability) of all 21 data points

    plot_multi_data_likelihood(
        uniform_data, theta=theta_a,
        data_label=r"Data: $K \sim \mathrm{Uniform}(1,21)$, $N=21$",
        figname=f"ToyLike_Uniform_theta{theta_a}.png"
    )


    # same thing for theta_b
    plot_multi_data_likelihood(
        uniform_data, theta=theta_b,
        data_label=r"Data: $K \sim \mathrm{Uniform}(1,21)$, $N=21$",
        figname=f"ToyLike_Uniform_theta{theta_b}.png",
        color='orange'
    )

    # how about 21,00?
    # this is uniform data: flat probability
    uniform_data_100 = 100*np.arange(1, 22)   # k = 1, 2, ..., 21

    # plot the uniform dataset as green markers, and the PMF(theta_a)
    # calculate the total Likelihood (Joint probability) of all 21 data points

    plot_multi_data_likelihood(
        uniform_data_100, theta=theta_a,
        data_label=r"Data: $K \sim \mathrm{Uniform}(1,21)$, $N=2100$",
        figname=f"ToyLike_Uniform_theta{theta_a}_100.png"
    )


    # same thing for theta_b
    plot_multi_data_likelihood(
        uniform_data_100, theta=theta_b,
        data_label=r"Data: $K \sim \mathrm{Uniform}(1,21)$, $N=2100$",
        figname=f"ToyLike_Uniform_theta{theta_b}_100.png",
        color='orange'
    )



    # more realistic poisson data, rather than uniform
    # generate 21 fake data points drawn from poisson(9.3) 
    poisson_data = poisson(mu=theta_fake_data).rvs(size=21, random_state=42)


    # plot the poisson dataset as green markers, and the PMF(theta_b)
    # calculate the total Likelihood (Joint probability) of all 21 data points
    plot_multi_data_likelihood(
        poisson_data, theta=theta_a,
        data_label=rf"Data: $K \sim \mathrm{{Poisson}}({{{theta_fake_data}}})$, $N=21$",
        figname=f"ToyLike_Poisson_theta{theta_a}.png"
    )


    # same thing for theta_a
    plot_multi_data_likelihood(
        poisson_data, theta=theta_b,
        data_label=rf"Data: $K \sim \mathrm{{Poisson}}({{{theta_fake_data}}})$, $N=21$",
        figname=f"ToyLike_Poisson_theta{theta_b}.png",
        color='orange'
    )
  
    # generate 2100 fake data points drawn from poisson(9.3) 
    poisson_data_100 = poisson(mu=theta_fake_data).rvs(size=2100, random_state=42)


    # plot the poisson dataset as green markers, and the PMF(theta_b)
    # calculate the total Likelihood (Joint probability) of all 21 data points
    plot_multi_data_likelihood(
        poisson_data_100, theta=theta_a,
        data_label=rf"Data: $K \sim \mathrm{{Poisson}}({{{theta_fake_data}}})$, $N=2100$",
        figname=f"ToyLike_Poisson_theta{theta_a}_100.png"
    )


    # same thing for theta_a
    plot_multi_data_likelihood(
        poisson_data_100, theta=theta_b,
        data_label=rf"Data: $K \sim \mathrm{{Poisson}}({{{theta_fake_data}}})$, $N=2100$",
        figname=f"ToyLike_Poisson_theta{theta_b}_100.png",
        color='orange'
    )  

# helper function
def kvals(dist):
    """Return an integer array of k values spanning 0.01%–99.99% of the distribution's CDF."""

    # Value of rv at which cdf is 0.01% of total
    k_low = dist.ppf(0.0001)

    # Value of rv at which cdf is 99.99% of total
    k_high = dist.ppf(0.9999)

    # Values for the RV (x-axis) must be integers for the Poisson distribution
    k = np.arange(0, int(k_high))

    return k

# helper function
def plot_toy_labelled(ax, kj, like, j, thetalab, color, label_data):
    """Draw vertical line, scatter point, horizontal line, and likelihood label on ax.

    Parameters
    ----------
    ax         : matplotlib Axes
    kj         : int   — data point value
    like       : float — likelihood (PMF) value at kj
    j          : int   — data point index (used in subscript label, e.g. k_1)
    thetalab   : str   — LaTeX string for the parameter label
    color      : str   — colour for this parameter's elements
    label_data : bool  — whether to add a legend entry for the data line
    """

    '''
    print(f" kj = {kj}")
    print(f" like = {like}")
    print(f" j = {j}")
    print(f" thetalab = {thetalab}")
    '''

    # To embed a variable as a subscript in an f-string LaTeX label,
    # use triple curly brackets: {{{ }}}
    # The outer {{ }} produce a literal { }; the inner {j} is the variable.
    d = rf"$k_{{{j}}}$"   # e.g. $k_1$
    dlabel = "Data " + d

    # Different line style for k1 and k2
    ls = '--' if j == 1 else ':'

    # Vertical line at data point kj — label only on first appearance
    if label_data:
        ax.axvline(kj, color='green', ls=ls, lw=2, label=dlabel)
    else:
        ax.axvline(kj, color='green', ls=ls, lw=2)

    # Green marker at y=0 to indicate the data point position
    sc = ax.scatter(kj, 0, color='green', marker='o', s=50, zorder=10)
    sc.set_clip_on(False)

    # Horizontal line indicating the likelihood value
    ax.plot([0, kj + 1], [like, like], color=color, ls=ls, lw=2)


    likelabel = r"$\mathcal{L}$" + f"( {thetalab}; {d})" + r" $\approx$"+ f" {like:.2e}"

    # Text label with a coloured box for clarity
    ax.text(
        kj + 1, like, likelabel,
        fontsize=14,
        va='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, lw=1)
    )


# helper function
def plot_multi_data_likelihood(data, theta, data_label, figname, color='dodgerblue'):
    """Plot a Poisson PMF with likelihood markers for multiple data points.

    Shows a scatter marker at each data point's PMF value and annotates
    the total likelihood L and log L.

    Parameters
    ----------
    data       : array-like — observed k values (integers)
    theta      : float      — Poisson parameter (mu) for the reference PMF
    data_label : str        — legend label describing the dataset
    figname    : str        — output filename (.png)
    color      : str        — colour for the PMF bars and data markers
    """

    dist = poisson(mu=theta)
    k = kvals(dist)
    pmf = dist.pmf(k=k)

    # Compute total likelihood and log-likelihood for this dataset
    pmfs_data = dist.pmf(data)
    L = np.prod(pmfs_data)
    logL = np.sum(np.log(pmfs_data))

    print(f"data: {data}, theta: {theta}")
    print(f"pmfs_data: {pmfs_data}")
    print(f"L: {L}")

    
    fig, ax = plt.subplots(1, 1)

    alpha = 0.3
    plabel = r"$p_K(k;\theta = $" + f"{theta} )"

    ax.bar(k, pmf, color=color, edgecolor=color, alpha=alpha, label=plabel)

    # Stack green markers at y=0 to indicate data positions.
    # When multiple measurements share the same k value, markers are stacked
    # vertically so none are hidden.
    offset = pmf.max() * 0.05   # vertical spacing between stacked markers
    unique_ks, counts = np.unique(data, return_counts=True)
    for ki, n in zip(unique_ks, counts):
        for i in range(n):
            sc = ax.scatter(ki, i * offset, color='green', marker='o', s=50, zorder=10)
            sc.set_clip_on(False)

    # Dummy legend entry for the data markers
    ax.scatter([], [], color='green', marker='o', s=50, label=data_label)


    # Likelihood annotation box (top-right corner, in axes coordinates)
    liketext = (
        r"$\mathcal{L}(\theta = $" + f"{theta}" + r"$;\,k)$"
        + r" $\approx$"+ f"{L:.2e}"
        + "\n" + r"$\log\mathcal{L}$"+ r" $\approx$" + f" {logL:.1f}"
        + "\n" + r"$NLL$" + r" $\approx$"+ f" {-logL:.1f}"
    )
    ax.text(
        0.97, 0.75, liketext,
        transform=ax.transAxes,
        fontsize=14, va='top', ha='right',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="green", lw=1.5)
    )

    ax.set_xlabel("K", fontsize=16)
    ax.set_ylabel("PMF", fontsize=16)

    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.4)

    xmax = ax.get_xlim()[1]
    ax.set_xlim(0, 26)

    plt.legend(fontsize=14, loc='upper left')
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

# helper function

def toy_likelihood_example(theta_a=None, theta_b=1, two_data=False, kj=1):
    """Plot PMF(s) with likelihood markers for one or two data points.

    Parameters
    ----------
    theta_a  : float — first Poisson parameter value
    theta_b  : float — second Poisson parameter value (used when two_data=False)
    two_data : bool  — if True, show two data points on one PMF;
                       if False, show one data point on two PMFs
    """

    poisson_b = poisson(mu=theta_b)
    if theta_a is not None:
        poisson_a = poisson(mu=theta_a)
    k = kvals(poisson_a if theta_a is not None else poisson_b)

    pmf_a = poisson_a.pmf(k=k) if theta_a is not None else None
    pmf_b = poisson_b.pmf(k=k)

    # k1 and k2 are fixed rather than randomly drawn so the plot is always clear.
    # (The random alternative would be: k1, k2 = poisson_a.rvs(size=2))
    #k1 = 4
    
    # if kj is not passed, default is kj==1
    if(kj==1):
        k1 = 4
        k2 = 14
        kk = 2
    else:
        k1 = 14
        k2 = 4
        kk = 1

   
    
    # Likelihoods of each data point under theta_b
    L1b = poisson_b.pmf(k1)
    L2b = poisson_b.pmf(k2)

    # Likelihoods of each data point under theta_a (only when theta_a is provided)
    if theta_a is not None:
        L1a = poisson_a.pmf(k1)
        L2a = poisson_a.pmf(k2)

    fig, ax = plt.subplots(1, 1)

    c_a = 'dodgerblue'
    alpha = 0.3
    plabel_a = r"$p_K(k;\theta_a = $"+f"{theta_a} )"
    thetalabel_a = r"$\theta_a$"

    c_b = 'orange'
    plabel_b = r"$p_K(k;\theta_b = $"+f"{theta_b} )"
    thetalabel_b = r"$\theta_b$"

    if theta_a is None:
        ax.bar(k, pmf_b, color=c_b, edgecolor=c_b, alpha=alpha, label=plabel_b)
        plot_toy_labelled(ax, k1, L1b, kj, thetalab=thetalabel_b, color=c_b, label_data=True)
    else:
        ax.bar(k, pmf_a, color=c_a, edgecolor=c_a, alpha=alpha, label=plabel_a)
        plot_toy_labelled(ax, k1, L1a, kj, thetalab=thetalabel_a, color=c_a, label_data=True)

    if two_data:
        # if theta_a not passed, compare data with theta_b
        if theta_a is None:
            figname = f"ToyLike_TwoData_theta{theta_b}.png"
            plot_toy_labelled(ax, k2, L2b, kk, thetalab=thetalabel_b, color=c_b, label_data=True)

        # else compare data with theta_a
        else:
            figname = f"ToyLike_TwoData_theta{theta_a}.png"
            plot_toy_labelled(ax, k2, L2a, kk, thetalab=thetalabel_a, color=c_a, label_data=True)
    else:
        
        figname = f"ToyLike_TwoParams_k1_{k1}.png"
        ax.bar(k, pmf_b, color=c_b, edgecolor=c_b, alpha=alpha, label=plabel_b)
        plot_toy_labelled(ax, k1, L1b, kj, thetalab=thetalabel_b, color=c_b, label_data=False)

    ax.set_xlabel("K", fontsize=16)
    ax.set_ylabel("PMF", fontsize=16)

    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.2)

    xmax = ax.get_xlim()[1]
    ax.set_xlim(0, 26)

    plt.legend(fontsize=14)
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    main()


