import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    from numpy import random

    return np, random


@app.cell
def _(random):
    # Continuous Random Variables

    # Draw 10 crvs in range (0,100)
    crvs_x = random.uniform(0,100,10)

    # Look at them
    print("Continuous random variables x1:\n",crvs_x)
    return (crvs_x,)


@app.cell
def _(random):
    # Discrete Random Variables

    # Draw 10 integer rvs in range (0,100) from randint
    drvs_x = random.randint(0,100,10)

    # Look at them
    print("Discrete random variables:\n",drvs_x)


    return (drvs_x,)


@app.cell
def _(drvs_x):
    # Note that a Discrete RV does not need to be an integer, but you will only get integers from numpy (or scipy) random methods. 
    hacked_drvs_x = drvs_x/10
    print("These are also discrete random variables:\n",hacked_drvs_x)
    return


@app.cell
def _(crvs_x, np):
    #The Mean of a Dataset

    # longhand
    mean_x = np.sum(crvs_x) / np.size(crvs_x)
    print("Mean x1: ",mean_x)
    # using numpy mean
    numpy_mean_x = np.mean(crvs_x)
    print("Mean x1 using numpy: ",numpy_mean_x)
    return


@app.cell
def _(np, random):
    # The Binned mean (there are a number of different approaches to this with python)

    # 10 values we made previously are not enough for binning example - make 100 values
    big_x = random.uniform(0,100,100)
    print("New dataset has mean x: {}  and N data points: {}".format(np.mean(big_x), np.size(big_x)))

    return (big_x,)


@app.cell
def _(big_x, np):
    # we will use ten bins
    nbins = 10

    # Use np.histogram to bin our data. This returns h =[counts][bin edges].
    h = np.histogram(big_x, bins=nbins)

    # h[0] is the array of bin counts
    bin_counts = h[0]
    # h[1] is the bin edges. There are n+1 bin edges for n bins
    bin_edges = h[1]

    # Get the bin centers
    # bin edges upper - lower:  bin width
    bin_width = bin_edges[1]-bin_edges[0] 
    # use all the bin edges except the zeroth [1:], and subtract half the bin width
    bin_centers = bin_edges[1:] - bin_width/2 

    # Calculate the binned mean for this dataset with this binning
    binned_mean=(1/len(big_x))*np.sum(bin_counts*bin_centers)

    print("New dataset has BINNED mean x: {}".format( np.mean(binned_mean) ))
    return


@app.cell
def _():
    # ASIDE Expected value of Dataset
    # Trivial example of how many sixes we expect in 100 dice rolls
    # To use the scipy stats expect function we need to turn our flat probability =1/6 into a PDF
    from scipy.stats import rv_discrete
    class fx_pmf(rv_discrete):
        def _pmf(self,x):
            return 100*x/6
    toy_pmf = fx_pmf(a=0, b=1)
    expect = toy_pmf.expect()
    print("Expected number of sixes in 100 dice rolls= {}".format(expect) )

    return


@app.cell
def _(crvs_x, np):
    # Variance

    # longhand
    var_x = np.sum((crvs_x-np.mean(crvs_x))**2) / np.size(crvs_x)
    print("Variance of dataset V[x1]: ",var_x)
    # using numpy var
    numpy_var_x = np.var(crvs_x)
    print("Variance of dataset V[x1] using numpy: ",numpy_var_x)

    return


@app.cell
def _(crvs_x, np, random):
    # Weighted Mean
    crvs2_x = random.uniform(0,100,10)
    crvs3_x = random.uniform(0,100,10)

    weights = np.array([1./np.var(crvs_x), 1./np.var(crvs2_x), 1./np.var(crvs3_x)])
    means = np.array([np.mean(crvs_x), np.mean(crvs2_x), np.mean(crvs3_x)])
    sigmas = np.array([np.std(crvs_x), np.std(crvs2_x), np.std(crvs3_x)])

    weighted_mean =np.sum(weights*means)/np.sum(weights)
    print("Weighted mean: {:.3f}".format(weighted_mean) )
    print("Individual dataset means:")
    ids=np.array([1,2,3])
    for id,m,s in zip(ids, means,sigmas):
        print("x{} = {:.3f} +/- {:.3f}".format(id,m, s))

    return


@app.cell
def _(crvs_x, np):
    # Reminders to compare with below
    print("Properties of random dataset: ")
    print("Mean x1: {:.2f}".format( np.mean(crvs_x) ) )
    print("Variance x1: {:.2f}".format( np.var(crvs_x) ) )
    print("Sigma x1: {:.2f}".format( np.std(crvs_x) ) )

    # Defs not used in this module
    print("\nDefinitions not used in this module: ")
    # Median: 50:50 value
    print("Median x1: {:.2f}".format( np.median(crvs_x) ) )
    # Mode: most probable value
    from scipy.stats import mode
    print("Mode x1: {:.2f}".format( mode(crvs_x)[0] ) )
    # RMS
    print("RMS x1: {:.2f}".format( np.sqrt(np.mean(crvs_x**2))) )
    # Geometric mean
    print("Geometric mean x1: {:.2f}".format( np.prod(crvs_x)**(1/np.size(crvs_x)) ) )
    #FWHM
    print("FWHM x1: {:.2f}".format( np.std(crvs_x)*2*np.sqrt(2*np.log(2)) ) )
    # Skewness
    from scipy.stats import skew
    print("Skewness x1: {:.2f}".format( skew(crvs_x) ) )

    return


if __name__ == "__main__":
    app.run()
