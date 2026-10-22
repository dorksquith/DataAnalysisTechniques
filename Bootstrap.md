---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: 'Python 3'
---

(chapter:Resampling)=
# Resampling

You toss a coin 1000 times, generating a dataset of size 1000. Then you find out you needed 1100 data points, not 1000. Which of these should you do?

1. Find the Maximum Likelihood Estimate for $\hat{p}$, and generate another 100 tosses using Bernoulli($\hat{p}$). 
2. Randomly choose one value from your existing dataset and repeat this 100 times. 

These two options are the same option, phrased differently! The first is **Parametric Resampling** and the second is **Nonparametric Resampling**.

For example:

1. If the MLE for $\hat{p} = 0.54$ you will expect to generate 54 new heads with Bernoulli(0.54).
2. If the MLE for $\hat{p} = 0.54$, this is because there are 540 heads in the dataset. You will expect to draw 54 heads from your random picks.

## The Bootstrap Method

A common reason for wanting a larger dataset is to calculate a Statistic from the data, for example the mean, mode, standard deviation, $\chi^2$.

An example of using the Bootstrap method to increase your dataset is as follows (assume a starting dataset size of $N_d =100$, and a required dataset size of $N_r = 10,000$):

1. Randomly select a value from the data. 
2. Do this 100 times, allowing yourself to select the same measurement more than once.
3. Calculate the statistic of interest from the 100 measurements you randomly selected. 
4. Repeat the above three steps 9999 times.

The point of this process is not to recalculate the statistic - you have already calculated that from the original dataset. The point is to estimate the statistical uncertainty on that measured value.

### Bootstrap Example: step-by-step

Let's generate some data using a seed such that every time we execute this function we get the same result.

```{code-cell} python
import numpy as np

# make a local instance of numpy's default (PCG64) pseudorandom number generator and set the seed - we only ever do this to get reproducible results for testing and for demonstrations.

#rng = np.random.default_rng(seed=42)
rng = np.random.default_rng(111)

# generate some fake data, normally distributed using arbitrary parameter choices

from scipy.stats import norm

norm_mean = 6.0
norm_sigma = 3.77
ndata = norm.rvs(loc=norm_mean, scale=norm_sigma, size=100, random_state=rng)

print(f"Generated Norm data with mu={norm_mean}, sigma={norm_sigma}")

```

Now we have a fake dataset of 100 measurements, let's bootstrap!

```{code-cell} python

def LongBootstrap(data):

  # Our statistic of interest is the standard deviation
  print( f"sample sigma: {np.std(data):.3f}")

  # Make an empty container to put our bootstrap stats in
  sigmas=[]

  # loop over the number of boostraps we want
  for i in range(9999):

    # Draw 100 random measurements from our dataset 

    # Important! values can be drawn more than once because we have replace=True. If we don't use replace=True, we will just get the original dataset back.

    rdata = rng.choice(data,replace=True, size=100)

    # Get the standard deviation of this set of 100 measurements

    rsigma = np.std(rdata)

    # Put the standard deviation in the container

    sigmas.append(rsigma)

    # now to the next of the 9999 iterations in the loop

  # end of loop, print the statistic with its uncertainty

  print( f"bootstrap estimate for sigma: {np.mean(sigmas):.3f} +/- {np.std(sigmas):.3f} ")

  return sigmas


# call the bootstrap function on the fake data we generated above

byhand_sigmas = LongBootstrap(ndata)

```


### Bootstrap Example: `scipy`

```{code-cell} python

def ScipyBootstrap(data):

  # thank you scipy for providing yet another lovely method

  from scipy.stats import bootstrap

  # first argument is data, second is the statistic of interest, optional rng parameter is for reproducibility

  sbs = bootstrap( (data,), np.std, rng=rng )

  sp_mean = np.mean(sbs.bootstrap_distribution)
  
  sp_ser = sbs.standard_error

  sp_ci_lo, sp_ci_hi = sbs.confidence_interval

  print( f"scipy bootstrap estimate for sigma: {sp_mean:.3f} +/- {sp_ser:.3f}")

  print( f"95% Confidence Interval: {sp_ci_lo:.3f}--{sp_ci_hi:.3f} ")

  return sbs.bootstrap_distribution

# call the function
scipy_sigmas = ScipyBootstrap(ndata)
```

### Parametric Bootstrap Example

If you know the underlying distribution (eg Norm in this case), you can do a parametric bootstrap using the parameters estimated from the data.

```{code-cell} python

def ParaBootstrap(data):

  psigmas=[]
  for i in range(9999):
    
    respin_data = norm.rvs(loc=np.mean(data), scale=np.std(data,ddof=1), size=100)

    psigmas.append( np.std(respin_data) )

  print( f"parametric bootstrap estimate for sigma: {np.mean(psigmas):.3f} +/- {np.std(psigmas):.3f} ") 

  return psigmas

# call the function

parametric_sigmas =ParaBootstrap(ndata)
```

### Plot Comparing Methods

This is a frequentist process, so the true value of $\sigma$ is expected to lie within our error margin in 68% of repeated experiments.

```{code-cell} python

def PlotBootsraps(a,b,c):

  import matplotlib.pyplot as plt

  fig, ax = plt.subplots()

  # don't have to define bins, but mpl makes strange choices
  sm = np.mean(a)
  ss = np.std(a)
  bins=np.linspace(sm-3*ss, sm+3*ss, 30)

  labs = ['byhand', 'scipy', 'parametric']
  cols = ['darkgreen', 'dodgerblue', 'grey']
  styles = ['-','--',':']
  for i, d in enumerate([a,b,c]):
    ax.hist(d, 
            fc='none',
            lw=2,
            ls=styles[i],
            ec=cols[i], 
            bins=bins,
            label=f'{labs[i]} {np.mean(d):.3f}+/-{np.std(d):.3f}')

  ax.set_xlabel(r"Bootstrap $\sigma$")

  # don't want legend to overlap with plot. Extend y-axis by 20%
  ymax = ax.get_ylim()[1]
  ax.set_ylim(0, ymax*1.2)

  plt.legend()
  plt.show()


# call the function

PlotBootsraps(byhand_sigmas,scipy_sigmas,parametric_sigmas)
```

