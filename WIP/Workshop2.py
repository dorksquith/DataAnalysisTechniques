# Alf's rolo obsession

import numpy as np

# The two datasets as numpy arrays
xA = np.array([52.010, 52.041, 52.105, 51.998, 51.981])
xL = np.array([52.05, 52.03, 52.07, 51.90, 51.94])

# It is important to use the unbiased version of variance/ standard deviation (with ddof=1) because these are very small datasets.

# Function to get the mean and sem of a given 1D array of data
def get_stats(x):
	mean = np.mean(x)
	sigma = np.std(x, ddof=1)
	N = len(x)
	sem = sigma / np.sqrt(N)
	return mean, sem, N

# call the function
meanA, semA, NA = get_stats(xA)

meanL, semL, NL = get_stats(xL)

print(f" Alf's data mean: {meanA:.3f} +/- {semA:.3f}")

print(f" Lily's data mean: {meanL:.3f} +/- {semL:.3f}")


# 2D array of both datasets
xAL = np.stack((xA, xL))

# print the 2D array and note the structure

print(f" xAL: {xAL} ")

# notice the len, size, and shape

print(f" len(xAL): {len(xAL)} ")

print(f" xAL.size: {xAL.size} ")

print(f" xAL.shape: {xAL.shape} ")

# function to calculate the weighted mean

def weighted_mean( data2D ):

	# axis=1: because each dataset is a row
	m = np.mean( data2D, axis=1)

	# does it make sense to you that this returns two numbers?
	# what happens if you don't provide an axis?
	# what if you have axis=0 ?

	 
	# unbiased variance
	v = np.var( data2D, axis=1,ddof=1)
	# size of datasets
	N = np.array( [len(data2D[0]) , len(data2D[1])] )
	# variances on the means
	vm = v/N
	# weights = reciprocal of the variances on the means
	w =1/vm

	print(f"weights: {w}")
	# have a look at w : it is an array just like v
	# compare with list comprehension:
	# lw = [ 1/lv for lv in list(v) ]
	# using numpy arrays means we don't need this ^

	wm = np.sum( m * w ) / np.sum(w)

	sem = np.sqrt( 1 / np.sum(w) )

	return wm, sem


wm, wsem = weighted_mean( xAL )

print(f"Weighted mean: {wm:.3f} +/- {wsem:.3f}")

# Alf's measurements have smaller variance, which means smaller SEM
# Alf's measurements will have a larger weight for this reason. Not because they have an additional sig fig.

