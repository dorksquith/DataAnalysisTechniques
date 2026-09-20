import matplotlib.pyplot as plt

# PLOT
fig, ax = plt.subplots()

ax.plot([1.41, 1.52, 1.56, 1.61, 1.70 ], 
	    [60, 55, 65, 67, 80], 
	    color='skyblue', 
	    marker='*',
	    linewidth=2,
	    label='my lovely data' 
	    )

ax.set_title("Example of a plot")
ax.set_xlabel("Height (m)")
ax.set_ylabel("Weight (kg)")
plt.legend()
#plt.show()
# to save and then close the figure
plt.savefig("MyFirstPlot.png")
plt.clf()

# HIST
import numpy as np
rng = np.random.default_rng()

heights = rng.normal(loc=165, scale=15, size=1000)

fig, ax = plt.subplots()

ax.hist(heights, 
	    bins=10,
	    histtype='stepfilled',
	    facecolor='skyblue', 
	    edgecolor='black', 
	    linewidth=2,
	    alpha=0.5,
	    density=False,
	    label='legend entry'
	    )

ax.set_title("Example of a histogram")
ax.set_xlabel("Height (m)")
ax.set_ylabel("Count/ bin")
plt.legend()
plt.savefig("MyFirstHist.png")
plt.clf()


# SCATTER
heights = rng.normal(loc=165, scale=15, size=1000)

weights = rng.normal(loc=65, scale=10, size=1000)

fig, ax = plt.subplots()

ax.scatter(heights[:500], weights[:500], 
	    color='lightsteelblue',
	    alpha=1, 
	    marker='*',
	    label='Sample 1')

ax.scatter(heights[500:1000], weights[500:1000], 
	    color='mediumvioletred', 
	    marker='o',
	    alpha=0.3, 
	    label='Sample 2')

ax.set_title("Two scatter plots on same axes")
ax.set_xlabel("Height (m)")
ax.set_ylabel("Weight (kg)")
plt.legend()
plt.savefig("MyFirstScatter.png")
plt.clf()


# Describing Data: more than one dimension

heights = rng.normal(loc=165, scale=15, size=1000)

weights = rng.normal(loc=65, scale=10, size=1000)

data = np.stack((heights, weights)) # <= you need nested brackets

print(f" dataset shape: {data.shape }")




# Describing Data: Covariance

# datasets:
a = np.array([1.1, 2.4, 3.3])
b = np.array([ 7.0, 5.0, 2.0])

# calculate the means:
mean_a = np.mean(a)
mean_b = np.mean(b)

# calculate the deviation of each point from the mean:
'''
da_0 = mean_a - a[0] # 1.17
da_1 = mean_a - a[1] # -0.13
da_2 = mean_a - a[2] # -1.03
'''
# one line :  
da =  [mean_a - ai for ai in a]

'''
db_0 = mean_b - b[0] # -2.33
db_1 = mean_b - b[1] # -0.33
db_2 = mean_b - b[2] # 2.67
'''
# one line :  
db =  [mean_b - bi for bi in b]

# multiply together point-by-point:
'''
cov_0 = da_0 * db_0 # -2.72
cov_1 = da_1 * db_1 #  0.04
cov_2 = da_2 * db_2 # -2.76
'''
# one line :  
cov =  [a*b for a,b in zip(da,db) ]


# sum and normalise
norm = 1 / ( len(a)-1 ) # 1/(N-1) unbiased version for small dataset
#cov_ab = norm * (cov_0 + cov_1 + cov_2) #  -2.72
# one line :  
cov_ab =  norm * np.sum(cov)


ab = np.stack((a,b))

print(f" np.cov(ab): {np.cov(ab)}") 




