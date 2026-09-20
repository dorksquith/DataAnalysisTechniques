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
