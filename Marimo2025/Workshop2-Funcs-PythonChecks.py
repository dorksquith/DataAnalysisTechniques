# workshop2 week 3 Funcs - python checks

from scipy.stats import uniform
import numpy as np 

# RV distibuted uniformly in the range [loc, loc+scale] = [-1,1]
X = uniform(loc=-1,scale=2)

# check we did what we wanted in terms of range
lb,ub=X.support()
print("Range of X: {}, {}".format(lb,ub))

# 1000 data points should be enough for sanity checks
x = X.rvs(size=1000)

# pdf
u_pdf = X.pdf(x)

# sanity check for F_X = (x+1)/2
xvals = np.linspace(lb,ub,10)
for xi in xvals:
	print(" Check x={}: F_X ={}; (x+1)/2={} ".format(xi, X.cdf(xi), (xi+1)/2 ) )


# PDF for Y=X^2
# Y=X^2
# X = sqrt(Y)
# dx/dy = 0.5 y ^{-0.5}
# f_Y = 0.5 y ^{-0.5} f_X

y=x**2
# sanity check for F_Y = (sqrt(y)+1)/2
yvals = np.linspace(0,1,5)
for yi in yvals:
	print(" Check y={}: (sqrt(y)+1)/2={} ".format(yi,  (np.sqrt(yi)+1)/2 ) )

