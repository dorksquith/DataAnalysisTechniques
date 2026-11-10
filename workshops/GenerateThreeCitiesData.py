import numpy as np
np.random.default_rng()
from scipy.stats import norm

def three_cities_data( reproducible=True ):

    if reproducible:
        np.random.seed(42)

    #--------------------------------------------------------------------------------------------#
    # Parameters for Normal distribution in theoretical universe with infinite data points
    #--------------------------------------------------------------------------------------------#
    
    mu    = 7.33
    sigma = 1.21
 
    dist = norm(loc=mu, scale=sigma)

    # Number of fake people asked
    sample_sizes = [2, 3, 10, 30, 100, 3_000]

    iterations = range( len(sample_sizes) )

    # Difference between subsequent sample sizes eg 3-2=1, 10-3=7
    size_inc     = [ j-i for i, j in zip( sample_sizes[:-1], sample_sizes[1:] ) ]
    
    # want first number to be 2 for our N=2 sample
    size_inc.insert(0, 2)

    #-----------------------#
    # Fake data for Brixton
    #-----------------------#
    B_data1 = dist.rvs(size = max(10_000, max(sample_sizes) ) )  
    B_data = []

    #-----------------------#
    # Fake data for Stockport
    #-----------------------#
    S_data1 = dist.rvs(size = max(10_000, max(sample_sizes) ) ) 
    S_data = []

    #-----------------------#
    # Fake data for Glasgow
    #-----------------------# 
    G_data1 = dist.rvs(size = max(10_000, max(sample_sizes) ) )
    G_data = []

    #-----------------------#
    # Loop over sample sizes
    #-----------------------#   
    for i in iterations:
       
        N = sample_sizes[i]
        Nplus = size_inc[i]

        # randomly draw N **additional** fake measurements

        for j in range(Nplus):
        
            B_data.append(np.random.choice(B_data1,  replace=False) )
            S_data.append(np.random.choice(S_data1,  replace=False) )
            G_data.append(np.random.choice(G_data1,  replace=False) )

    return mu, sigma, B_data, S_data, G_data

reproducible = True
true_mu, true_sigma, B_data, S_data, G_data = three_cities_data(reproducible)


