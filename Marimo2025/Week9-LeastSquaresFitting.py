import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.optimize import least_squares, curve_fit
    from numpy.random import default_rng
    import matplotlib.pyplot as plt
    from matplotlib import rc
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 16}
    rc('font', **font)
    rc('text', usetex=True)
    return default_rng, least_squares, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ##1.1 Least Squares Parameters and Uncertainites with $\textcolor{blue}{\texttt{scipy.optimize.least\_squares}}$##

    Let's setup the common parts here.
    """)
    return


@app.cell
def _(default_rng, np):
    # Set up the parameters
    m = -3
    c = 2
    N_params = 2

    # range of x
    x_min = 0
    x_max = 10
    n_points = 10

    # values of control variable x to make fake data points
    x_vals = np.linspace(x_min, x_max, n_points)

    # values of control variable x to use in fit
    x_test = np.linspace(x_min, x_max, n_points*10)

    noise=0.1


    rng = default_rng()

    # generate y-values from y=mx+c and y-errors from standard normal distribution
    def generate_y(x, m, c,  noise=0.1, seed=123):
        rng = default_rng(seed)
        y = m * x + c
        e_y = noise * rng.standard_normal(x.size) 
        return y+e_y, e_y 

    y_vals, e_vals = generate_y(x_vals, m, c, noise=noise)

    # Degrees of freedom = Number of measurements - Number of parameters - Number of constraints
    N_degrees_freedom = len(y_vals) - N_params

    # residuals: mx + c - y
    def residuals(p, x, y):
        return  p[1] * x + p[0]  - y 

    # initial estimate for parameter values p0 = [c0,m0]
    p0 = np.array([1.0,1.0])

    return N_degrees_freedom, generate_y, p0, residuals, x_test, x_vals, y_vals


@app.cell
def _(mo):
    mo.md(r"""
    **This method will perform a Least Squares fit to a straight line $y=mx+c$ assuming homoscedastic uncertainties on the measurements $y_i$.**
    """)
    return


@app.cell
def _(
    N_degrees_freedom,
    generate_y,
    least_squares,
    np,
    p0,
    residuals,
    x_test,
    x_vals,
    y_vals,
):
    def scipy_straightline(print_full_result=False):
  
        res_ls = least_squares( residuals, p0, args=(x_vals, y_vals) )
    
        if(print_full_result):
            print(res_ls)

        y_ls = generate_y(x_test, *res_ls.x)

        # now the uncertainties 
        # .fun returns the residuals
        sumsq_res = np.sum(res_ls.fun**2)

        # The expected value of chi^2 is E[chi^2] = nu : the number of degrees of freedom
        # E[chi^2] = E [sumsq_res / sigma^2] = N_degrees_freedom
        # so E[ sigma^2 ] = E [sumsq_res]/N_degrees_freedom
        reduced_chisq = sumsq_res/N_degrees_freedom
    
        # Jacobian
        # Why Jacobian? Because Hessian (second derivs) is hard for non-humans.
        # H = J^T.J + sum (r grad^2 r) : this is approx J^T.J because the second term (second deriv) is small at the minimum
        # see slide 25 of MaxLike lecture:
        # First derivative provides the score: Jacobian = s
        # Second derivative provides the information: Hessian
        # We showed that E[ds/dtheta] = E[s^2] -> I = s^2 => H = J^2
        J = res_ls.jac
        # squaring a matrix means multiplying it by its transpose
        H = J.T.dot(J)
        # inverse of this gives us the covariance matrix (but we aren't quite there!)
        Hinv = np.linalg.inv(H)
        # The chi^2 denominator (sum sigma^2) is left out of the computations for efficiency
        # scaling by reduced_chisq = scaling by our missing sigma^2 
        # Covariance matrix is inverse of Hessian times reduced_chisq. 
        cov = Hinv * reduced_chisq

        # Diagonal elements of Covariance Matrix are the parameter variances sigma_theta^2
        var = np.diagonal(cov)

        # parameter uncertainties: sqrt of parameter variances
        sigma = var**0.5
        # off-diagonal terms are covariance
        covmc = cov[0][1]
        # linear correlation coefficient tells us how strongly linearly correlated m and c are
        rhomc = covmc / ( sigma[0] * sigma[1] )

        # =============
        # IMPORTANT!
        # I have truncated the precision with which the results are printed using  
        # eg {:.2e} which means "print as scientific notation with 2 numbers after decimal place"
        # Before doing this, I looked at the full precision, which you should always do.
        # ===============
        print("Scipy's least_squares parameter estimation:")
        print("* Intercept c: {:.2e},  sigma_c:{:.2e}".format( res_ls.x[0], sigma[0] ) )
        print("* Slope m: {:.2e},  sigma_m:{:.2e}".format( res_ls.x[1], sigma[1] ) )

        print("* Covariance(c,m): {:.2e},  linear correlation rho(c,m):{:.2e}".format( covmc, rhomc ) )

        # performance - how close to the true parameter values are we?
        # MSE = bias^2 + variance

        # Bias: b = E[estimate] - TrueValue = E[estimate-TrueValue] = E[residuals]
        # The bias for least squares is zero by construction
        # because the average of the residuals is always zero (within some tolerance) in least squares
        # print("Average of residuals: ", np.mean(res_ls.fun ))

        # Variance: E[(estimate-TrueValue)^2] = E[residuals^2] 
        # print("Average squared residual: ", np.mean(sumsq_res ))
        print("* Estimator MSE: {:.2e}".format(np.mean(sumsq_res )))
    
        
    scipy_straightline()

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.2 Checking the results with $\textcolor{blue}{our\; own\; calculations}$ ##

    The example above assumes all of the uncertainties $\sigma_i$ on the measurements $y_i$ are the same: the uncertainties on the data are **homoscedastic**. We will make the same assumption in the by-hand check here.
    """)
    return


@app.cell
def _(N_degrees_freedom, np, residuals, x_vals, y_vals):
    def byhand_straightline():
        # see LeastSquares.pdf slide 23
        N = len(y_vals)
        s1 = N
        sxx = np.sum( x_vals**2 )
        sx = np.sum( x_vals )
        sy = np.sum( y_vals )
        sxy = np.sum( x_vals*y_vals )
    
        delta = N*sxx - sx**2 
    
        m = ((sxy*N)-(sx*sy))/delta
        c = ((sxx*sy)-(sxy*sx))/delta

        p =(c,m)
        resids = residuals(p, x_vals, y_vals)    
        sumsq_res = np.sum( (resids)**2 )
    
        sigma_y = (sumsq_res/N_degrees_freedom)**0.5

        sigma_m = sigma_y * np.sqrt( N/delta )
        sigma_c = sigma_y * np.sqrt( sxx/delta )
        covmc = sigma_y**2 * (-sx/delta)
        rhomc = covmc / (sigma_c * sigma_m)

        print("By-hand least_squares parameter estimation:")
        print("* Intercept c: {:.2e},  sigma_c:{:.2e}".format( c, sigma_c) )
        print("* Slope m: {:.2e},  sigma_m:{:.2e}".format( m, sigma_m) )

        print("* Covariance(c,m): {:.2e},  linear correlation rho(c,m):{:.2e}".format( covmc, rhomc ) )

    byhand_straightline()
    
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.3 Dealing with outliers: $\textcolor{blue}{\texttt{soft\_l1}}$ ##

    Thanks to the squared residual term in the numerator of $\chi^2=\sum\limits_i^N\dfrac{ (y_i - \mu)^2}{\sigma_i}$, measurements $y_i$ that deviate significantly from the expectation (for example due to human error in making or recording the data) can really mess up the fit.

    If we have this problem, we must use **Robust Least Squares**. We can manage outliers and altering the numerator such that is less sensitive to large deviations. One approach to this is to use the L1 Loss function:
    $(y_i - \mu)^2 \rightarrow |y_i - \mu|$. Scipy's least_squares has a few variations on this. We will use the $\texttt{soft\_l1}$ loss in the examples below.
    """)
    return


@app.cell
def _(default_rng, least_squares, np, plt):
    def scipy_quadratic_outliers(print_full_result=False):
        # function is linear in parameters
        # y = ax + bx^2
    
        # parameters
        a = 0.5
        b = 2.0
    
        # range of x
        x_min = -10
        x_max = 10
        n_points = 15
        x_vals = np.linspace(x_min, x_max, n_points)
        x_test = np.linspace(x_min, x_max, n_points*10)
    
        noise=0.1    

        n_outliers = 2
    
        fig, ax1 = plt.subplots(1,1)
    
        rng = default_rng()
        #print(rng)

        def generate_y(x, a, b, noise=0.01, n_outliers=0, seed=123):
        	rng = default_rng(seed)    
        	y = a * x + b * x**2        
        	y_uncertainties = noise * rng.standard_normal(x.size) 
        	outliers = rng.integers(0, x.size, n_outliers) 
        	y_uncertainties[outliers] *= 10000
    
        	return y + y_uncertainties 
    
    
        y_vals = generate_y(x_vals, a, b, noise=0.01, n_outliers=n_outliers)
    
        def residuals(p, x, y):
        	return  p[0] * x + p[1] * x**2 - y 
    
        p0 = np.array([1.0,1.0])

        # using the default loss function (y_i-mu)^2
        res_ls = least_squares( residuals, p0, args=(x_vals, y_vals)  )

        y_ls = generate_y(x_test, *res_ls.x)
    
        # using soft l1  and f_scale
        res_ls2 = least_squares( residuals, p0, args=(x_vals, y_vals), loss='soft_l1',f_scale=noise )
        y_ls2 = generate_y(x_test, *res_ls2.x)

        if(print_full_result):
            print("Default: \n", res_ls)
            print("Soft L1: \n", res_ls2)
        
        # plot data points
        ax1.plot(x_vals, y_vals, color='black',marker='o',linestyle='',label='data')

        # plots default least_squares fit
        ax1.plot(x_test, y_ls,  color='r',label='LS' )

        # plots robust soft_l1 least_squares fit
        ax1.plot(x_test, y_ls2, color='b',label='soft_l1' )
    
    
        ax1.set_xlabel("x",fontsize=20)
        ax1.set_ylabel("y")
        #ax1.set_ylim(-5,5)
        plt.legend()

        plt.show()
    scipy_quadratic_outliers(False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ##1.4 When a function is $\textcolor{blue}{nonlinear}$ in its parameters##

    The soft_l1 method above is not just useful for outliers, it is a very good choice when we have a function that is nonlinear in its parameters.
    """)
    return


@app.cell
def _(default_rng, least_squares, np, plt):
    def scipy_nonlinear(print_full_result=False):
        # example when a function is nonlinear in parameters
        # y = 1 / ( a + b * np.exp(c*x) )
    
        # parameters
        a = 0.5
        b = 2.0
        c = -0.5
    
        # range of x
        x_min = 0
        x_max = 10
        n_points = 15
        x_vals = np.linspace(x_min, x_max, n_points)
        x_test = np.linspace(x_min, x_max, n_points*10)
    
        noise=0.1
        n_outliers = 0
    
        fig, ax1 = plt.subplots(1,1)
    
        rng = default_rng()
        print(rng)

        def generate_y(x, a, b, c,  noise=0.01, n_outliers=0, seed=123):
        	rng = default_rng(seed)    
        	y =  a + b * np.exp(c * x)     
        	y_uncertainties = noise * rng.standard_normal(x.size)     
        	outliers = rng.integers(0, x.size, n_outliers) 
        	y_uncertainties[outliers] *= 100    
        	return y + y_uncertainties 
    
    
        y_vals = generate_y(x_vals, a, b, c, noise=0.01, n_outliers=n_outliers)
    
        def residuals(p, x, y):
        	return  p[0] + p[1] * np.exp(p[2]*x)  - y 

        p0 = np.array([1.0,1.0,1.0])
    
        res_ls = least_squares( residuals, p0, args=(x_vals, y_vals)  )
        y_ls = generate_y(x_test, *res_ls.x)
    
        res_ls2 = least_squares( residuals, p0, args=(x_vals, y_vals), loss='soft_l1',f_scale=noise )

        y_ls2 = generate_y(x_test, *res_ls2.x)

        if(print_full_result):
            print("Default: \n", res_ls)
            print("Soft L1: \n", res_ls2)
        
        # plot data points
        ax1.plot(x_vals, y_vals, color='black',marker='o',linestyle='',label='data')

        # plots default least_squares fit
        ax1.plot(x_test, y_ls,  color='r',label='LS' )

        # plots robust soft_l1 least_squares fit
        ax1.plot(x_test, y_ls2, color='b',label='soft_l1' )
       
        ax1.set_xlabel("x",fontsize=20)
        ax1.set_ylabel("y")

        plt.legend()

        plt.show()

    scipy_nonlinear(False)
    return


if __name__ == "__main__":
    app.run()
