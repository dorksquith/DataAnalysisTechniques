import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # general imports and settings
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 16}
    rc('font', **font)
    rc('text', usetex=True)
    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    ##Part 1: Using $\textcolor{blue}{\texttt{sympy}}$ for calculus##
    """)
    return


@app.cell
def _():
    #pip install sympy
    from sympy import init_printing
    init_printing()
    from sympy import symbols, diff
    from sympy import integrate,Symbol
    return diff, integrate, symbols


@app.cell
def _(diff, integrate, symbols):
    def poly_ex():
        print("Polynomial Example")
        print("------------------")
        x,y,z = symbols('x y z')
        def f(x,y,z):
        	return 2*x**2 + 3*y**3 + 4*z**4 +x*y*z

        print(r"f = 2x^2 + 3y^3 + 4z^4 + xyz")

        # partial derivative df/dx
        print("df/dx: ", diff(f(x,y,z), x))
        # partial derivative df/dy
        print("df/dy: ", diff(f(x,y,z), y))
        # second order partial derivative 
        print("d2f/dx2: ", diff(f(x,y,z), x,2))

        print("Definite int_0^1 f dx: ", integrate( f(x,y,z), (x,0,1) ))
        print("Indefinite int f dx: ", integrate( f(x,y,z), x ))

    poly_ex()
    return


@app.cell
def _(diff, integrate, symbols):
    def log_ex():
        print("Natural Log Example")
        print("-------------------")
        from sympy import log
        print(r"f = log(x/y)")
        # function with natural log
        x,y = symbols('x y')
        def fL(x,y):
        	return log( x/y ) 
        print("df/dx: ", diff(fL(x,y), x))
        print("df/dy: ", diff(fL(x,y), y))

        print("int f dx: ", integrate( fL(x,y), x ))
        print("int f dy: ", integrate( fL(x,y), y ))

    log_ex()
    return


@app.cell
def _(diff, integrate, symbols):
    def exp_ex():
        print("Exponential Example")
        print("-------------------")
        from sympy import exp
        print(r"f = exp(2x)")
        # function with exp
        x = symbols('x')
        def fE(x):
        	return exp( 2*x)
        print("df/dx: ",diff(fE(x), x))
        print("int f dx: ", integrate( fE(x), x ))

    exp_ex()
    return


@app.cell
def _(diff, symbols):
    def binom_loglike_ex():
        print("Binomial Log Likelihood Example")
        print("-------------------------------")
        from sympy import log, factorial
        print(r"f= log p_K, K~Binom(n,p)")
        n,k,p = symbols('n k p')
        def fMLE(n,k,p):
        	return log( factorial(n)/(factorial(k)*factorial(n-k)) ) + k*log(p) + (n-k)*log(1-p)

        # first derivative  
        print("Score (first derivative of log likelihood): ", diff(fMLE(n,k,p), p))

        # second derivative
        print("Information (second derivative of log likelihood): ", diff(fMLE(n,k,p), p, 2))

    binom_loglike_ex()
    return


@app.cell
def _(integrate, symbols):
    def expon_EV_ex():
        print("Exponential Example: expected value and variance")
        print("------------------------------------------------")
        print(r"E[T] = int_0^{infty} t r exp(-rt) dt")
        from sympy import exp, oo # infinity
        # integrate exponential PDF
        r,t = symbols('r t')
        ET = integrate(  t*r*exp(-r*t), (t, 0, oo) ).args[0][0]
        ET2 = integrate(  t**2*r*exp(-r*t), (t, 0, oo) ).args[0][0]
        print( "E[T]: " , integrate(  t*r*exp(-r*t), (t, 0, oo) ).args[0][0] )
        print( "V[T]: " , ET2-ET**2)

    expon_EV_ex()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ##Part 2: Using $\textcolor{blue}{\texttt{jax}}$ for matrix multiplication##
    """)
    return


@app.cell
def _():
    # pip install jax
    from jax import jacfwd 
    import jax.numpy as jnp
    return jacfwd, jnp


@app.cell
def _(jacfwd, jnp):
    from scipy.stats import uniform
    N=1000
    x = uniform(-20,40).rvs(size=N)
    y = uniform(-20,40).rvs(size=N)
    z = uniform(-20,40).rvs(size=N)

    def jax_example_linear_func(print_cov):
        print("Linear Function example: f = x + 2y + 3z ")
        print("-----------------------------------------")

        # Coefficients of x,y,z
        A = jnp.array([1,2,3])
        # Uniform RVs in range -20,20
        X=jnp.array([x,y,z])   
        # Covariance matrix for our three RVs
        C = jnp.cov(X)
        if print_cov:
            print("Covariance Matrix:")
            print(C)    
        # means
        MU = jnp.mean(X)

        # Linear function of three RVs
        # f = x + 2y + 3z
        def f(X,A):
        	return A*X

        # jacfwd sets up partial derivatives for the function defined above
        Jf = jacfwd(f)
        # evaluate Jacobian at X=MU (all RVs equal to their mean)
        J = Jf( MU,A)
        # Transpose of Jacobian (switch rows and columns)
        JT = J.T
        # Variance V[f] = [J][C][J.T] matrix multiplication
        V = J.dot(C).dot(JT)
        print("V[f]: ", V)

    jax_example_linear_func(print_cov=True)
    return x, y


@app.cell
def _(jacfwd, jnp, x, y):
    def jax_example_nonlinear_func(print_cov):
        print("Nonlinear Function example: f = 2x^2 + 3xy ")
        # f = x(2x+3y) = f1*f2
        X=jnp.array([x,y])
        C = jnp.cov(X)
        if print_cov:
            print("Covariance Matrix:")
            print(C) 
        MU = jnp.mean(X)
        A=jnp.array([2,3])
        def f(X,A):
        	return (A*X)*X
        Jf = jacfwd(f)
        J = Jf( MU,A)
        JT = J.T
        V = J.dot(C).dot(JT)
        print("sigma[f]: ", jnp.sqrt(V))

    jax_example_nonlinear_func(True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ##Part 3. Intro to MLE with $\textcolor{blue}{\texttt{optimize}}$, $\textcolor{blue}{\texttt{fsolve}}$ and $\textcolor{blue}{\texttt{fit}}$##
    """)
    return


@app.cell
def _(np):
    from scipy.stats import binom,fit
    import scipy.optimize as opt
    from scipy.optimize import approx_fprime

    # simple example from lecture

    def binomial_p_est(n=50,k=27):
        print("Binomial n={},k={}: Estimate p from the data".format(n,k))
        p=k/n
        print("true  p={}".format(p))
        # fake data
        data = binom.rvs(n, p, size=1000)

        # well-hidden nnlf method for negative log likelihood 
        # https://github.com/scipy/scipy/blob/v1.16.2/scipy/stats/_distn_infrastructure.py#L1544-L1558
        def binom_nnl(p, n, data):
            return binom.nnlf( (n, p, 0), data )
        
        # leaving out the constant of integration
        def pmf(p,n,data):
            return  p**(data) * (1-p)**(n-data) 

        def norm_const(p,n,data):
            return np.sum(pmf(p,n,data))
        # normalisation
        #from scipy.integrate import simpson 
        #def norm_const(p,n,data):
        #	return simpson( approx_binom_pmf(p,n,data), data)

    
        c=norm_const(p,n,data)
        print(f"norm const: {c}")
                
        def nll(p, n, data):
            # log( pdf/c ) = log(pdf) - log(c)
            c=norm_const(p,n,data)
            log_pmf = np.log( pmf(p,n,data) ) - np.log(c)      
            sum_log_pmf = np.sum( log_pmf )
            return -1*sum_log_pmf
      
        def score(p, n, data):
            tol=1e-9
            sc = approx_fprime( p, lambda p: nll(p, n, data), tol )  
            return sc
        
        p_guess=0.1

        # I reckon you could spend the best part of a year getting familiar with minimize.
        # SLSQP is "Sequential Least Squares" and takes constraints 
        # https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_minimize.py#L54-L828
        resultC = opt.minimize(binom_nnl, p_guess, args=(n, data), bounds=[(0,1)], method='SLSQP')
        print("minimize (SLSQP) estimates  p={}".format(resultC.x))

        # Nelder Mead is "Simplex"
        resultD = opt.minimize(binom_nnl, p_guess, args=(n, data), bounds=[(0,1)], method='Nelder-Mead' )
        print("minimize (Nelder-Mead) estimates  p={}".format(resultD.x))

        # fsolve is wrapper for minpack 
        # https://github.com/scipy/scipy/blob/v1.16.2/scipy/optimize/_minpack_py.py#L46-L191
        result_fsolve = opt.fsolve(binom_nnl, p_guess, args=(n, data),full_output=1 )
    
        print("fsolve estimates p: {}".format(result_fsolve[0])) 

        # fix n parameter
        bounds = {'n': (n,n)}
        res = fit(binom, data, bounds)
        print("fit estimates p: ",res.params[1])

    binomial_p_est()    

    return (opt,)


@app.cell
def _(np, opt):
    # Expon: waiting for a taxi
    def expon_rate_est():
        print("Exponential t = [15,18,11]: Estimate rate from the data ")
        from scipy.stats import expon
        data = np.array([15,18,11])
        loc = 0
        # we saw in lecture that MLE for rate = 3/44
        time_guess=44/3
        # larger fake data set if we want
        # data = expon.rvs(loc, time_guess, size=1000)
        print("true  rate={} (t = {}) ".format(1/time_guess,time_guess ))

        def expon_nnl(loc, scale, data):
            return expon.nnlf( (loc, scale, 0), data )

        # for continuous rvs we can use fit method directly
        # default method for fit is MLE
        fitloc,fitscale = expon.fit(data,floc=0)
        print("fit estimates t: ",fitscale)

        # fsolve
        result_fsolve = opt.fsolve(expon_nnl, time_guess, args=(loc, data),full_output=1 )
        print("fsolve estimates t: {}".format(result_fsolve[0])) 

        # minimize doesn't like this example - to figure out 
        #resultC = opt.minimize(expon_nnl, time_guess, args=(loc, data), bounds=[(0,100)], method='SLSQP')
        #print("minimize (SLSQP) estimates t: {}".format(resultC.x))
        #resultE = opt.minimize(expon_nnl, time_guess, args=(loc,data), method='Nelder-Mead')
        #print("minimize (Simplex Nelder-Mead) estimates waiting time={}".format(resultE.x))
    expon_rate_est()
    return


if __name__ == "__main__":
    app.run()
