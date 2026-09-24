import marimo

__generated_with = "0.19.0"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Part 1: Using $\textcolor{blue}{\texttt{sympy}}$ for calculus##
    """)
    return


@app.cell
def _():
    def base_example(n):
        import numpy as np
        b2 = np.base_repr(n,2)     # converts the decimal number n to binary (base=2)
        b10 = np.base_repr(n,10)   # as n is already in base 10, b10 will just be n
        bdef = np.base_repr(n)     # no base provided, so default (base=2) will be used
        b16 = np.base_repr(n,16)   # hexadecimal just for giggles

        print( f'decimal: {n} is binary: {b2}, hex: {b16}')

    base_example(123.45)
    return


@app.cell
def _():
    def logexp_example(n):
        import numpy as np
    
        print(f' e = {np.exp(1) }') # np.exp(1) = e^1 = e, where e is Euler's number ~2.781
    
        e_n = np.exp(n) 
        print(f' exp({n}) = e^({n}) = {e_n}')
    
        log_n = np.log(n) # log is natural (base e) by default
        log2_n = np.log2(n)
        log10_n = np.log10(n)

        print(f' log({n})) = {log_n}')
        print(f' log_2({n})) = {log2_n}')
        print(f' log_10({n})) = {log10_n}')
    
    
    logexp_example(2)
    
    return


@app.cell
def _():
    def sumprod_example(arr):
        import numpy as np

        print(f'numbers: {arr}')

        print(f'logs of numbers: {np.log(arr)}')
   
        sum = np.sum(arr)
        prod = np.prod(arr)
        sumlog = np.sum( np.log(arr) )
        logprod = np.log( np.prod(arr) )

        print(f'The sum of the logs is {sumlog}')
        print(f'The log of the product is {logprod}')

        # are they the same? If not, why not?
   
    sumprod_example([1,2,3,4])
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
        def f2(x):
            return 2*x**3 + 9
        print(f'y= {f2(x)}')

        d1 = diff( f2(x),x )
        print(f'dy/dx= {d1}')

        d2 = diff( f2(x),x,2 )
        print(f'd2y/dx2= {d2}')


        i1 = integrate( f2(x),x )
        print(f'integral y= {i1}')

        i2 = integrate( d1,x )
        print(f'integral dy/dx= {i2}')

        i3 = integrate( f2(x),(x,0,1) )
        print(f'definite integral y= {i3}')

        print("df/dx: ", diff(f(x,y,z), x))
        # partial derivative df/dy
        print("df/dy: ", diff(f(x,y,z), y))
        # second order partial derivative 
        print("d2f/dx2: ", diff(f(x,y,z), x,2))

        print("Indefinite int f dx:   ", integrate( f(x,y,z), x ))
        print("Definite int_0^1 f dx: ", integrate( f(x,y,z), (x,0,1) ))

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
        r = symbols('r', positive=True)
        t = symbols('t')
        ET = integrate(  t*r*exp(-r*t), (t, 0, oo) )
        ET2 = integrate(  t**2*r*exp(-r*t), (t, 0, oo) )
        variance = ET2 - ET**2
        print( "E[T]: " , ET)
        print( "V[T]: " , variance)

    expon_EV_ex()
    return


@app.cell(hide_code=True)
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
        X = jnp.array([x,y,z])   
        # Covariance matrix for our three RVs
        C = jnp.cov(X)

        if print_cov:
            print("Covariance Matrix:")
            print(C)    
        # means
        MU = jnp.mean(X, axis=1)

        # Linear function of three RVs
        # f = x + 2y + 3z
        def f(X,A):
        	return jnp.dot(A, X)

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
        MU = jnp.mean(X, axis=1)
        A=jnp.array([2,3])
        def f(X,A):
        	return A[0]*X[0]**2 + A[1]*X[0]*X[1]
        Jf = jacfwd(f)
        J = Jf( MU,A)
        JT = J.T
        V = J.dot(C).dot(JT)
        print("sigma[f]: ", jnp.sqrt(V))

    jax_example_nonlinear_func(True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Part 3. Intro to MLE with $\textcolor{blue}{\texttt{minimize}}$, $\textcolor{blue}{\texttt{fsolve}}$ and $\textcolor{blue}{\texttt{fit}}$##
    """)
    return


@app.cell
def _(np):
    from scipy.stats import binom, fit
    import scipy.optimize as opt

    # simple example from lecture
    def binomial_p_est(n=50,k=27):
        print("Binomial n={},k={}: Estimate p from the data".format(n,k))
        p=k/n
        print("true  p={}".format(p))
        # fake data
        data = binom.rvs(n, p, size=100)


        # Lecture week 4 SpecialPDFs.pdf
        # PMF = c * p**k * (1-p)**(n-k) for 0<= k <= n 
        # c is the binomial coefficient
        # c = n! / ( k! (n-k)! )
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.binom.html#scipy.special.binom 
        # c = GammaFunc(n+1) / ( GammaFunc(k+1) GammaFunc(n-k+1) )
        # these are equivalent, because for non-negative integers z GammaFunc(z) = (z-1)!
        # and for Binomial PMF the n, k, and (n-k) are all non-negative integers.
        from scipy.special import binom as bnc

        # Binomial Log PMF
        def binom_negloglike(p,n,data):
            k = data
            c = bnc(n,k) 
            #pmf = c * p**k * (1-p)**(n-k)
            if (c.any()<=0 or p<=0 or p==1): return np.inf

            log_pmf = np.log(c) + k * np.log(p) + (n-k) * np.log(1-p)
            sum_log_pmf = np.sum( log_pmf )
            if np.isnan(sum_log_pmf): return np.inf

            return -sum_log_pmf

        # Binomial Score: derivative of Log Likelihood wrt p
        def binom_score(p,n,data):
            if (p<=0 or p==1): return np.inf
            k = data
            # d/dp log(c) = 0
            # d/dp k*log(p) = k/p
            # d/dp (n-k)*log(1-p)  = d/dp f(g(p)) = df/dg dg/gp
            # outer function f(g) = (n-k)*log(g)
            # inner function g(p) = 1-p
            # d/dp (n-k)*log(1-p) = [(n-k)/(1-p)][-1]
            deriv_loglike = np.sum( k/p - (n-k)/(1-p) )
            if np.isnan(deriv_loglike): return np.inf
            return deriv_loglike


        p_guess=0.1

        # Find the parameters for which fun=0
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html#scipy.optimize.fsolve
        result_fsolve = opt.fsolve(binom_score, p_guess, args=(n, data),full_output=1 )
        print("fsolve estimates p: {}".format(result_fsolve[0])) 


        # Pass scipy.stats.binom distribution directly to scipy.stats.fit
        # fit uses MLE
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fit.html#scipy.stats.fit
        # bounds: for parameter we are not estimating. Fix parameter n (must be tuple).
        bounds = {'n': (n,n)}
        result_fit = fit(binom, data, bounds)
        print("fit estimates p: ",result_fit.params[1])



        # Minimization of scalar function of one or more variables.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        # Basic usage: minimize(fun, x0, args=() ) 
        # bounds needed to ensure 0<=p<=1
        # default is L-BFGS-B
        resultC1 = opt.minimize(binom_negloglike, p_guess, args=(n, data), bounds=[(0.,1.)], method='SLSQP')
        print("minimize (SLSQP) estimates  p={}".format(resultC1.x))
        #print(resultC1)



    binomial_p_est()    
    return fit, opt


@app.cell
def _(fit, np, opt):
    # Expon: waiting for a taxi
    def expon_rate_est():
        print("Exponential t = [15,18,11]: Estimate scale from the data ")
        from scipy.stats import expon
        data = np.array([15,18,11])
        loc = 0
        # Lecture week 8 MaxLike.pdf
        scale_true  = 44./3.
        scale_guess = 10.
        # larger fake data set if we want
        big_data = expon.rvs(loc=loc, scale=scale_true , size=1000)
        #data=big_data
        print(f'True  scale={scale_true}')

        # Negative Log Likelihood has a minimum at the MLE
        # for scale parameter
        def expon_nll(scale,data):
            x=data
            #pdf =  np.exp(-x/scale)/scale
            log_pdf = -np.log(scale) - x/scale  
            LL = np.sum( log_pdf )
            if np.isnan(LL):
                return -np.inf
            return -LL

        # alternatively, get log of pdf direct from the scipy expon distribution
        def direct_nll(scale, data):
            return -np.sum(expon.logpdf(data, loc=0, scale=scale))

        # to use fsolve we need a function that is zero for some choice of parameter
        # the (negative, log) likelihood has a minimum, but it is not at zero
        # so we take the derivative, which is zero at the MLE.
        # Derivative of Log Likelihood wrt scale
        def expon_score(scale,data):
            x=data
            # pdf =  1/scale * np.exp(-x/scale)
            # log pdf = -log(scale) - x/scale
            # derivative = -1/scale + x/scale**2
            # sum = -n/scale + sum(x)/scale**2
            dLL_ds = -len(x)/scale + np.sum(x)/scale**2
            if np.isnan(dLL_ds):
                return -np.inf
            return dLL_ds


        # for continuous rvs we can use fit method directly
        # default method for fit is MLE
        fitloc,fitscale = expon.fit(data,floc=0)
        print("direct expon.fit estimates scale: ", fitscale )

        # for continuous or discrete, scipy.stats.fit
        bounds = {'loc': (0,0),'scale': (min(data),max(data))} # bounds on parameters 
        result_fit = fit(expon, data, bounds)
        print("direct fit(expon, data, bounds) estimates scale: ",result_fit.params[1])

        # fsolve scale, using derivative of log likelihood (which is zero at MLE for scale)   
        result_fsolve = opt.fsolve( expon_score, scale_guess, args=(data) )
        print("fsolve (score) estimates scale: {}".format(result_fsolve[0])) 

        # minimize with negative log likelihood - using L-BFGS-B, but could also use Nelder-Mead, CG, see
        # https://scipy-lectures.org/advanced/mathematical_optimization/#knowing-your-problem
        result_m = opt.minimize(expon_nll, scale_guess , args=(data), method='L-BFGS-B')
        print("minimize (L-BFGS-B) estimates scale: {}".format(result_m.x[0]))


    expon_rate_est()
    return


if __name__ == "__main__":
    app.run()
