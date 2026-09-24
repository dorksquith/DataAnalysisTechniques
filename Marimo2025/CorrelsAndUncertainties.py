import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    # GENERAL IMPORTS
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import expon, norm, uniform, beta, poisson
    return beta, expon, norm, np, plt, uniform


@app.cell
def _(np):
    def variance_of_dataset():
        k=[5.10, 4.95, 4.75, 5.15, 4.90]
        mean_k = np.mean(k)
        var_k = np.var(k)
        sigma_k = np.std(k)
        print(r"Mean[k]: {:.2f}, V[k]:{:.2f}, sigma_k:{:.2f}".format(mean_k,var_k,sigma_k))

    variance_of_dataset()
    return


@app.cell
def _(beta, expon, norm, np, plt, uniform):
    def plots_showing_sigma():    
        fig, axs = plt.subplots(2, 2)
        ax=axs.flatten()    

        def get_dist(i):
        	if i==0:
        		return [uniform(0,1), "X~U(0,1)"]
        	if i==1:
        		return [expon(0.5), "X~Expon(2)" ] # math def uses Expon(rate) but python uses expon(1/rate)
        	if i==2:
        		return [beta(2,1), "X~Beta(2,1)" ]
        	if i==3:
        		return [norm(0,1), "X~N(0,1)" ]
        	else:
        		"out of range!"
        		return [0,"NULL"]

        for i in range(4):
            ax1=ax[i]    
            X,title= get_dist(i)
            mean = X.expect()
            sig = X.std()
            xvals = np.linspace(min(mean-5*sig,0), max(mean+5*sig,1),1_000)    
            pdf=X.pdf(xvals)    
            ax1.plot(xvals, pdf, '-',linewidth=1)    
            ax1.fill_between(xvals, pdf, color='lightsteelblue')    
            ax1.set_ylabel('PDF')
            ax1.set_xlabel('X')
            ax1.set_ylim(bottom=0)    
            ax1.axvline(x=mean, color="darkgreen", linestyle="--",linewidth=2)
            ax1.axvspan(mean-sig, mean+sig, alpha=0.3, color='yellow')
            ax1.set_title(title)

        return plt.show()

    plots_showing_sigma()    
    return


@app.cell
def _(np):
    # BINOMIAL PMF
    def varxy():
        from scipy.stats import uniform

        Xdist=uniform(-10,20)
        x = Xdist.rvs(size=100)
        y = x**2

        print("numpy var(x+y) = ",np.var(x+y))
        eX = x - np.mean(x)
        eY = y - np.mean(y)
        print("numpy  mean([eX+ey]^2) = ", np.mean( (eX+eY)**2 ) )


        print( "numpy var(x) = ", np.var(x) )
        print( "numpy var(y) = ", np.var(y) )
        print( "numpy mean(eXeY) = ", np.mean( (eX*eY) ) )

        print("\n numpy cov(x,y)", np.cov(x,y) )
        # notice that np.cov prints a matrix with elements v[x], cov[x,y] \\ v[y], cov[x,y] \\

        # notice that cov(x,y) is different from our calculations above!
        # this is because np.cov uses the UNBIASED variance, whereas np.var is biased

        print( "\nnumpy var(x) , with scale factor N/N-1 = ", (100/99)*np.var(x) )
        print( "numpy var(y) , with scale factor N/N-1 = ", (100/99)*np.var(y) )
        print( "numpy mean(eXeY), with scale factor N/N-1 on eXeY = ", np.mean( (100/99)*(eX*eY) ) )



    varxy()
    return


@app.cell
def _(np):
    # playing around with V[ax] = a^2 V[x]
    def var_ax(a):
        from scipy.stats import uniform
        Xdist=uniform(-10,20)
        x = Xdist.rvs(size=10_000) # bias on variance will be very small now with large N=10,000
        ax = a*x
        var_x = np.var(x)
        var_ax = np.var(ax)

        print("Check it out, V[x] = {:.2f}, V[{}x] = {:.2f}, which is {}*{} V[x]".format(var_x, a, var_ax, a,a) )

    var_ax(a=1)
    var_ax(a=4)
    var_ax(a=10)
    return


@app.cell
def _(np, uniform):
    def cov_algebra():

        Xdist=uniform(-10,20)
        x = Xdist.rvs(size=10_000)

        print("cov[X,X] = \n", np.cov(x,x) )
        print("\nV[X] from np.cov = {:.4f}".format( np.cov(x,x)[0,1]) )
        print("\n2*V[X] from np.cov = {:.4f}".format( 2*np.cov(x,x)[0,1]) )

        print("\ncov[X,2X]= \n", np.cov(x,2*x) )

        print("\ncov[X,2X+4]= \n", np.cov(x,(2*x+4)) )



    cov_algebra()
    return


@app.cell
def _(np):
    def correls():
        from scipy.stats import uniform

        Xdist=uniform(-1,2)
        x = Xdist.rvs(size=10_000)

        fx =x**2 
        fx2 =2*x+1

        Ydist=uniform(-1,2)
        y = Ydist.rvs(size=10_000)  # independent of x 

        print("Correlation matrix rho(x,y):\n",np.corrcoef(x,y) )
        print("Correlation coefficient rho(x,y):",np.corrcoef(x,y)[0,1] )


        print("\nCorrelation matrix rho(x,fx) where fx= x^2 (nonlinear function of x):\n",np.corrcoef(x,fx) )
        print("Correlation coefficient rho(x,fx):",np.corrcoef(x,fx)[0,1] )

        print("\nCorrelation matrix rho(x, fx) where fx= 2x+1 (linear function of x):\n",np.corrcoef(x,fx2) )
        print("Correlation coefficient rho(x,fx):",np.corrcoef(x,fx2)[0,1] )

    correls()
    return


@app.cell
def _(np, uniform):
    # a linear function is f(x) = ax+b
    def example_linearfunc(a,b):
        N=10_000
        print("Using N={}, so the mean is not the expected value and formulae will only give approximate agreement.".format(N))

        Xdist=uniform(-10,20)
        x = Xdist.rvs(size=N)
        mean_x = np.mean(x)
        var_x = np.var(x)

        f = lambda x, a, b : a*x + b

        var_f = np.var( f(x, a,b) )
        mean_f = np.mean( f(x, a,b) )

        # these are not identical, but if we had N=infinity the mean is the expected value and they will be identical.
        print("\nLecture slides claim E[ax+b] = aE[x]+b...")
        print("mean[ax+b]: ",mean_f)
        print("a*mean[x]+b: ",(a*mean_x +b) )

        print("\nLecture slides claim V[ax+b] = a^2 V[x]...")
        print("V[ax+b]: ",var_f)
        print("a^2*V[x]: ",(a**2 * var_x))

    example_linearfunc(a=2,b=2)      
    return


@app.cell
def _(np, uniform):
    # a linear function is f(x,y) = ax+by
    def example_linearfxy(a,b):
        N=10_000
        print("Using N={}, so the mean is not the expected value and formulae will only give approximate agreement.".format(N))

        Xdist=uniform(-10,20)
        x = Xdist.rvs(size=N)
        mean_x = np.mean(x)
        var_x = np.var(x)

        Ydist=uniform(-10,20)
        y = Ydist.rvs(size=N)
        mean_y = np.mean(y)
        var_y = np.var(y)

        cov_xy = np.cov(x,y)[0,1]

        f = lambda x,y,a, b : a*x + b*y

        var_f = np.var( f(x,y,a,b) )
        mean_f = np.mean( f(x,y,a,b) )

        # these are not identical, but if we had N=infinity the mean is the expected value and they will be identical.
        print("\nLecture slides claim E[ax+by] = aE[x]+bE[y]...")
        print("mean[ax+by]: ",mean_f)
        print("a*mean[x]+b*mean[y]: ",(a*mean_x +b*mean_y) )

        print("\nLecture slides claim V[ax+by] = a^2V[x] +b^2V[x] +ab cov(x,y)")
        print("V[ax+by]: ",var_f)
        print("a^2*V[x] +b^2*V[y]: ",(a**2 * var_x + b**2 * var_y ))
        print("a^2*V[x] +b^2*V[y] + 2ab cov(x,y): ", (a**2 * var_x + b**2 * var_y + 2*a*b*cov_xy ))

        # the RVs x and y have a tiny bit of covariance despite being randomly pulled from the uniform distribution
        # checking the standardised covariance aka correlation coefficient gives us a clearer idea of how small this is:

        correl_xy = np.corrcoef(x,y)[0,1]
        print("My (pseudo)random variables X and Y have a linear correlation of ", correl_xy)

        # see what happens when you increase/decrease N


    example_linearfxy(a=3,b=1)  
    return


@app.cell
def _(np):
    # Variance of a linear (or nonlinear!) function with Jacobian
    def jacobian_example():
        from jax import jacfwd
        import jax.numpy as jnp
        from scipy.stats import uniform
    
        # f = ax + by + cz = A*X
        def f(X,A):
            return A*X
        
        # coefficients
        a,b,c =1,2,3
        A = np.array([a,b,c])
    
        # RVs
        N=10_000
        x1 = uniform(-20,40).rvs(size=N)
        x2 = uniform(-20,40).rvs(size=N)
        x3 = uniform(-20,40).rvs(size=N)
        X=np.array([x1,x2,x3])
    
        # Means
        MU = np.mean(X)
    
        # Covariance matrix for our three RVs
        C = np.cov(X)
    
        # jacfwd sets up partial derivatives for the function defined above
        Jf = jacfwd(f)
    
        # evaluate Jacobian at X=MU (all RVs equal to their mean)
        J = Jf(MU,A)
    
        # Transpose of Jacobian (switch rows and columns)
        JT = J.T
    
        # Variance V[f] = [J][C][J.T] matrix multiplication
        V = J.dot(C).dot(JT)
        print("V[f]: ", V)
    
        # Cross-check: Variance V[f] = sum_i sum_j ai aj cov(xi, xj)
        # note that np.cov(x,y) returns a 2x2 matrix [[ vx cov(x,y)] [cov(y,x) vy] ]
        # - we want off-diagonal element [0,1]  (or [1,0] equivalently)
        def check_v(X,A):
            Vcc = 0
            for ai,xi in zip(A,X):
                for aj,xj in zip(A,X):   
                    Vcc += ai*aj*np.cov(xi,xj)[0,1] 
            return Vcc
        alt_V=check_v(X,A)
        print("Cross-check alt_V[f]: ",alt_V)

        # I am not a huge fan of the numpy isclose because I think its easy to mess up
        # https://numpy.org/devdocs/reference/generated/numpy.isclose.html    
        # set tolerance to 1 part in a billion
        delta = 1e-9
        howclose = (V-alt_V)/V
        print("Check passed? {} (with % diff: {} and delta: {})".format( howclose<=delta, howclose, delta ) )

    
        # nonlinear function? no problem h = ax^2 + by^3 + cz^4 
        def h(XM,A):
            return A*XM
        XM=np.array([x1**2,x2**3,x3**4])
        MUM = np.mean(XM)
        CM = np.cov(XM)
        Jh = jacfwd(h)
        JM = Jh(MUM,A)
        JMT = JM.T
    
        VM = JM.dot(CM).dot(JMT)
        print("V[h]: ", VM)
        alt_VM=check_v(XM,A)
        print("Cross-check V[h]: ",alt_VM)

        howclose = (VM-alt_VM)/VM
        print("Check passed? {} (with % diff: {} and delta: {})".format( howclose<=delta, howclose, delta ) )
    
    jacobian_example()

    return


if __name__ == "__main__":
    app.run()
