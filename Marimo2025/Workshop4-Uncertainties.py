import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###**Exercise 5.1 [Quick Pen & Paper Exercise]: Covariance/ Correlation of RVs**###

    I toss a coin three times. I define X as the number of heads, and Y as the number of tails.

    Use a pen an paper to:

    a) find the covariance, cov(X,Y).

    b) find the correlation coefficient, $\rho(X,Y)$


    ###**Exercise 5.2 [Quick Python Exercise]: Covariance/Correlation of RVs**###

    Measurements of two RVs are as follows:

    X1 = [18_841, 20_449, 20_987, 21_854, 22_778, 24_075, 25_253, 26_155, 27_227, 27_092,]

    X2 = [11, 16.3333, 23.75, 29.8333, 34.3333, 42.25, 47.5, 53.25, 57.0833, 58.1667,]

    a) Use python to calculate the mean, variance, and standard deviation of each RV

    b) Use python to find the covariance matrix for these two RVs

    c) What do the results indicate to you?


    ###**Exercise 5.3 [Python Plot & Covariance Calculation]: Correlation $\neq$ Independence**###

    The kinetic energy E of a sports cars is related to its speed v as $E\propto v^2$.

    a) Extract 100 values of speed from a Uniform(-160,160) distribution.

    b) Plot energy (y-axis) versus speed values (x-axis)

    c) Use numpy methods to find the covariance cov(E,v) and correlation coefficient $\rho(E,v)$

    d) What do the results indicate to you?


    ###**Exercise 5.4 [Python]: Variance of a Linear Function of Two RVs**###
    A function of two RVs is f(X,Y) = 3x+y.
    You have the following data measurements:

    x=[62, 69, 67, 63, 63, 69, 67]

    y=[12, 11, 10, 14, 12, 13, 13]

    a) Use python to express the mean and uncertainty of the function in the form $f(X,Y) = \overline{f} \pm \sigma_f$


    ###**Exercise 5.5 [Pen & Paper]: Two Approaches to finding the Variance of a Linear Function**###
    A function of three RVs is $f(X) = 2X_{(1)} + 3X_{(2)} + 4X_{(3)}$.

    Different methods for finding the variance of this function were given in lectures:

    $V[f]  = \sum\limits_{i=1}^{3} \sum\limits_{j=1}^{3} \alpha_i \alpha_j \mathsf{cov} [X_{(i)}, X_{(j)}]$ using expectation algebra, where $f = \alpha_1 X_{(1)} + \alpha_2 X_{(2)} + \alpha_3 X_{(3)}$.

    $V[f]  = \sum\limits_{i=1}^{3} \sum\limits_{j=1}^{3} \dfrac{\partial f}{\partial x_{(i)} }  \dfrac{\partial f}{\partial x_{(j)} } \bigg|_{x_{(j)}=\mu} \mathsf{cov}[x_{(i)}, x_{(j)}]$ using the first two terms of the Taylor expansion for $f$.

    a) Use pen and paper to show that these methods give the same answer for this function.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ###**Solution 5.1 [Pen & Paper Exercise]: Covariance/ Correlation of RVs**###


    a) find the covariance, cov(X,Y) where X=Nheads and Y=Ntails, from three coin tosses.

    Note that Y=3-X.

    We know from covariance algebra (week 4 uncertainties lectures)

    cov($\alpha$ X + c,\, $\beta$ Y + d) = $\alpha \beta$ cov(X,Y)

    and

    cov(X,X) = V[X]

    So, cov(X,Y) = cov(X, 3-X) = -cov(X,X) = -V[X]

    For X~Binomial we know from (week 3 special PDFs) lectures that:

    V[X] = V[Y] = np(1-p) = (3)(0.5)(0.5) = 0.75

    So cov(X,Y) = -0.75

    b) find the correlation coefficient, $\rho$(X,Y)

    From correlation coefficient definition (week 4 uncertainties lectures):

    $\rho$(X,Y) = $\dfrac{\mathsf{cov(x,y)}}{\sigma_x \sigma_y }$

    And from the definition of standard deviation $\sigma_x = \sigma_y = \sqrt{V[X]} = \sqrt{0.75}$.

    So, $\sigma_x \sigma_y = 0.75$.

    $\rho$(X,Y) $= \dfrac{-0.75}{0.75} = -1$

    X and Y have the highest possible level of negative linear correlation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ###**Solution 5.2 [Quick Python Exercise]: Covariance/Correlation of RVs**###

    a) Use python to calculate the mean, variance,standard deviation of each RV

    b) Use python to find the covariance matrix and correlation matrix for these two RVs

    c) What do the results indicate to you?
    """)
    return


@app.cell
def _():
    # spurious correlations
    # these data are a mixture from the fantastic https://www.tylervigen.com/spurious/correlation/
    import numpy as np

    # usa bachelors degrees awarded in math and stats, per year, ordered by year
    X1 = np.array([18_841, 20_449, 20_987, 21_854, 22_778, 24_075, 25_253, 26_155, 27_227, 27_092])
    print("\n X1 mean: {},  var: {}, sigma: {}".format( np.mean(X1), np.var(X1), np.std(X1) ) )
    # google searches for reddit (relative to some baseline)
    X2 = np.array([11, 16.3333, 23.75, 29.8333, 34.3333, 42.25, 47.5, 53.25, 57.0833, 58.1667])
    print("\n X2 mean: {},  var: {}, sigma: {}".format( np.mean(X2), np.var(X2), np.std(X2) ) )

    cov_mat = np.cov(X1,X2)
    print("\n Covariance matrix: \n",cov_mat)

    print("\n Yikes, why does cov(x,y)[0,0] not give me the same answer as var(X1)?!")
    # this is because numpy var defaults to the biased variance calculation with 1/N normalisation, whereas cov uses the unbiased 1/(N-1) normalisation

    print("\n X1 unbiased var: {}".format( np.var(X1,ddof=1) ) )
    print("\n X2 unbiased var: {}".format( np.var(X2,ddof=1) ) )

    print("\n With var(Xi,ddof=1) we get the matching diagonals")

    cc_mat = np.corrcoef(X1,X2)
    print("\n Correlation matrix: \n",cc_mat)

    print("\n We notice a very high positive linear correlation of ", cc_mat[0,1])

    print("\n This indicates a strong linear correlation between X1 and X2. So it may surprise you to learn the source of these data...")

    print("\n X1: USA bachelors degrees awarded in math and stats, per year, ordered by year")

    print("\n X2: google searches for reddit (relative to some baseline)")

    print("\n For more spurious correlations, see https://www.tylervigen.com/spurious/correlation/")
    return (np,)


@app.cell
def _(mo):
    mo.md(r"""
    ###**Solution 5.3 [Python Plot & Covariance Calculation]: Small Linear Correlation does not imply Independence**###

    a) Extract 100 values of speed from a Uniform(-160,160) distribution.

    b) Plot energy (y-axis) versus speed values (x-axis)

    c) Use numpy methods to find the covariance cov(E,v) and correlation coefficient $\rho(E,v)$

    d) What do the results indicate to you?
    """)
    return


@app.cell
def _(np):
    from scipy.stats import uniform
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    fig1, ax1 = plt.subplots(1, 1)

    uniform_dist = uniform(-160,320)
    speeds = uniform_dist.rvs(size=100)
    energies = 0.5*speeds**2

    ax1.scatter(speeds, energies)
    ax1.set_xlim(-160,160)
    plt.show()

    cov_mat_ev = np.cov(energies, speeds)
    print("\n Covariance matrix: \n",cov_mat_ev)

    cc_mat_ev = np.corrcoef(energies, speeds)
    print("\n Correlation matrix: \n",cc_mat_ev)

    print("\n Correlations are tiny: {}".format( cc_mat_ev[0,1] ) )
    print("\n Indication is that v and E are not *linearly correlated*. This is expected because E~v^2.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ###**Solution 5.4 [Python]: Variance of a Linear Function of Two RVs**###

    a) Use python to express the mean and uncertainty of the function in the form $f(X,Y) = \overline{f} \pm \sigma_f$
    """)
    return


@app.cell
def _(np):
    #A function of two RVs is f(X,Y) = 3x+y. 
    a=3
    b=1
    x=[62, 69, 67, 63, 63, 69, 67]
    y=[12, 11, 10, 14, 12, 13, 13]
    covxy = np.cov(x,y)
    var_x = covxy[0,0] # 8.9047619
    var_y = covxy[1,1] # 1.80952381
    cov_xy = covxy[0,1]

    mean_f = a*np.mean(x) + np.mean(y) # because E[sum] = sum(E) and E[aX] = aE[X]
    var_f = a**2 * var_x + b**2 * var_y + 2*a*b*covxy[0,1]

    print("f = {} +/- {}".format( mean_f, np.sqrt(var_f)))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ###**Solution 5.5 [Pen & Paper]: Two Approaches to finding the Variance of a Linear Function**###
    A function of three RVs is $f(x) = 2x_{(1)} + 3x_{(2)} + 4x_{(3)}$.

    Different methods for finding the variance of this function were given in lectures:

    $V[f]  = \sum\limits_{i=1}^{3} \sum\limits_{j=1}^{3} \alpha_i \alpha_j \mathsf{cov} [x_{(i)}, x_{(j)}]$ using expectation algebra, where $f = \alpha_1 x_{(1)} + \alpha_2 x_{(2)} + \alpha_3 x_{(3)}$.

    $V[f]  = \sum\limits_{i=1}^{3} \sum\limits_{j=1}^{3} \dfrac{\partial f}{\partial x_{(i)} }  \dfrac{\partial f}{\partial x_{(j)} } \bigg|_{x=\mu} \mathsf{cov}[x_{(i)}, x_{(j)}]$ using the first two terms of the Taylor expansion for $f$.

    a) Use pen and paper to show that these methods give the same answer for this function.

    **Speedy answer:**

    For the second method, we have partial derivatives in place of the coefficients $\alpha$. We can quickly see that the partial derivatives of a linear function do indeed return the coefficents. For example:

    $\dfrac{\partial f}{\partial x_{(1)} }\bigg|_{x=\mu} = \dfrac{\partial}{\partial x_{(1)} }\left(\alpha_1 X_{(1)} + \alpha_2 X_{(2)} + \alpha_3 X_{(3)}\right) = \alpha_1$

    This is sufficient to show that the two approaches are equivalent for a linear function.

    Notice that the definition has a specifier that after taking the derivative, we evaluate the result at $x=\mu$. This is irrelevant for a linear function, because there will not be any $x$ terms after taking the derivatives.

    **A bit more detail:**

    For the first method note that the $i,j$ sums are cycling through pairs, so we get $3^2=9$ terms.

    $\alpha_1=2, \alpha_2=3, \alpha_3=4$

    $V[f] = \alpha_1 \alpha_1\, \mathsf{cov}(x_{(1)}, x_{(1)}) + \alpha_1 \alpha_2\, \mathsf{cov}(x_{(1)}, x_{(2)}) + \alpha_1 \alpha_3 \,\mathsf{cov}(x_{(1)}, x_{(3)}) +$

    $\;\;\;\;\;\;\;\;\;\;\alpha_2 \alpha_1\, \mathsf{cov}(x_{(2)}, x_{(1)}) + \alpha_2 \alpha_2\, \mathsf{cov}(x_{(2)}, x_{(2)}) + \alpha_2 \alpha_3\, \mathsf{cov}(x_{(2)}, x_{(3)}) +$

    $\;\;\;\;\;\;\;\;\;\;\alpha_3 \alpha_1\, \mathsf{cov}(x_{(3)}, x_{(1)}) + \alpha_3 \alpha_2\, \mathsf{cov}(x_{(3)}, x_{(2)}) + \alpha_3 \alpha_3\, \mathsf{cov}(x_{(3)}, x_{(3)})$
    """)
    return


if __name__ == "__main__":
    app.run()
