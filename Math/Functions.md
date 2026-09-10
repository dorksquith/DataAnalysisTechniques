## **Functions**

Plotting some common functions with `matplotlib` and noting their characteristics ("eyeballing").

#### **Constant (aka Uniform)**

$y = c$

"A polynomial function of degree 0" = "a zeroth order polynomial".

![](y0.png)

#### **Linear**

$y \sim x$

"A polynomial function of degree 1"= "a first order polynomial".

$y = ax +b$
$a$
$x$
$y=ax+b$

Linear functions are **monotonic.** This means they never change direction.

![](y1.png)

#### **Quadratic**

$y \sim x^2$

"A polynomial function of degree 2" = "a second order polynomial".

$y=ax^2+bx +c$
$x$
$x$

A **positive** quadratic is **convex** (like a bowl) and symmetric about its **local minimum**, and a **negative** quadratic is **concave** and  symmetric about its **local maximum.**

$y=ax^2+bx +c$

**Question: are quadratic functions monotonic?**

![](y2.png)

#### **Cubic**

$y \sim x^3$

"A polynomial function of degree 3" = "a third order polynomial".

$y=ax^3+bx^2 +cx + d$
$x$
$y=ax^3+bx^2 +cx + d$

![](y3.png)

#### **Higher Order Polynomials**

$x^3$
$x^2$![](all.png)

#### **Mystery Polynomial**

What kind of polynomial is this? We will look at figuring this out shortly.

![](mystery.png)

#### **Absolute Value**

$y =x\\; \mathsf{for}\\; x\geq0$
$|x|$

`python: `[absolute](https://numpy.org/devdocs/reference/generated/numpy.absolute.html)

![](abs1.png)

**Reciprocal**

$y =\dfrac{1}{x}$

`python:`  [reciprocal](https://numpy.org/devdocs/reference/generated/numpy.reciprocal.html)

![recip-safe.png](https://canvas.sussex.ac.uk/courses/37537/files/6419527/preview)

![recip-safe-3.png](https://canvas.sussex.ac.uk/courses/37537/files/6419528/preview)

---

#### **Exercise 1.1**

**A) Make a plot** of your favourite polynomial, and put it in your portfolio notebook along with the python code you used to make it. Here is some example python in case it helps with a quick-start:

```py
def plot_absolute():
    plt.figure( figsize=(6,6) )
    x = np.linspace(-5.,5.,500)     
    y = np.abs(x)                          
    y2 = np.abs(x**2)                   
    plt.plot( x, y , c='dodgerblue', lw=2, ls='-', label=r'$y = |x|$')
    plt.plot( x, y2 , c='mediumseagreen', lw=2, ls=':', label=r'$y = |x^2|$')
    plt.grid()
    plt.xlabel( 'x', fontsize=16)
    plt.ylabel( 'y', fontsize=16)
    plt.ylim(0, 5)
    plt.xlim(-5,5)
    plt.legend(loc='lower right',fontsize=12)
    plt.title('My lovely function',fontsize=16)
    plt.show()
plot_absolute()
```

If you want to make your plots pretty, this is a good resource:[matplotlib quickstart](https://matplotlib.org/stable/users/explain/quick_start.html)

**B) Add latex markup formulae to your portfolio notebook.**

Examples:

- **Result** **Latex syntax for marimo markdown block** **Tips/gotchas**
- $y = ax^3 + bx^2 + cx + d$ (Marimo bug) Don't leave any space between the opening $ and the first character. Same with last character and closing $. \dfrac{numerator}{denominator} typesets as a nice big fraction. If you want a little fraction, use \frac{numerator}{denominator}
- $y=|x|$
- $\dfrac{dy}{dx} = \pi$

**NEW: How to make a table in marimo markdown cell - paste this in and then edit as desired**

`| Left Aligned | Center Aligned | Right Aligned |
| :----------- | :------------: | ------------: |
| Text         | $y=x^2$        | **bold text** |
| *Italic*     | $f(x)=\dfrac{1}{x^2}$        | dum de dum |`

If you are writing a dissertation as part of your MSc, I would recommend using LaTeX. You can find an introduction [here](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes) but note that you do not need to look at that for this module, as we will only be using little bits of latex for our notebooks and will learn as we go.

---

## **Python Polynomials and a sneak peek at fitting**

Let's have a look at the mystery polynomial.

![](mystery.png)

Download the data file from here: [MysteryPolynomial-XY.txt](https://canvas.sussex.ac.uk/courses/37537/files/6283184?wrap=1 "MysteryPolynomial-XY.txt")

We will try to fit the data using a library from `numpy` called [polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html).

The below code snippet tries to fit polynomials of degree 0 and 1 to the data. The fits are not good because they don't look anything like the data.  

```py
def fitpoly():
    x,y = np.loadtxt('MysteryPolynomial-XY.txt')
    plt.plot( x, y ,'ro',label='y=?')
    for deg in range(2):
	    pfit,stats = np.polynomial.polynomial.polyfit(x, y, deg, full=True)
	    yfit = np.polynomial.polynomial.polyval(x, pfit)
	    plt.plot( x, yfit, label=f'$y \sim x^{deg}$' )
	    print( f'stats for fit of degree {deg}: {stats}')
    plt.legend()
    plt.show()
fitpoly()
```

#### **Exercise 1.2**

**A) Run the code** snippet in your marimo notebook. The plot should look like this:

![](mystery-fit.png)

**B) Adapt the code**:

Add comments to describe what each line of code is doing

What is the second object (which we are calling stats) returned by the call to polyfit? Is the [documentation](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html) clear and helpful?

Adapt the for-loop to show higher-degree polynomial fits

Decide by eyeballing the fits what degree you think the polynomial is.

**C) Discussion.**

***** NEW***** The file used to make the mystery data txt file is [make-mystery-data.py](https://canvas.sussex.ac.uk/courses/37537/files/6422400?wrap=1 "make-mystery-data.py")

---

## **Probability Functions sneak peek**

The `scipy.stats` library is arguably the most useful resource for data scientists.

It has a multitude of useful methods, the most boring of which is the [uniform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html) probability distribution, `scipy.stats.uniform`.

Let's look at uniform and some of the things we can do with it. Here is a snippet:

```py
def uniform_prob():
    from scipy.stats import uniform
    a = 0.
    b = 1.
    n = 1000
    nrvs = 100
    u = uniform(loc=a, scale=b)
    start = u.ppf(q=0.01)
    stop  = u.ppf(q=0.99)
    x = np.linspace(start=start, stop=stop, num=n)
    u_pdf  = u.pdf(x=x)
    plt.plot(x, u_pdf, label='pdf')
    u_cdf  = u.cdf( x=x )
    plt.plot(x, u_cdf, color='red', ls='--', label='cdf')
    u_rvs  = u.rvs(size=nrvs)
    plt.hist(u_rvs, density=True,color='lightsteelblue',alpha=0.5,label=f'{nrvs} rvs')
    plt.title( f'Y ~ Uniform(a={a}, b={a+b})')
    plt.legend()
    plt.show()
uniform_prob()
```

#### **Exercise 1.3 (together)**

**A) Run the code** snippet in your marimo notebook.

Everybody's rvs plot is going to look slightly different. Add your name to the title and post in the slack channel so we can compare.

![](uniform-prob.png)

Discussion on randomness!

**B) Add comments to the code**

`uniform(loc=a, scale=b) `This is a uniform continuous probability distribution. The parameters specify the lower and upper edge of its domain. It returns an object.

`ppf(q) `is the [percent point function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.ppf.html) and its argument `q` specifies the probability as a fraction of 1. It returns a number, which is the value of `x` for which `uniform` has value `q`.

`pdf(x) `is the probability density function.

`cdf(x) `is the cumulative distribution function.

`rvs(size) `is an array of random variates.

**C) Play with the code**.

What happens if you change the `scale` parameter?

What is `pdf(0.5)` ?

What is `cdf(0.5)` ?

What `size` of `rvs` array is needed for your distribution to look uniform?

```py

```
