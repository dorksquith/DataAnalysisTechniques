## **Prep**

**1. Join the slack channel** for [Math4Data](https://join.slack.com/t/math4data/shared_invite/zt-3mm46rkvo-HQOB5qeSW~GdSa16rYqdhA)

**2. Make a marimo notebook** called `YourName_MDA.py`. This will be your portfolio/ lab book for the module, and will be graded after term finishes.

First cell: `import marimo as mo`, `import numpy as np` and `import matplotlib.pyplot as plt`

Add a markdown cell with the heading `**1.1 Functions**`.

Share -> publish html to web.

**3. Send me the link** to your notebook on slack as a direct message.

---

## **Functions**

Plotting some common functions with `matplotlib` and noting their characteristics ("eyeballing").

#### **Constant (aka Uniform)**

Written ![LaTeX: y = c](https://canvas.sussex.ac.uk/equation_images/y%2520%253D%2520c?scale=1 "y = c"), where ![LaTeX: c](https://canvas.sussex.ac.uk/equation_images/c?scale=1 "c") is a constant term (a number), or equivalently ![LaTeX: y = c\, x^0](https://canvas.sussex.ac.uk/equation_images/y%2520%253D%2520c%255C%252C%2520x%255E0?scale=1 "y = c\\, x^0") (because ![LaTeX: x^0 = 1](https://canvas.sussex.ac.uk/equation_images/x%255E0%2520%253D%25201?scale=1 "x^0 = 1")).

"A polynomial function of degree 0" = "a zeroth order polynomial".

![y0.png](https://canvas.sussex.ac.uk/courses/37537/files/6264931/preview)

#### **Linear**

Written ![LaTeX: y \sim x](https://canvas.sussex.ac.uk/equation_images/y%2520%255Csim%2520x?scale=1 "y \\sim x") ; spoken: "![LaTeX: y](https://canvas.sussex.ac.uk/equation_images/y?scale=1 "y") goes like ![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x")".

"A polynomial function of degree 1"= "a first order polynomial".

Can be written ![LaTeX: y = ax +b](https://canvas.sussex.ac.uk/equation_images/y%2520%253D%2520ax%2520%252Bb%2520?scale=1 "y = ax +b") where the **coefficients** ![LaTeX: a, b](https://canvas.sussex.ac.uk/equation_images/a%252C%2520b?scale=1 "a, b") can take any positive or negative value.

In this example, the coefficient ![LaTeX: a](https://canvas.sussex.ac.uk/equation_images/a?scale=1 "a") is attached to the **gradient term**, and ![LaTeX: b](https://canvas.sussex.ac.uk/equation_images/b?scale=1 "b") to the **constant term.**

A linear function of ![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x") is a **straight line**. If it has no constant term ![LaTeX: b](https://canvas.sussex.ac.uk/equation_images/b?scale=1 "b") (sometimes called the **intercept**), it passes through the origin ![LaTeX: \left(0,0\right)](https://canvas.sussex.ac.uk/equation_images/%255Cleft(0%252C0%255Cright)?scale=1 "\\left(0,0\\right)").

The **first derivative** of **![LaTeX: y=ax+b](https://canvas.sussex.ac.uk/equation_images/y%253Dax%252Bb?scale=1 "y=ax+b")** is constant: ![LaTeX: \dfrac{dy}{dx} = a](https://canvas.sussex.ac.uk/equation_images/%255Cdfrac%257Bdy%257D%257Bdx%257D%2520%253D%2520a?scale=1 "\\dfrac{dy}{dx} = a") . We will explore derivatives in the next topic [Calculus I](https://canvas.sussex.ac.uk/courses/37537/pages/calculus-i "Calculus I") .

Linear functions are **monotonic.** This means they never change direction.

![y1.png](https://canvas.sussex.ac.uk/courses/37537/files/6264932/preview)![add-lin.png](https://canvas.sussex.ac.uk/courses/37537/files/6264967/preview)

#### **Quadratic**

Written ![LaTeX: y \sim x^2](https://canvas.sussex.ac.uk/equation_images/y%2520%255Csim%2520x%255E2?scale=1 "y \\sim x^2") ; spoken: "![LaTeX: y](https://canvas.sussex.ac.uk/equation_images/y?scale=1 "y") goes like ![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x") squared".

"A polynomial function of degree 2" = "a second order polynomial".

Often written ![LaTeX: y=ax^2+bx +c](https://canvas.sussex.ac.uk/equation_images/y%253Dax%255E2%252Bbx%2520%252Bc?scale=1 "y=ax^2+bx +c") where ![LaTeX: a, b, c](https://canvas.sussex.ac.uk/equation_images/a%252C%2520b%252C%2520c%2520?scale=1 "a, b, c") are constants.

A quadratic function of ![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x") can have a constant term, and can have a linear term, but cannot have higher order terms in ![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x").

A quadratic function of ![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x") is a **parabola**. It is symmetric. If it has no constant term, its minimum touches the origin at ![LaTeX: \left(0,0\right)](https://canvas.sussex.ac.uk/equation_images/%255Cleft(0%252C0%255Cright)?scale=1 "\\left(0,0\\right)").

A **positive** quadratic is **convex** (like a bowl) and symmetric about its **local minimum**, and a **negative** quadratic is **concave** and  symmetric about its **local maximum.**

The **first** **derivative** of ![LaTeX: y=ax^2+bx +c](https://canvas.sussex.ac.uk/equation_images/y%253Dax%255E2%252Bbx%2520%252Bc?scale=1 "y=ax^2+bx +c") is ![LaTeX: \dfrac{dy}{dx} = 2ax + b](https://canvas.sussex.ac.uk/equation_images/%255Cdfrac%257Bdy%257D%257Bdx%257D%2520%253D%25202ax%2520%252B%2520b?scale=1 "\\dfrac{dy}{dx} = 2ax + b") : linear in ![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x").

**Question: are quadratic functions monotonic?**

![y2.png](https://canvas.sussex.ac.uk/courses/37537/files/6264933/preview)![add-quad.png](https://canvas.sussex.ac.uk/courses/37537/files/6264968/preview)

#### **Cubic**

Written ![LaTeX: y \sim x^3](https://canvas.sussex.ac.uk/equation_images/y%2520%255Csim%2520x%255E3?scale=1 "y \\sim x^3") ; spoken: "![LaTeX: y](https://canvas.sussex.ac.uk/equation_images/y?scale=1 "y") goes like ![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x") cubed".

"A polynomial function of degree 3" = "a third order polynomial".

Can be written ![LaTeX: y=ax^3+bx^2 +cx + d](https://canvas.sussex.ac.uk/equation_images/y%253Dax%255E3%252Bbx%255E2%2520%252Bcx%2520%252B%2520d?scale=1 "y=ax^3+bx^2 +cx + d") where ![LaTeX: a, b, c, d](https://canvas.sussex.ac.uk/equation_images/a%252C%2520b%252C%2520c%252C%2520d%2520?scale=1 "a, b, c, d") are constants.

A cubic function of ![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x") is s-shaped **(sigmoid)** and has the same sign as **![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x").**

The derivative of ![LaTeX: y=ax^3+bx^2 +cx + d](https://canvas.sussex.ac.uk/equation_images/y%253Dax%255E3%252Bbx%255E2%2520%252Bcx%2520%252B%2520d?scale=1 "y=ax^3+bx^2 +cx + d") is ![LaTeX: \dfrac{dy}{dx} = 3ax^2 + 2bx + c](https://canvas.sussex.ac.uk/equation_images/%255Cdfrac%257Bdy%257D%257Bdx%257D%2520%253D%25203ax%255E2%2520%252B%25202bx%2520%252B%2520c?scale=1 "\\dfrac{dy}{dx} = 3ax^2 + 2bx + c") : quadratic in ![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x").

![y3.png](https://canvas.sussex.ac.uk/courses/37537/files/6264934/preview)![add-cubic.png](https://canvas.sussex.ac.uk/courses/37537/files/6264966/preview)

#### **Higher Order Polynomials**

If the order of a polynomial is an odd number, it will resemble ![LaTeX: x^3](https://canvas.sussex.ac.uk/equation_images/x%255E3?scale=1.16666875 "x^3") : an **odd function,** for which **![LaTeX: f(-x) =-f(x)](https://canvas.sussex.ac.uk/equation_images/f(-x)%2520%253D-f(x)?scale=1 "f(-x) =-f(x)")**.

If it is even, it will resemble ![LaTeX: x^2](https://canvas.sussex.ac.uk/equation_images/x%255E2?scale=1.16666875 "x^2"): an **even function**, for which ![LaTeX: f(-x) =f(x)](https://canvas.sussex.ac.uk/equation_images/f(-x)%2520%253Df(x)?scale=1 "f(-x) =f(x)").

![all.png](https://canvas.sussex.ac.uk/courses/37537/files/6264969/preview)

#### **Mystery Polynomial**

What kind of polynomial is this? We will look at figuring this out shortly.

![mystery.png](https://canvas.sussex.ac.uk/courses/37537/files/6283185/preview)

#### **Absolute Value**

The absolute value function maps ![LaTeX: y =x\; \mathsf{for}\; x\geq0](https://canvas.sussex.ac.uk/equation_images/y%2520%253Dx%255C%253B%2520%255Cmathsf%257Bfor%257D%255C%253B%2520x%255Cgeq0?scale=1 "y =x\\; \\mathsf{for}\\; x\\geq0") and ![LaTeX: y =-x\; \mathsf{for}\; x&lt;0](https://canvas.sussex.ac.uk/equation_images/y%2520%253D-x%255C%253B%2520%255Cmathsf%257Bfor%257D%255C%253B%2520x%253C0?scale=1 "y =-x\\; \\mathsf{for}\\; x&lt;0").

The symbol ![LaTeX: |x|](https://canvas.sussex.ac.uk/equation_images/%257Cx%257C?scale=1 "|x|") indicates the absolute value, also called the **modulus** or mod of ![LaTeX: x](https://canvas.sussex.ac.uk/equation_images/x?scale=1 "x").

`python: `[absolute](https://numpy.org/devdocs/reference/generated/numpy.absolute.html)

![abs1.png](https://canvas.sussex.ac.uk/courses/37537/files/6264985/preview)![abs2.png](https://canvas.sussex.ac.uk/courses/37537/files/6264986/preview)

**Reciprocal**

The reciprocal function maps ![LaTeX: y =\dfrac{1}{x}](https://canvas.sussex.ac.uk/equation_images/y%2520%253D%255Cdfrac%257B1%257D%257Bx%257D?scale=1 "y =\\dfrac{1}{x}") , such that ![LaTeX: y\to\infty\;\mathsf{for}\; x=0](https://canvas.sussex.ac.uk/equation_images/y%255Cto%255Cinfty%255C%253B%255Cmathsf%257Bfor%257D%255C%253B%2520x%253D0?scale=1 "y\\to\\infty\\;\\mathsf{for}\\; x=0"). Equivalent notation is ![LaTeX: y = x^{-1}](https://canvas.sussex.ac.uk/equation_images/y%2520%253D%2520x%255E%257B-1%257D?scale=1 "y = x^{-1}"). The notation ![LaTeX: y\to\infty](https://canvas.sussex.ac.uk/equation_images/y%255Cto%255Cinfty?scale=1 "y\\to\\infty") means that "![LaTeX: y](https://canvas.sussex.ac.uk/equation_images/y?scale=1 "y") approaches infinity". The value of ![LaTeX: y](https://canvas.sussex.ac.uk/equation_images/y?scale=1 "y") is **undefined** at ![LaTeX: x=0](https://canvas.sussex.ac.uk/equation_images/x%253D0?scale=1 "x=0"). This means the reciprocal function is **discontinuous.**

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
- LaTeX: y=ax^3+bx^2 +cx + d $y = ax^3 + bx^2 + cx + d$ (Marimo bug) Don't leave any space between the opening $ and the first character. Same with last character and closing $. \dfrac{numerator}{denominator} typesets as a nice big fraction. If you want a little fraction, use \frac{numerator}{denominator}
- LaTeX: y=|x| $y=|x|$
- LaTeX: \dfrac{dy}{dx} = \pi $\dfrac{dy}{dx} = \pi$

**NEW: How to make a table in marimo markdown cell - paste this in and then edit as desired**

`| Left Aligned | Center Aligned | Right Aligned |
| :----------- | :------------: | ------------: |
| Text         | $y=x^2$        | **bold text** |
| *Italic*     | $f(x)=\dfrac{1}{x^2}$        | dum de dum |`

If you are writing a dissertation as part of your MSc, I would recommend using LaTeX. You can find an introduction [here](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes) but note that you do not need to look at that for this module, as we will only be using little bits of latex for our notebooks and will learn as we go.

---

## **Python Polynomials and a sneak peek at fitting**

Let's have a look at the mystery polynomial.

![mystery.png](https://canvas.sussex.ac.uk/courses/37537/files/6283185/preview)

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

![mystery-fit.png](https://canvas.sussex.ac.uk/courses/37537/files/6284255/preview)

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

![uniform-prob.png](https://canvas.sussex.ac.uk/courses/37537/files/6284256/preview)

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
