---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: 'Python 3'
numbering:
  title: false
  headings: true
  equation: true
---

(chapter:math-methods-a)=
# Math Methods A: Functions and Calculus

This appendix introduces the mathematical tools used throughout the module: functions, logarithms, differentiation, and integration. <!--Plotting examples for the functions and results covered here are collected in [PlottingTipsAndTricks](#chapter:plotting-tips).-->

---

(sec:functions)=
## Functions

A **function** $f$ maps each input value $x$ to exactly one output value $y$:

$$y = f(x)$$

### Common Function Types

**Constant:** $f(x) = c$. The output is the same for every input.

**Linear:** $f(x) = mx + c$. The gradient $m$ gives the slope; $c$ is the y-intercept.

**Quadratic:** $f(x) = ax^2 + bx + c$. The sign of $a$ determines whether the parabola opens upward ($a>0$) or downward ($a<0$).

**Cubic:** $f(x) = ax^3 + bx^2 + cx + d$.

**General polynomial of degree $n$:**

$$f(x) = \alpha_0 + \alpha_1 x + \alpha_2 x^2 + \cdots + \alpha_n x^n = \sum_{k=0}^{n} \alpha_k x^k$$

The **degree** of a polynomial is the highest power of $x$ with a non-zero coefficient.

### Symmetry

A function is **even** if $f(-x) = f(x)$ for all $x$ — it is symmetric about the y-axis. Examples: $x^2$, $\cos x$, $|x|$.

A function is **odd** if $f(-x) = -f(x)$ for all $x$ — it has rotational symmetry about the origin. Examples: $x$, $x^3$, $\sin x$.

### Special Cases

**Absolute value:** $|x| = x$ for $x \geq 0$ and $|x| = -x$ for $x < 0$.

**Reciprocal:** $f(x) = \dfrac{1}{x}$, defined for $x \neq 0$.

---

(sec:logs-exponents)=
## Bases, Exponents, and Logarithms

### Bases and Exponents

Any positive real number $a$ (the **base**) raised to a power $n$ (the **exponent**):

$$a^n = \underbrace{a \times a \times \cdots \times a}_{n \text{ times}}$$

Key rules:

| Rule | Expression |
|------|-----------|
| Product | $a^m \cdot a^n = a^{m+n}$ |
| Quotient | $a^m / a^n = a^{m-n}$ |
| Power of power | $(a^m)^n = a^{mn}$ |
| Zero exponent | $a^0 = 1$ |
| Negative exponent | $a^{-n} = 1/a^n$ |
| Fractional exponent | $a^{1/n} = \sqrt[n]{a}$ |

### Euler's Number

Euler's number $e \approx 2.71828\ldots$ is defined by:

$$e = \lim_{n\to\infty}\left(1 + \frac{1}{n}\right)^n$$

It is the unique base for which $\dfrac{d}{dx}e^x = e^x$.

### Logarithms

The logarithm $\log_b(x)$ is the inverse of exponentiation: $\log_b(b^x) = x$ and $b^{\log_b x} = x$.

The **natural logarithm** $\ln(x) \equiv \log_e(x)$ uses base $e$:

$$\ln(e^x) = x, \qquad e^{\ln x} = x$$

```{note}
In NumPy, `np.log(x)` computes the **natural logarithm** $\ln(x)$, not $\log_{10}(x)$. Use `np.log10(x)` for base-10 and `np.log2(x)` for base-2.
```

Key log rules:

| Rule | Expression |
|------|-----------|
| Product | $\ln(ab) = \ln a + \ln b$ |
| Quotient | $\ln(a/b) = \ln a - \ln b$ |
| Power | $\ln(a^n) = n \ln a$ |
| Change of base | $\log_b(x) = \ln(x)/\ln(b)$ |

### Motivation: Floating-Point Underflow

Likelihoods are products of many probability values, each $\leq 1$. For large datasets these products underflow to zero in floating-point arithmetic. Taking the logarithm converts the product to a sum, avoiding underflow:

$$\ln\mathcal{L} = \ln\prod_i f(x_i;\theta) = \sum_i \ln f(x_i;\theta)$$

### Trigonometric Functions

For an angle $\theta$ (in radians):

$$\sin\theta = \frac{\text{opposite}}{\text{hypotenuse}}, \quad \cos\theta = \frac{\text{adjacent}}{\text{hypotenuse}}, \quad \tan\theta = \frac{\sin\theta}{\cos\theta}$$

Key values:

| $\theta$ | $0$ | $\pi/6$ | $\pi/4$ | $\pi/3$ | $\pi/2$ |
|---|---|---|---|---|---|
| $\sin\theta$ | $0$ | $1/2$ | $1/\sqrt{2}$ | $\sqrt{3}/2$ | $1$ |
| $\cos\theta$ | $1$ | $\sqrt{3}/2$ | $1/\sqrt{2}$ | $1/2$ | $0$ |

Useful identities: $\sin^2\theta + \cos^2\theta = 1$; $\sin(-\theta) = -\sin\theta$; $\cos(-\theta) = \cos\theta$.

### Factorial

The factorial of a non-negative integer $n$ is:

$$n! = n \times (n-1) \times (n-2) \times \cdots \times 2 \times 1, \qquad 0! = 1$$

```{code-cell} python
import math
import numpy as np

# Logarithms
x = 100
print(f"np.log({x})   = {np.log(x):.4f}   (natural log, base e)")
print(f"np.log10({x}) = {np.log10(x):.4f}  (base 10)")
print(f"np.log2({x})  = {np.log2(x):.4f}  (base 2)")
print(f"np.exp(1)     = {np.exp(1):.6f}  (Euler's e)")

# Base representation
print(f"\n42 in base 2:  {np.base_repr(42, base=2)}")
print(f"42 in base 16: {np.base_repr(42, base=16)}")

# Factorial
for n in [0, 1, 5, 10]:
    print(f"{n}! = {math.factorial(n)}")
```

---

(sec:diff-calc-1)=
## Differential Calculus I

### Notation

The derivative of $y = f(x)$ with respect to $x$ can be written in several equivalent ways:

| Notation | Symbol | Named after |
|----------|--------|-------------|
| Leibniz | $\dfrac{dy}{dx}$ or $\dfrac{df}{dx}$ | Gottfried Leibniz |
| Lagrange | $f'(x)$ | Joseph-Louis Lagrange |
| Newton | $\dot{x}$ (typically for time derivatives) | Isaac Newton |

### The Power Rule

For $y = ax^n$:

$$\frac{dy}{dx} = a \cdot n \cdot x^{n-1}$$

Each term in a polynomial is differentiated independently:

$$\frac{d}{dx}\left(\sum_k \alpha_k x^k\right) = \sum_k k\,\alpha_k\, x^{k-1}$$

### Second Derivatives and Curvature

The second derivative $\dfrac{d^2y}{dx^2} = f''(x)$ measures the **curvature** of $f$:

- $f''(x) > 0$: concave up (curve bends upward)
- $f''(x) < 0$: concave down (curve bends downward)

### Stationary Points

A **stationary point** occurs where $f'(x) = 0$. Its nature is determined by the second derivative:

- $f''(x) > 0$: local **minimum**
- $f''(x) < 0$: local **maximum**
- $f''(x) = 0$: inconclusive (may be a point of inflection)

(sec:mma_taylor)=
### Taylor and Maclaurin Series

Consider a polynomial function $f(x) = \alpha_0 + \alpha_1 x + \alpha_2 x^2 + \alpha_3 x^3 + \cdots$

Evaluating the derivatives at $x=0$:

```{math}
\begin{aligned}
f(0) &= \alpha_0 \\
f'(0) &= \alpha_1 \\
f''(0) &= 2\alpha_2 \\
f'''(0) &= (3)(2)\alpha_3 \\
\end{aligned}
```

The pattern gives the coefficients in terms of derivatives:

```{math}
\alpha_n = \frac{f^{(n)}(0)}{n!}
```

where $f^{(n)}$ means the $n$th derivative of $f$.

Substituting back into the series:

```{math}
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!}\, x^n
```

And, more generally, expanding about an arbitrary point $a$:

```{math}
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}\, (x-a)^n
```

Any smooth function can be approximated near a point $a$ by a polynomial:

$$\label{eq:mma_taylor} f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

The special case $a = 0$ is the **Maclaurin series**:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!}\,x^n = \alpha_0 + \alpha_1 x + \alpha_2 x^2 + \cdots$$

Comparing with the general polynomial $f(x) = \sum_k \alpha_k x^k$, the coefficients satisfy:

$$\alpha_n = \frac{f^{(n)}(0)}{n!} \qquad \Longleftrightarrow \qquad f^{(n)}(0) = n!\,\alpha_n$$

This lets us read off polynomial coefficients directly from repeated derivatives at zero.

:::{note}
For a linear function of $x$, all terms after $n=1$ are zero — the Taylor series terminates exactly at first order.
:::


### Symbolic Differentiation with `sympy`

```{code-cell} python
from sympy import symbols, diff

x = symbols('x')

# Define a polynomial
f = 3*x**4 - 2*x**2 + 5*x - 1

print(f"f(x)    = {f}")
print(f"f'(x)   = {diff(f, x)}")
print(f"f''(x)  = {diff(f, x, 2)}")
print(f"f'''(x) = {diff(f, x, 3)}")
```

```{code-cell} python
# Verify the Maclaurin coefficient result: f^(n)(0) = n! * alpha_n
# For f(x) = 3x^4 - 2x^2 + 5x - 1, the coefficient of x^2 is alpha_2 = -2
# So f''(0) should equal 2! * (-2) = -4

import math
from sympy import symbols, diff

x = symbols('x')
f = 3*x**4 - 2*x**2 + 5*x - 1

for n in range(5):
    d = diff(f, x, n).subs(x, 0)
    alpha = d / math.factorial(n)
    print(f"n={n}: f^(n)(0) = {int(d):>5},  alpha_{n} = {alpha}")
```

---

(sec:diff-calc-2)=
## Differential Calculus II

### Definition from First Principles

The derivative is formally defined as a limit:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

This can be evaluated numerically by using a small finite $h$:

```{code-cell} python
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

def f(x):
    return x**3

x0 = 2.0
print(f"Numerical f'({x0}) ≈ {numerical_derivative(f, x0):.6f}")
print(f"Exact     f'({x0})  = {3 * x0**2:.6f}   (power rule: 3x² at x=2)")
```

### Derivatives of Special Functions

| Function $f(x)$ | Derivative $f'(x)$ |
|-----------------|-------------------|
| $e^x$ | $e^x$ |
| $e^{ax}$ | $a\,e^{ax}$ |
| $\ln x$ | $1/x$ |
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x = 1/\cos^2 x$ |
| $a^x$ | $a^x \ln a$ |

```{code-cell} python
from sympy import symbols, diff, exp, log, sin, cos, tan

x = symbols('x')

special_funcs = [exp(x), log(x), sin(x), cos(x), tan(x)]
for f in special_funcs:
    print(f"d/dx [{f}] = {diff(f, x)}")
```

### Partial Derivatives

For a function of several variables $f(x, y, \ldots)$, the **partial derivative** with respect to $x$ treats all other variables as constants:

$$\frac{\partial f}{\partial x} \equiv \lim_{h \to 0} \frac{f(x+h,\, y, \ldots) - f(x,\, y, \ldots)}{h}$$

```{code-cell} python
from sympy import symbols, diff, exp

x, y = symbols('x y')

f = x**2 * y + exp(x * y)

print(f"f(x,y)   = {f}")
print(f"∂f/∂x    = {diff(f, x)}")
print(f"∂f/∂y    = {diff(f, y)}")
print(f"∂²f/∂x²  = {diff(f, x, 2)}")
print(f"∂²f/∂x∂y = {diff(diff(f, x), y)}")
```

### Product Rule

For a product of two functions $h(x) = u(x)\,v(x)$:

$$h'(x) = u'(x)\,v(x) + u(x)\,v'(x)$$

This extends to three or more factors. For $h = uvw$:

$$h' = u'vw + uv'w + uvw'$$

(sec:mma-chain-rule)=
### Chain Rule

For a composite function $h(x) = f(g(x))$:

$$h'(x) = f'(g(x)) \cdot g'(x) \qquad \text{or equivalently} \qquad \frac{dh}{dx} = \frac{df}{dg}\cdot\frac{dg}{dx}$$

### Converting Symbolic Expressions to Numerical Functions

`sympy.lambdify` converts a symbolic expression into a callable Python function for numerical evaluation:

```{code-cell} python
from sympy import symbols, diff, sin, lambdify
import numpy as np

x = symbols('x')
f_sym = sin(x)**2

# Symbolic derivative
df_sym = diff(f_sym, x)
print(f"Symbolic derivative: {df_sym}")

# Convert to a numerical function
df_num = lambdify(x, df_sym, 'numpy')

# Evaluate at several points
x_vals = np.linspace(0, 2*np.pi, 5)
print(f"\nNumerical values at x = {x_vals.round(3)}:")
print(df_num(x_vals).round(4))
```

---

(sec:multivariable-calc)=
## Multivariable Calculus

(sec:mma-grad)=
### The Gradient

For a scalar function $f(x_1, x_2, \ldots, x_n)$, the **gradient** $\nabla f$ is a column vector of all first-order partial derivatives:

$$\nabla f = \begin{pmatrix} \dfrac{\partial f}{\partial x_1} \\[6pt] \dfrac{\partial f}{\partial x_2} \\ \vdots \\ \dfrac{\partial f}{\partial x_n} \end{pmatrix}$$

The gradient points in the direction of steepest ascent of $f$. Setting $\nabla f = \mathbf{0}$ locates stationary points in multiple dimensions.

(sec:mma_hessian)=
### The Hessian

The **Hessian** $\mathbf{H}$ (also written $\nabla^2 f$) is the square matrix of all second-order partial derivatives:

$$\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i\,\partial x_j}$$

The Hessian is always **symmetric**: $\mathbf{H}^T = \mathbf{H}$ (provided the mixed partials are continuous). For a function of two variables:

$$\mathbf{H} = \begin{pmatrix} \dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x\,\partial y} \\[10pt] \dfrac{\partial^2 f}{\partial y\,\partial x} & \dfrac{\partial^2 f}{\partial y^2} \end{pmatrix}$$

The Hessian is the Jacobian of the gradient (see below).

```{code-cell} python
from sympy import symbols, diff, Matrix

x, y = symbols('x y')

f = x**3 + 2*x*y**2 - y**3

# Gradient (as a column vector)
grad = Matrix([diff(f, x), diff(f, y)])
print("Gradient:")
print(grad)

# Hessian
H = Matrix([[diff(f, x, 2),          diff(diff(f, x), y)],
            [diff(diff(f, y), x),    diff(f, y, 2)]])
print("\nHessian:")
print(H)
print(f"\nSymmetric (H = Hᵀ)? {H == H.T}")
```

### The Jacobian

```{note}
The Jacobian is introduced here for completeness; a full worked example will be added in a later revision once the notation has been reviewed.
```

For a **vector-valued** function $\vec{f}(\mathbf{x}) = [f_1, f_2, \ldots, f_j]^T$ of $i$ variables $x_1, \ldots, x_i$, the **Jacobian** $\mathbf{J}$ is the matrix of all first-order partial derivatives, with one **row** per function component and one **column** per input variable (shape $j \times i$):

$$\mathbf{J}_{km} = \frac{\partial f_k}{\partial x_m}$$

The Jacobian of a scalar function's gradient is its Hessian.

---

(sec:integral-calc)=
## Integral Calculus

### Anti-differentiation

Integration is the reverse of differentiation. The **indefinite integral** of $f(x)$ is a function $F(x)$ such that $F'(x) = f(x)$:

$$\int f(x)\,dx = F(x) + k$$

where $k$ is an arbitrary **constant of integration**. Because any constant differentiates to zero, $k$ cannot be determined from $f$ alone.

```{warning}
`sympy.integrate()` does **not** include the constant of integration $k$. Always add it yourself when writing up indefinite integrals.
```

### The Power Rule for Integration

$$\int ax^n\,dx = \frac{a}{n+1}\,x^{n+1} + k \qquad (n \neq -1)$$

For $n = -1$: $\displaystyle\int \frac{1}{x}\,dx = \ln|x| + k$.

### Definite Integrals

The **definite integral** from $a$ to $b$ gives the signed area under the curve:

$$\int_a^b f(x)\,dx = \Big[F(x)\Big]_a^b = F(b) - F(a)$$

### Useful Integrals

| $f(x)$ | $\displaystyle\int f(x)\,dx$ |
|--------|----------------|
| $x^n$ $(n\neq-1)$ | $\dfrac{x^{n+1}}{n+1} + k$ |
| $\dfrac{1}{x}$ | $\ln\|x\| + k$ |
| $e^x$ | $e^x + k$ |
| $e^{ax}$ | $\dfrac{e^{ax}}{a} + k$ |
| $a^x$ | $\dfrac{a^x}{\ln a} + k$ |
| $\ln x$ | $x\ln x - x + k$ |
| $\log_b x$ | $\dfrac{x\log_b x - x}{\ln b} + k$ |
| $\sin x$ | $-\cos x + k$ |
| $\cos x$ | $\sin x + k$ |

### Integration by Substitution

Integration by substitution is the reverse of the chain rule. For a substitution $u = g(x)$, so that $du = g'(x)\,dx$:

$$\int f(g(x))\,g'(x)\,dx = \int f(u)\,du$$

**Example 1:** $\displaystyle\int 3x^2 e^{x^3}\,dx$

Let $u = x^3$, so $du = 3x^2\,dx$:

$$\int 3x^2 e^{x^3}\,dx = \int e^u\,du = e^u + k = e^{x^3} + k$$

```{code-cell} python
from sympy import symbols, integrate, exp

x = symbols('x')

def f(x):
    return 3*x**2 * exp(x**3)

result = integrate(f(x), x)
print(f"∫ {f(x)} dx = {result}")
# Note: sympy omits the constant of integration k
```

**Example 2:** $\displaystyle\int (x+1)^2\,dx$

Let $u = x+1$, so $du = dx$:

$$\int (x+1)^2\,dx = \int u^2\,du = \frac{u^3}{3} + k = \frac{(x+1)^3}{3} + k$$

```{code-cell} python
from sympy import symbols, integrate, expand

x = symbols('x')

def f(x):
    return (x+1)**2

result = integrate(f(x), x)
print(f"∫ {f(x)} dx = {result}")
print()

# sympy expands and drops the additive constant 1/3
# Verify by expanding (x+1)^3/3:
check = expand( ((x+1)**3) / 3 )
print(f"Check: (x+1)³/3 expanded = {check}")
print()
print("sympy's result is x³/3 + x² + x.")
print("The full expansion of (x+1)³/3 is x³/3 + x² + x + 1/3.")
print("The 1/3 is an additive constant, absorbed into k.")
```

```{warning}
`sympy.integrate()` can silently drop **additive constants** from a result. In the example above, both $x^3/3 + x^2 + x$ and $(x+1)^3/3$ are valid anti-derivatives — they differ only by the constant $1/3$ — but sympy returns the expanded form rather than the compact factored form. I am not currently aware of a workaround for this.
```

(sec:mma-int-by-parts)=
### Integration by Parts

Integration by parts is the reverse of the product rule. For two functions $u(x)$ and $v(x)$:

$$\int v\,du = u\,v - \int u\,dv$$

**Strategy:** choose $u'$ as the factor that is easy to integrate, and $v$ as the factor whose derivative simplifies the remaining integral.

**Example:** $\displaystyle\int \ln x\,dx$

Choose $u' = 1$ (so $du = dx$, $u = x$) and $v = \ln x$ (so $dv = 1/x\, dx$):

$$\int \ln x\,dx = x\ln x - \int x \cdot \frac{1}{x}\,dx = x\ln x - \int 1\,dx = x\ln x - x + k$$

```{code-cell} python
from sympy import symbols, integrate, log

x = symbols('x')

result = integrate(log(x), x)
print(f"∫ ln(x) dx = {result}")
# sympy uses log() for the natural logarithm
```

---

(sec:joint-integrals)=
## Joint and Marginal Integrals

For a function of two variables $f(x, y)$, the **double integral** over a rectangular region is:

$$\int_{x_0}^{x_1}\int_{y_0}^{y_1} f(x,y)\,dy\,dx$$

Evaluation proceeds by integrating the **inner variable first** (here $y$, treating $x$ as a constant), which yields a function of $x$ alone — sometimes called the **marginal integral**. The outer integral over $x$ then gives a scalar.

By **Fubini's theorem**, the order of integration can be swapped without changing the result (provided $f$ is continuous):

$$\int_{x_0}^{x_1}\int_{y_0}^{y_1} f(x,y)\,dy\,dx = \int_{y_0}^{y_1}\int_{x_0}^{x_1} f(x,y)\,dx\,dy$$

### Example 1: $f(x,y) = 4xy$ over $0 < x < 1$, $\;0 < y < 2$

Inner integral (over $y$, $x$ held constant):

$$\int_0^2 4xy\,dy = 4x\left[\frac{y^2}{2}\right]_0^2 = 4x \cdot 2 = 8x$$

Outer integral (over $x$):

$$\int_0^1 8x\,dx = 8\left[\frac{x^2}{2}\right]_0^1 = 4$$

```{code-cell} python
from sympy import symbols, integrate

x, y = symbols('x y')

def f(x, y):
    return 4 * x * y

result = integrate(f(x, y), (y, 0, 2), (x, 0, 1))
print(f"∬ {f(x, y)} dy dx  (0<y<2, 0<x<1) = {result}")
```

### Example 2: $f(x,y) = e^{2x} - 3y$ over $-1 < x < 1$, $\;-1 < y < 1$

The $3y$ term integrates to zero over the symmetric interval $[-1, 1]$, leaving:

$$\int_{-1}^{1}\int_{-1}^{1}(e^{2x}-3y)\,dy\,dx = \int_{-1}^{1}2e^{2x}\,dx = \left[e^{2x}\right]_{-1}^{1} = e^2 - e^{-2} \approx 7.25$$

```{code-cell} python
from sympy import symbols, integrate, exp

x, y = symbols('x y')

def f(x, y):
    return exp(2*x) - 3*y

result = integrate(f(x, y), (y, -1, 1), (x, -1, 1))
print(f"∬ {f(x, y)} dy dx  (-1<y<1, -1<x<1) = {result}")
print(f"Numerically ≈ {float(result):.4f}")
```

---

(sec:sympy-methods-ma)=
## `sympy` Methods Reference

A reference table of `sympy` methods introduced in this appendix.

```{warning}
`sympy.integrate()` does **not** include the constant of integration. It may also return an expanded form that omits additive constants — always verify by differentiating or expanding symbolically.
```

| Method | Description | Section |
|--------|-------------|---------|
| `symbols('x y')` | Define one or more symbolic variables | §3 |
| `Symbol('x')` | Define a single symbolic variable | §3 |
| `diff(expr, x)` | Differentiate `expr` with respect to `x` | §3 |
| `diff(expr, x, n)` | $n$th derivative with respect to `x` | §3 |
| `diff(diff(expr, x), y)` | Mixed partial derivative $\partial^2/\partial x\,\partial y$ | §4, §5 |
| `integrate(expr, x)` | Indefinite integral (no constant of integration added) | §6 |
| `integrate(expr, (x, a, b))` | Definite integral from $a$ to $b$ | §6 |
| `integrate(expr, (y, a, b), (x, c, d))` | Double integral — inner variable listed first | §7 |
| `expand(expr)` | Expand a symbolic expression | §6 |
| `lambdify(x, expr, 'numpy')` | Convert symbolic expression to callable numerical function | §4 |
| `Matrix([...])` | Create a symbolic matrix (for gradient / Hessian) | §5 |
| `log(x)` | Natural logarithm $\ln(x)$ — **not** $\log_{10}(x)$ | §2 |
| `exp(x)` | Exponential $e^x$ | §2, §6 |
| `expr.subs(x, val)` | Substitute a numerical value for a symbol | §3 |
