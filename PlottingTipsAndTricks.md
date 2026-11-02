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

(chapter:plotting-tips)=
# Plotting Tips and Tricks

A practical reference for the `matplotlib` plotting examples used across the module. Each section gives a minimal working pattern followed by a complete example.

:::{note}
This is a first draft - I will add some more useful examples when I have time!
:::

---

(sec:plot-functions)=
## Basic Function Plots

The standard pattern for a clean single-panel line plot:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 300)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, x**2,      label=r'$x^2$')
ax.plot(x, x**3,      label=r'$x^3$')
ax.plot(x, np.abs(x), label=r'$|x|$')
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$f(x)$', fontsize=13)
ax.set_title('Common function types', fontsize=13)
ax.legend(fontsize=12)
#plt.tight_layout()
plt.show()
```

**Tips:**

- Use raw strings (`r'...'`) for LaTeX labels: `r'$x^2$'` renders as $x^2$.
- `np.linspace(a, b, n)` gives `n` evenly spaced points — 200–300 is usually enough for smooth curves.
- `ax.axhline(0)` and `ax.axvline(0)` draw reference lines through the origin.

### Multiple panels

Use `plt.subplots(1, n)` or `plt.subplots(m, n)` to create a grid of axes:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 300)

funcs = [
    (x,      r'$f(x) = x$',      'linear'),
    (x**2,   r'$f(x) = x^2$',    'quadratic'),
    (x**3,   r'$f(x) = x^3$',    'cubic'),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, (y, label, title) in zip(axes, funcs):
    ax.plot(x, y, 'steelblue')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axvline(0, color='black', linewidth=0.6)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(label, fontsize=12)
#plt.tight_layout()
plt.show()
```

---

(sec:plot-semilog)=
## Semilog Plots

When one axis spans several orders of magnitude, a logarithmic scale makes structure visible that would be invisible on a linear scale.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.1, 5, 300)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# --- semilog y (log scale on y-axis) ---
ax = axes[0]
ax.semilogy(x, np.exp(x),   label=r'$e^x$')
ax.semilogy(x, np.exp(2*x), label=r'$e^{2x}$')
ax.semilogy(x, np.exp(-x),  label=r'$e^{-x}$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f(x)$  (log scale)')
ax.set_title('semilogy')
ax.legend()

# --- semilog x (log scale on x-axis) ---
x2 = np.logspace(-1, 2, 300)   # 0.1 to 100 on a log scale
ax = axes[1]
ax.semilogx(x2, np.log(x2),  label=r'$\ln x$')
ax.semilogx(x2, np.log10(x2), label=r'$\log_{10} x$')
ax.set_xlabel(r'$x$  (log scale)')
ax.set_ylabel(r'$f(x)$')
ax.set_title('semilogx')
ax.legend()

#plt.tight_layout()
plt.show()
```

**Tips:**

- `ax.semilogy(x, y)` is equivalent to `ax.plot(x, y)` followed by `ax.set_yscale('log')`.
- `np.logspace(a, b, n)` gives `n` points from $10^a$ to $10^b$ — ideal for the x-axis of a `semilogx` plot.
- Values $\leq 0$ cannot be plotted on a log axis; make sure your domain avoids zero.

---

(sec:plot-derivatives)=
## Visualising Derivatives

Plotting a function alongside its derivative helps build intuition: the derivative is zero where $f$ has a stationary point, and its sign tells you whether $f$ is rising or falling.


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, sin

x_sym = symbols('x')

# Define f and compute its first two derivatives symbolically
f_sym  = sin(x_sym)**2
df_sym = diff(f_sym, x_sym)
d2f_sym = diff(f_sym, x_sym, 2)

# Convert to numerical functions
f   = lambdify(x_sym, f_sym,   'numpy')
df  = lambdify(x_sym, df_sym,  'numpy')
d2f = lambdify(x_sym, d2f_sym, 'numpy')

x = np.linspace(0, 2 * np.pi, 400)


fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
color_set = [(f,   r"$f(x) = \sin^2 x$",       'steelblue'),
             (df,  r"$f'(x)$",                  'darkorange'),
             (d2f, r"$f''(x)$",                 'seagreen'),
            ]
for ax, (func, label, col) in zip(axes, color_set):
    ax.plot(x, func(x), color=col, linewidth=2)
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax.set_xlabel(r'$x$')
    ax.set_title(label, fontsize=13)

plt.tight_layout()
plt.show()

```

**Tips:**

- `sharex=True` links the x-axes of all panels so zooming one zooms all.
- Using `sympy` + `lambdify` avoids hand-coding finite-difference approximations.
- Notice that $f'(x) = 0$ exactly where $f(x)$ reaches its peaks and troughs.

---

(sec:plot-bivariate)=
## Bivariate (2D) Function Plots

To plot a function of two variables $f(x, y)$, first build a grid of $(x, y)$ evaluation points with `np.meshgrid`, then evaluate $f$ at every point on the grid.

### Setting up the grid

```{code-cell} python
import numpy as np

x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)

# X and Y are 2-D arrays; X[i,j] = x_vals[j], Y[i,j] = y_vals[i]
X, Y = np.meshgrid(x_vals, y_vals)

# Evaluate f at every grid point — numpy broadcasting handles this automatically
Z = 4 * X * Y

print(f"X shape: {X.shape},  Z shape: {Z.shape}")
print(f"Z ranges from {Z.min():.2f} to {Z.max():.2f}")
```

### Contour plots

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

x_vals = np.linspace(0, 1, 200)
y_vals = np.linspace(0, 2, 200)
X, Y   = np.meshgrid(x_vals, y_vals)
Z      = 4 * X * Y

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Filled contour
ax = axes[0]
cf = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
fig.colorbar(cf, ax=ax, label=r'$f(x,y)$')
ax.set_title('contourf (filled)')
ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')

# Line contours with labels
ax = axes[1]
cs = ax.contour(X, Y, Z, levels=10, colors='steelblue')
ax.clabel(cs, inline=True, fontsize=8)
ax.set_title('contour + clabel')
ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')

# Filled + line overlay
ax = axes[2]
cf2 = ax.contourf(X, Y, Z, levels=20, cmap='plasma', alpha=0.8)
ax.contour(X, Y, Z, levels=10, colors='white', linewidths=0.6)
fig.colorbar(cf2, ax=ax, label=r'$f(x,y)$')
ax.set_title('contourf + contour overlay')
ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')

plt.suptitle(r'$f(x,y) = 4xy$', fontsize=13, y=1.02)
#plt.tight_layout()
plt.show()
```

### Surface plots

3D surface plots use `mpl_toolkits.mplot3d` and `projection='3d'`:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — required for projection='3d'

x_vals = np.linspace(0, 1, 60)
y_vals = np.linspace(0, 2, 60)
X, Y   = np.meshgrid(x_vals, y_vals)
Z      = 4 * X * Y

fig = plt.figure(figsize=(14, 4))

for i, (elev, azim, title) in enumerate([(30, -60, 'default view'),
                                          (50, 30,  'elevated view'),
                                          (10, 90,  'side view')]):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85)
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$f$')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

plt.suptitle(r'$f(x,y) = 4xy$', fontsize=13, y=1.02)
#plt.tight_layout()
plt.show()
```

**Tips:**

- `ax.view_init(elev, azim)` sets the camera angle: `elev` is the elevation in degrees above the horizontal, `azim` is the azimuthal angle in degrees.
- Use a coarser grid (e.g. 60×60 rather than 200×200) for surface plots — fine grids slow rendering without improving appearance.
- `alpha=0.85` makes the surface slightly transparent, which can help when overlaying a contour on the floor of the 3D plot.

### Combined contour + surface figure

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

def joint_integral_plot1(f_label=r'$f(x,y) = 4xy$',
                         x_range=(0, 1), y_range=(0, 2), n=80):
    """
    Plot a bivariate function f(x,y) = 4xy as:
      Row 1: contourf | contour+labels | contourf+overlay
      Row 2: surface (3 viewpoints)

    To adapt for a different function, change the definition of Z below.
    """
    import numpy as np
    x_vals = np.linspace(*x_range, n)
    y_vals = np.linspace(*y_range, n)
    X, Y   = np.meshgrid(x_vals, y_vals)

    # ---- Define f(x, y) here ----
    Z = 4 * X * Y
    # -----------------------------

    fig = plt.figure(figsize=(14, 8))

    # --- Row 1: contour variants ---
    ax1 = fig.add_subplot(2, 3, 1)
    cf  = ax1.contourf(X, Y, Z, levels=15, cmap='viridis')
    fig.colorbar(cf, ax=ax1)
    ax1.set_title('contourf'); ax1.set_xlabel(r'$x$'); ax1.set_ylabel(r'$y$')

    ax2 = fig.add_subplot(2, 3, 2)
    cs  = ax2.contour(X, Y, Z, levels=10, colors='steelblue')
    ax2.clabel(cs, inline=True, fontsize=7)
    ax2.set_title('contour + labels'); ax2.set_xlabel(r'$x$'); ax2.set_ylabel(r'$y$')

    ax3 = fig.add_subplot(2, 3, 3)
    cf3 = ax3.contourf(X, Y, Z, levels=15, cmap='plasma', alpha=0.8)
    ax3.contour(X, Y, Z, levels=10, colors='white', linewidths=0.5)
    fig.colorbar(cf3, ax=ax3)
    ax3.set_title('contourf + overlay'); ax3.set_xlabel(r'$x$'); ax3.set_ylabel(r'$y$')

    # --- Row 2: surface plots ---
    for col, (elev, azim, title) in enumerate([(30, -60, 'default'),
                                                (50,  30, 'elevated'),
                                                (10,  90, 'side')], start=4):
        ax = fig.add_subplot(2, 3, col, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85)
        ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$'); ax.set_zlabel(r'$f$')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

    plt.suptitle(f_label, fontsize=13, y=1.01)
    #plt.tight_layout()
    plt.show()

joint_integral_plot1()
```

---

<!--
(sec:plot-bivariate-exercise)=
## Exercise: Adapt `joint_integral_plot1` for a New Function

The function `joint_integral_plot1()` above plots $f(x, y) = 4xy$.

Edit it to instead plot:

$$f(x, y) = e^{2x} - 3y, \qquad -1 \leq x \leq 1, \quad -1 \leq y \leq 1$$

You will need to change:

1. The definition of `Z` inside the function (one line).
2. The `x_range` and `y_range` keyword arguments in the function call.
3. The `f_label` string passed to `suptitle`.

```{note}
This is the same function whose joint integral you computed analytically in [](#sec:joint-integrals): $\displaystyle\int_{-1}^{1}\int_{-1}^{1}(e^{2x}-3y)\,dy\,dx = e^2 - e^{-2} \approx 7.25$. Does the contour plot make the $3y$ term's symmetry visually obvious?
```
-->
---

(sec:plot-methods-ref)=
## `matplotlib` Methods Reference

A reference table of plotting methods introduced in this appendix.

| Method | Description |
|--------|-------------|
| `plt.subplots(m, n, figsize=(...))` | Create a figure with an $m\times n$ grid of axes |
| `ax.plot(x, y, label=..., color=..., linewidth=...)` | Line plot |
| `ax.axhline(y)` / `ax.axvline(x)` | Horizontal / vertical reference line |
| `ax.set_xlabel(s)` / `ax.set_ylabel(s)` | Axis labels (use raw strings for LaTeX) |
| `ax.set_title(s)` | Panel title |
| `ax.legend(fontsize=...)` | Add a legend |
| `plt.suptitle(s, y=...)` | Overall figure title (use `y>1` to lift above panels) |
| `plt.tight_layout()` | Adjust spacing to prevent label overlap |
| `ax.semilogy(x, y)` | Line plot with log-scaled y-axis |
| `ax.semilogx(x, y)` | Line plot with log-scaled x-axis |
| `np.meshgrid(x_vals, y_vals)` | Build 2-D coordinate arrays from 1-D arrays |
| `ax.contourf(X, Y, Z, levels=..., cmap=...)` | Filled contour plot |
| `ax.contour(X, Y, Z, levels=..., colors=...)` | Line contour plot |
| `ax.clabel(cs, inline=True)` | Add value labels to contour lines |
| `fig.colorbar(cf, ax=ax, label=...)` | Add a colourbar to a filled contour or surface |
| `fig.add_subplot(..., projection='3d')` | Add a 3-D axes panel |
| `ax.plot_surface(X, Y, Z, cmap=..., alpha=...)` | 3-D surface plot |
| `ax.view_init(elev=..., azim=...)` | Set 3-D camera angle |
