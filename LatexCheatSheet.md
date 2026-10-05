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
  headings: false
  equation: true
---

# LaTeX Cheat Sheet

This is a very small selection intended for quick reference. See the [comprehensive list](https://tug.ctan.org/info/symbols/comprehensive/symbols-a4.pdf).

See the [Markdown Cheat Sheet](md-cheatsheet) for putting LaTeX in markdown math blocks rather than the inline examples here. 


## Integrals, Derivatives, Sums

$\int \limits_0^{\infty} f_X(x)\, dx$ : ```$\int \limits_0^{\infty} f_X\, dx$```

$ \int \limits_{a}^{b} x^2\, dx = \dfrac{x^3}{3}\bigg|_{a}^{b}$ : ```$\int \limits_{a}^{b} x^2\, dx = \dfrac{x^3}{3}\bigg|_{a}^{b}$```


$\dfrac{dy}{dx} = 3x^2 + 1$ : ```$\dfrac{dy}{dx} = 3x^2 + 1$```

$\dfrac{\partial y}{\partial x}$ : ```$\dfrac{\partial y}{\partial x}$```

$\dfrac{\partial^2 y}{\partial x^2}$ : ```$\dfrac{\partial^2 y}{\partial x^2}$```

$\sum \limits_0^{\infty} p_K(k)$ : ```$\sum \limits_0^{\infty} p_K(k)$```

## Fractions, Exponents, Subscripts

$\dfrac{a}{b} = e^{-x}$ : ```$\dfrac{a}{b}= e^{-x}$```

$e^{\frac{a}{b}} = d_1$ : ```$e^{\frac{a}{b}} = d_1$```

$\left( \dfrac{1}{N} \right) \cdot \left( \dfrac{1}{k} \right)$ : ```$\left( \dfrac{1}{N} \right) \cdot \left( \dfrac{1}{k} \right)$```

$ \dfrac{\rho +\bf{r} }{ \pi }$: ```$ \dfrac{\rho +\bf{r} }{ \pi }$```

## Selected Greek and Math Symbols

$\theta + \gamma + \Gamma + \beta$ : ```$\theta + \gamma + \Gamma + \beta$```

$\overline{x} \pm \sigma$: ```$ \overline{x} \pm \sigma$```

$0 \leq x \leq \infty$ : ```$0 \leq x \leq \infty$```

$\alpha \geq \epsilon$ : ```$\alpha \geq \epsilon$```

$\ln \mathcal{L}$ : ```$\ln \mathcal{L}$```

$P(A|B)$ : ```$P(A|B)$```

$P(A\cap B)$ : ```$P(A \cap B)$```

$P(A\cup B)$ : ```$P(A \cup B)$```

$\forall A$ : ```$\forall A$```

$A \in S$ : ```$A \in S$```

$A \notin S$ : ```$A \notin S$```

## Matrices

$\begin{bmatrix} A & B \\ C & D \end{bmatrix}$ : ```$\begin{bmatrix} A & B \\ C & D \end{bmatrix}$```


$\begin{vmatrix} A & B \\ C & D \end{vmatrix}$ : ```$\begin{vmatrix} A & B \\ C & D \end{vmatrix}$```


## Text and Whitespace

$I\; want\; text$: ```$I\; want\; text$```

$\mathsf{I\; want\; sans\; serif\; text}$: ```$\mathsf{I\; want\; sans\; serif\; text}$```

$\mathsf{Whitespace is ignored in math mode}$: ``` $\mathsf{Whitespace is ignored in math mode}$```

$Use\, \backslash, for\, small\, spaces$: ``` $Use\, \backslash, Comma\, gives\, small\, space$```

$Use\; \backslash; for\; bigger\; spaces$: ``` $Use\; \backslash; for\; bigger\; spaces$```

$\mathbf{Too\;\; much\;\; space\; ?}$: ```$Too\;\; much\;\; space\; ?$```