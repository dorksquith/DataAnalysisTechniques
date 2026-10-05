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

(md-cheatsheet)=
# Appendix B: Markdown Cheatsheet

## Text

::::{tab-set}
:::{tab} Output
You can make text **bold** or *italic* or `monospace`.
:::

:::{tab} Markdown
````
You can make text **bold** or *italic* or `monospace`.
````
:::
::::

## Math

See the [LaTeX Cheat Sheet](latex-cheatsheet) for quick ref on typesetting math in LaTeX.

::::{tab-set}
:::{tab} Output
An example of writing math in markdown (LaTeX) is given in [](#eq:equation).

```{math}
:label: eq:equation

E = mc^2
```
:::
:::{tab} Markdown
````
An example of writing math in markdown is given in [](#eq:equation).

```{math}
:label: eq:equation

E = mc^2
```
````
:::
::::

::::{tab-set}
:::{tab} Output
You can also put math inline like this $F=ma$.
:::
:::{tab}Markdown
```
You can also put math inline like this $F=ma$.
```
:::
::::

::::{tab-set}
:::{tab} Output
An example of a multi-line equation is given in [this long example](#eq:multiline-equation) which uses the LaTeX `align` package.

```{math}
:label: eq:multiline-equation

\begin{align}

P(x | \theta) & = \int \limits_a^b e^{-4x} + \ln{3\theta} dx \\

              & = \mathsf{some\; more\; math} \\

              & = \pi \\              
\end{align}
```
:::
:::{tab}Markdown
````
An example of a multi-line equation is given in [this long example](#eq:multiline-equation)

```{math}
:label: eq:multiline-equation

\begin{align}

P(x | \theta) & = \int \limits_a^b \exp{-4x} + \ln{3\theta}\, dx \\

              & = \mathsf{some\; more\; math} \\

              & = \pi \\   
\end{align}
```
````
:::
::::

## Tables

::::{tab-set}
:::{tab} Output
An example of a simple table is given in [](tab:table).

```{table} Title of my table
:label: tab:table

|      | Thing 1      | Thing 2 | 
| ---  | ---      | ---   | 
| Hello  |  $E=mc^2$     |  **woo bold**   | 
| Goodbye |  3.14    |  1 | 
```
:::
:::{tab}Markdown
````
An example of a simple table is given in [](tab:table).

```{table} Title of my table
:label: tab:table

|         | Thing 1       | Thing 2         | 
| ---     | ---           | ---             | 
| Hello   |  $E=mc^2$     |  **woo bold**   | 
| Goodbye |  3.14         |  1              | 
```

````
:::
::::


## Executable code cells


Here is an executable code cell which will automatically run in your browser when you run `jupyter book start --execute`:

::::{tab-set}
:::{tab} Output
```{code-cell} python

import numpy as np

# a function to print the value of  3x^2 +4 for a given value of x
def afunc(x):
  y = 3 * x**2 + 4
  print(f"---> Part 1: y = {y}")	

# call the function for x=12
afunc(12)

```
:::
:::{tab}Markdown

````
```{code-cell} python

import numpy as np

# a function to print the value of  3x^2 +4 for a given value of x
def afunc(x):
  y = 3 * x**2 + 4
  print(f"--> Part 1: y = {y}")	

# call the function for x=12
afunc(12)

```
````
:::
::::

## Links

::::{tab-set}
:::{tab} Output
* This is a link to an external url: [](https://en.wikipedia.org/wiki/Main_Page). 
* This is a link to a label in this markdown file: [](#here). 
* This is a footnote[^fn].

(here)=
```something to link to```

[^fn]: some footnote that will appear at the end of the page. 

:::
:::{tab}Markdown
```
* This is a link to an external url: [](https://en.wikipedia.org/wiki/Main_Page). 
* This is a link to a label in this markdown file: [](#here). 
* This is a footnote[^fn].

(here)=
```something to link to```

[^fn]: some footnote that will appear at the end of the page.  
```
:::
::::


## Lists

::::{tab-set}
:::{tab} Output
1. Here
2. is
3. a list
:::
:::{tab}Markdown
```
1. Here
2. is
3. a list
```
:::
::::

::::{tab-set}
:::{tab} Output
- [ ] This 
- [ ] is
- [ ] a checklist
:::
:::{tab}Markdown
```
- [ ] This 
- [ ] is
- [ ] a checklist
```
:::
::::

::::{tab-set}
:::{tab} Output
This is a definition list
: which is quite tidy

:::
:::{tab}Markdown
```
This is a definition list
: which is quite tidy

```
:::
::::

## Headings

The level of a heading is controlled by the number of has symbols preceeding the title, for example `## Heading`, `### Subheading`.

Headings are helpful for navigation as they will appear in the contents to the left of the page.


## Horizontal line

If you like you can add a horizontal separator line with three dashes `---` on a new line.

---





