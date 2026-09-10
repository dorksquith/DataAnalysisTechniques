kernelspec:
  name: python3
  display_name: 'Python 3'

```{code-cell} python
hello = "hello"
there = "there"
phrase = f"{hello}, {there}!"
print(phrase)
```


```{code-cell} python3
:label: example1
    def plot_absolute():,
    	plt.figure( figsize=(6,6) ),
    	x  = np.linspace(-5,5,500),
    	y  = np.abs(x),
    	y2 = np.abs(x**2),
    
    	plt.plot( x, y , c='dodgerblue', lw=2, ls='-', label=r'$y = |x|$'),
    	plt.plot( x, y2 , c='mediumseagreen', lw=3, ls=':', label=r'$y = |x^2|$'),

    	plt.grid(),
    	plt.xlabel( 'x', fontsize=16),
    	plt.ylabel( 'y', fontsize=16),
    	plt.ylim(0, 5),
    	plt.xlim(-5,5),
    	plt.legend(loc='lower right',fontsize=12),
    	plt.title('My lovely function',fontsize=16),

		plt.show(),

    plot_absolute()
```

And here I reference [](#example1).




# Introduction

+++ {"part": "abstract"}
This is my abstract!
+++

I am a book about ... something! Wikipedia has [information about books](wiki:book): hover over the link for more information.

% An admonition containing a note
:::{note}
Books are usually written on paper ... But Jupyter Book can create _websites_!
:::

If you sold 100 books at \$10 per book, you'd have \$1000 dollars according to [](#eq:book). If instead you publish your Jupyter Book to the web for free, you'd have \$0 dollars!

% An arbitrary math equation
:::{math}
:name: eq:book

x \times y = z
:::

Sometimes when reading it is helpful to foster a _tranquil_ environment. The image in [](#fig:mountains) would be a perfect spot!

% A figure of a photograph of some mountains, followed by a caption
:::{figure} https://github.com/rowanc1/pics/blob/main/mountains.png?raw=true
:label: fig:mountains

A photograph of some beautiful mountains to look at whilst reading.
:::

Hover over [this link to a cool figure](xref:guide#subfigure)!

![A cool figure with two subfigures](xref:guide#subfigure)









