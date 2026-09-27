# Changing Variables


### Transforming PDFs

In general, if the relationship between two RVs $X$ and $Y$ is monotonic, we can show that their probabilities (the CDFs) are related as [](#eq:monoPDFs).

```{math}
:label: eq:monoPDFs
P(Y \leq y) = P(X \leq x)
```

:::{dropdown} Proof of [](#eq:monoPDFs)

$Y = f(X) \therefore X = f^{-1}(Y)$

```{math}
\begin{align}

P(Y \leq y ) & = P( f(X) \leq y )  \\
             & = P( f^{-1}( f(X) ) \leq f^{-1}(y) ) \\
             & = P( X ) \leq x )  

```
