(chapter:vocab)=
# Vocabulary & Glossary

A concise reference defining the key terms used throughout this module, organised by topic in the order they are introduced.

---

(sec:vocab_data)=
## Data & Sampling

**Hypothesis**
: An educated guess about the outcome of an event or the value of a parameter. See [](#sec:tandc_hypmod).

**Null Hypothesis ($H_0$)**
: The default assumption — typically that there is *no* effect or relationship. See [](#sec:tandc_hypmod).

**Alternative Hypothesis ($H_1$)**
: The claim that there *is* an effect or relationship; accepted (provisionally) if the data provide sufficient evidence against $H_0$. See [](#sec:tandc_hypmod).

**Model**
: A mathematical equation, computer program, or object that describes how something works or is used to make predictions. See [](#sec:tandc_hypmod).

**Theory**
: A complete, internally consistent description of a set of models and the relationships between them; a candidate explanation for the Underlying Truth. See [](#sec:tandc_hypmod).

**Underlying Truth**
: The true (and unknowable) probability distribution from which all observations are drawn. Models and theories are approximations to it. See [](#sec:tandc_hypmod).

**Population**
: The complete set of all possible observations relevant to a hypothesis or model. See [](#sec:tandc_popsam).

**Sample**
: A finite subset of the population, used as a proxy for it. See [](#sec:tandc_popsam).

**Representative Sample**
: A sample that is large enough and sufficiently unbiased to faithfully reflect the population. See [](#sec:tandc_popsam).

**Random Variable (RV)**
: A quantity whose value is the outcome of a random (sampling) process. Usually denoted by a capital letter, e.g. $X$. See [](#sec:tandc_ranvar).

**Realisation**
: A single observed value drawn from an RV, usually denoted by the corresponding lower-case letter, e.g. $x$. See [](#sec:tandc_ranvar).

**Parameter**
: A number characterising the Underlying Truth distribution. Usually denoted $\theta$ or $\boldsymbol{\theta}$. See [](#sec:tandc_ranvar).

**Statistic**
: A number computed from a sample that characterises that sample (e.g. the sample mean). See [](#sec:tandc_ranvar).

**Discrete RV**
: An RV that can only take values from a countable set (e.g. integers). Its probabilities are described by a [PMF](#sec:vocab_dist). See [](#sec:tandc_dis).

**Continuous RV (CRV)**
: An RV that can take any value on the real line (or an interval of it). Its probabilities are described by a [PDF](#sec:vocab_dist). See [](#sec:tandc_con).

**Independent & Identically Distributed (IID)**
: A collection of RVs is IID if every variable has the *same* probability distribution **and** the variables are mutually independent. See [](#sec:tandc_iid).

**Mutual Independence**
: Measuring (or knowing) any one variable gives no information about the probability distribution of any other. See [](#sec:tandc_iid).

---

(sec:vocab_sumstat)=
## Summary Statistics

**Summary Statistic**
: A single number summarising a property of an entire dataset (e.g. mean, variance). See [](#sec:tandc_sumsta).

**Sample Mean ($\overline{x}$)**
: The arithmetic average of a dataset of $N$ values: $\overline{x} = \frac{1}{N}\sum_{i=1}^N x_i$. See [](#sec:tandc_mean).

**Expected Value / Expectation ($E[X]$)**
: The theoretical (population-level) mean of an RV, defined in terms of probabilities rather than observed counts. For a discrete RV: $E[X] = \sum_i x_i p_i$. The expectation is a **parameter**, not a statistic. See [](#sec:tandc_mean).

**Binned Mean**
: The mean computed from binned (histogram) data: $\overline{x} = \frac{1}{N}\sum_i n_i c_i$, where $n_i$ is the count and $c_i$ the centre of bin $i$. See [](#sec:tandc_binmea).

**Weighted Mean ($\overline{x}_w$)**
: A generalisation of the mean where each measurement is scaled by a weight $w_i$ (e.g. the inverse variance): $\overline{x}_w = \frac{\sum_i x_i w_i}{\sum_i w_i}$. See [](#sec:tandc_weimea).

**Median**
: The central value of a sorted dataset — half of observations lie above it, half below. See [](#sec:tandc_medmod).

**Mode**
: The value occurring with the greatest frequency in a dataset. See [](#sec:tandc_medmod).

**Sample Variance ($v_x$)**
: A measure of spread: $v_x = \frac{1}{N}\sum_i(x_i - \overline{x})^2$. The *unbiased* form uses $N-1$ in the denominator. See [](#sec:tandc_var).

**Standard Deviation ($\sigma_x$)**
: The square root of the variance; in the same units as the data. See [](#sec:tandc_stadev).

**Coefficient of Variation (CV)**
: The standard deviation divided by the mean, expressed as a percentage: $CV = \sigma_x / \overline{x} \times 100\%$. A dimensionless measure of relative spread. See [](#sec:tandc_coevar).

---

(sec:vocab_prob)=
## Probability

**Probability ($P$)**
: A real number in $[0, 1]$ quantifying how likely an event is. See [](#chapter:pe).

**Frequentist Probability**
: Probability defined as the long-run frequency of an event over infinitely many identical trials. See [](#sec:pe_twophi).

**Bayesian Probability**
: Probability interpreted as a degree of belief, updated using Bayes' theorem as new evidence arrives. See [](#sec:pe_twophi).

**Sample Space ($\Omega$)**
: The set of all possible outcomes of an experiment. See [](#chapter:pe).

**Event**
: A subset of the sample space — one or more outcomes grouped together. See [](#chapter:pe).

**Conditional Probability ($P(A|B)$)**
: The probability of event $A$ given that event $B$ has occurred: $P(A|B) = P(A \cap B) / P(B)$. See [](#chapter:pe).

**Joint Probability ($P(A \cap B)$)**
: The probability that both events $A$ and $B$ occur. See [](#chapter:pe).

**Marginal Probability**
: The probability of an event considered without reference to any other event; obtained by summing (or integrating) over all values of the other variable. See [](#chapter:pe).

**Bayes' Theorem**
: Relates the conditional and marginal probabilities: $P(A|B) = P(B|A)\,P(A) / P(B)$. The foundation of Bayesian inference. See [](#chapter:pe) and [](#sec:bay_baythm).

**Law of Total Probability**
: If $\{B_i\}$ partition $\Omega$, then $P(A) = \sum_i P(A|B_i)\,P(B_i)$. See [](#chapter:pe).

---

(sec:vocab_dist)=
## Probability Distributions

**Probability Mass Function (PMF)**
: Gives the probability $P(X = k)$ for each possible value $k$ of a **discrete** RV $X$. Must be non-negative and sum to 1. See [](#chapter:pmf).

**Probability Density Function (PDF)**
: Describes the relative likelihood of a **continuous** RV $X$ taking a given value. The probability that $X$ falls in $[a,b]$ is $\int_a^b f_X(x)\,dx$. The PDF must be non-negative and integrate to 1 over its support. See [](#chapter:pdf).

**Cumulative Distribution Function (CDF)**
: $F_X(x) = P(X \leq x)$; the integral (or sum) of the PDF (or PMF) from $-\infty$ to $x$. See [](#chapter:pdf).

**Support**
: The set of values for which a PMF or PDF is non-zero. See [](#chapter:pdf).

**Location Parameter**
: Shifts the distribution along the number line without changing its shape (e.g. the mean $\mu$ of a Normal distribution). See [](#chapter:pdf).

**Scale Parameter**
: Stretches or compresses the distribution (e.g. the standard deviation $\sigma$ of a Normal distribution). See [](#chapter:pdf).

**Shape Parameter**
: Changes the shape of the distribution without being a simple shift or rescaling (e.g. $k$ in a Gamma distribution). See [](#chapter:gpdf).

**Bernoulli Distribution**
: Models a single binary trial with success probability $p$. Special case of the Binomial with $n=1$. See [](#chapter:pmf).

**Binomial Distribution**
: Models the number of successes in $n$ independent Bernoulli trials, each with success probability $p$. See [](#chapter:pmf).

**Poisson Distribution**
: Models the number of events occurring in a fixed interval when events happen at a constant average rate $\lambda$ and independently of one another. See [](#chapter:pmf).

**Uniform Distribution**
: All outcomes in an interval $[a,b]$ are equally likely. See [](#chapter:pdf).

**Normal (Gaussian) Distribution**
: The bell-shaped distribution with mean $\mu$ and standard deviation $\sigma$. Central to statistics via the Central Limit Theorem. See [](#chapter:norm).

**Standard Normal Distribution**
: The Normal distribution with $\mu = 0$ and $\sigma = 1$, denoted $Z \sim \mathcal{N}(0,1)$. See [](#chapter:norm).

**Chi-squared Distribution ($\chi^2_k$)**
: The sum of the squares of $k$ independent standard Normal RVs. Arises naturally in goodness-of-fit tests and in the distribution of the sample variance. See [](#chapter:normt) and [](#chapter:lse).

**Beta Distribution**
: A flexible continuous distribution on $[0,1]$ with shape parameters $\alpha, \beta > 0$; used as a conjugate prior for Binomial likelihoods. See [](#chapter:gpdf).

**Gamma Distribution**
: A continuous distribution on $(0,\infty)$ with shape $k > 0$ and rate $\lambda > 0$ (or scale $\theta = 1/\lambda$); used as a conjugate prior for Poisson likelihoods. See [](#chapter:gpdf).

**Conjugate Prior**
: A prior distribution that, when combined with a given likelihood via Bayes' theorem, yields a posterior of the same distributional family. See [](#chapter:gpdf) and [](#chapter:bay).

---

(sec:vocab_est)=
## Estimators & Estimation

**Estimand**
: The true (unknown) quantity we wish to estimate, e.g. the population mean $\mu$. See [](#chapter:ei).

**Estimator ($\hat{\theta}$)**
: A function of the sample data used to estimate a parameter $\theta$. It is itself a random variable. See [](#chapter:ei).

**Estimate**
: The numerical value produced by an estimator when applied to a specific dataset — a single realisation of the estimator. See [](#chapter:ei).

**Bias**
: The systematic difference between the expected value of an estimator and the true parameter: $\text{Bias}[\hat\theta] = E[\hat\theta] - \theta$. See [](#sec:ei_meanest).

**Unbiased Estimator**
: An estimator whose bias is zero: $E[\hat\theta] = \theta$ for all $\theta$. See [](#sec:ei_meanest).

**Consistent Estimator**
: An estimator that converges to the true parameter value as the sample size $N \to \infty$. See [](#chapter:ei).

**Mean Squared Error (MSE)**
: Combines bias and variance: $\text{MSE}[\hat\theta] = \text{Bias}[\hat\theta]^2 + \text{Var}[\hat\theta]$. The MSE of an unbiased estimator equals its variance. See [](#chapter:mmb).

**Efficiency**
: A relative measure of how much variance an estimator has compared to the theoretical minimum (the Cramér–Rao bound). An efficient estimator achieves the minimum variance. See [](#chapter:ei).

**Minimum Variance Unbiased Estimator (MVUE)**
: The unbiased estimator with the smallest possible variance, i.e. one that achieves the Cramér–Rao bound. See [](#sec:mmb_mvue).

**Standard Error (SE)**
: The standard deviation of an estimator's sampling distribution — a measure of how much the estimate varies from sample to sample. See [](#chapter:ei).

**Standard Error of the Mean (SEM)**
: The SE of the sample mean estimator: $\text{SEM} = \sigma / \sqrt{N}$. See [](#sec:ei_meanest).

**Standard Error of the Variance (SEV)**
: The SE of the standard deviation estimator: $\text{SEV} \approx \sigma / \sqrt{2(N-1)}$. See [](#sec:ei_sev).

**Interval Estimation**
: Expressing an estimate as a range of plausible values (a confidence interval or credible interval) rather than a single point. See [](#sec:ei_intest).

**Bootstrap**
: A resampling method that estimates the sampling distribution of a statistic by repeatedly sampling (with replacement) from the observed dataset. See [](#chapter:ei) and [](#chapter:mmb).

**Sufficient Statistic**
: A statistic $T(X)$ that captures all the information in the data relevant to estimating $\theta$; the conditional distribution of the data given $T(X)$ does not depend on $\theta$. See [](#sec:mmb_raobl).

---

(sec:vocab_mle)=
## Maximum Likelihood Estimation

**Likelihood ($\mathcal{L}$)**
: The probability of observing the data as a function of the parameter(s), with the data held fixed: $\mathcal{L}(\theta; \mathbf{x}) = P(\mathbf{x}|\theta)$. The likelihood is **not** a probability over $\theta$. See [](#sec:ei_like).

**Log-Likelihood ($\ell$)**
: The natural logarithm of the likelihood: $\ell(\theta) = \ln\mathcal{L}(\theta)$. Used in practice because it converts products to sums and is easier to differentiate. See [](#sec:mle_loglike).

**Negative Log-Likelihood (NLL)**
: $-\ell(\theta)$; minimising the NLL is equivalent to maximising the likelihood. Preferred for numerical optimisation. See [](#sec:ei_nll).

**Maximum Likelihood Estimate (MLE)**
: The value $\hat\theta_\text{MLE}$ that maximises the likelihood (or equivalently the log-likelihood) for a given dataset. See [](#chapter:mle).

**Score Function**
: The derivative of the log-likelihood with respect to the parameter(s): $s(\theta) = \partial\ell/\partial\theta$. At the MLE, the score equals zero. See [](#sec:mle_sco).

**Fisher Information ($\mathcal{I}$)**
: The expected value of the squared score: $\mathcal{I}(\theta) = E[(s(\theta))^2]$. Measures the information a single observation carries about $\theta$. See [](#sec:mle_inf).

**Information Matrix**
: The multiparameter generalisation of Fisher information: $\mathcal{I}_{ij} = -E\!\left[\partial^2\ell/\partial\theta_i\partial\theta_j\right]$. Its inverse gives the lower bound on the covariance matrix of any unbiased estimator. See [](#sec:mle_inf).

**Cramér–Rao Bound (CRB) / Minimum Variance Bound (MVB)**
: The lower bound on the variance of any unbiased estimator: $\text{Var}[\hat\theta] \geq 1/\mathcal{I}(\theta)$. An estimator achieving this bound is called efficient. See [](#sec:mle_mvb).

**Hessian Matrix**
: The matrix of second-order partial derivatives of the log-likelihood (or negative log-likelihood); its inverse at the MLE approximates the covariance matrix of the estimates. See [](#sec:mle_hes).

---

(sec:vocab_lse)=
## Least Squares

**Residual ($r_i$)**
: The difference between the $i^{\text{th}}$ observed value and the corresponding model prediction: $r_i = y_i - \hat{y}_i$. See [](#sec:lse_res).

**Chi-squared ($\chi^2$)**
: In least-squares fitting, the weighted sum of squared residuals: $\chi^2 = \sum_i r_i^2 / \sigma_i^2$. Minimising $\chi^2$ yields the least-squares estimates. See [](#sec:lse_chi2).

**Ordinary Least Squares (OLS)**
: Minimises the sum of squared residuals when all measurements have equal (or unknown) uncertainty. See [](#sec:lse_straline).

**Weighted Least Squares (WLS)**
: Minimises the chi-squared statistic, giving less weight to measurements with larger uncertainties. See [](#chapter:lse).

**Design Matrix ($A$)**
: The matrix encoding the values of the predictor variables for each observation; the linear model is $\mathbf{y} = A\boldsymbol{\theta} + \boldsymbol{\epsilon}$. See [](#sec:lse_mat).

**Normal Equations**
: The system of linear equations whose solution gives the least-squares estimates: $A^T W A\,\hat{\boldsymbol{\theta}} = A^T W \mathbf{y}$. See [](#sec:lse_mat).

**Gauss–Markov Theorem**
: States that, under certain conditions (zero-mean, uncorrelated, equal-variance errors), OLS gives the Best Linear Unbiased Estimator (BLUE). See [](#sec:lse_gaumar).

**Homoscedastic**
: Describing data where all measurements have the same variance ($\sigma_i = \sigma$ for all $i$). See [](#sec:lse_hom).

**Heteroscedastic**
: Describing data where measurements have different variances ($\sigma_i$ varies with $i$). See [](#sec:lse_hom).

**Covariance Matrix**
: For a vector of fitted parameters $\hat{\boldsymbol{\theta}}$, the matrix $C_{ij} = \text{Cov}(\hat\theta_i, \hat\theta_j)$; in WLS this is $(A^T W A)^{-1}$. The diagonal entries are the variances of each parameter estimate. See [](#sec:lse_unc).

**Standardised Residual**
: A residual divided by its estimated standard deviation, allowing outliers to be identified on a common scale. See [](#sec:lse_res).

---

(sec:vocab_bay)=
## Bayesian Inference

**Prior Distribution ($\pi(\theta)$)**
: Encodes knowledge about $\theta$ *before* observing any data. See [](#sec:bay_priors).

**Posterior Distribution ($\pi(\theta|\mathbf{x})$)**
: The updated probability distribution for $\theta$ *after* observing data $\mathbf{x}$, via Bayes' theorem: $\pi(\theta|\mathbf{x}) \propto \mathcal{L}(\theta;\mathbf{x})\,\pi(\theta)$. See [](#sec:bay_covidpost).

**Likelihood (Bayesian usage)**
: The same function as in maximum likelihood — $P(\mathbf{x}|\theta)$ — but here used as one factor in Bayes' theorem to update the prior. See [](#sec:bay_covidlike).

**Evidence / Marginal Likelihood**
: The normalising constant $P(\mathbf{x}) = \int \mathcal{L}(\theta)\,\pi(\theta)\,d\theta$; ensures the posterior integrates to 1. See [](#sec:bay_evcon).

**Maximum A Posteriori (MAP) Estimate**
: The mode of the posterior distribution — the parameter value with the highest posterior probability. See [](#sec:bay_map).

**Credible Interval**
: A Bayesian interval $[a,b]$ such that $P(a \leq \theta \leq b | \mathbf{x}) = 1-\alpha$; directly interpretable as the probability that the true parameter lies in the interval given the observed data. See [](#sec:bay_credint).

**Bayesian Updating**
: The iterative process of treating today's posterior as tomorrow's prior as new data arrive. See [](#sec:bay_bayupd).

**Predictive Posterior Distribution**
: The distribution of a future observation $\tilde{x}$ given the current data, obtained by integrating the likelihood over the posterior: $P(\tilde{x}|\mathbf{x}) = \int P(\tilde{x}|\theta)\,\pi(\theta|\mathbf{x})\,d\theta$. See [](#sec:bay_predpost).

---

(sec:vocab_mcmc)=
## MCMC

**Markov Chain**
: A sequence of random variables where the probability of each state depends only on the immediately preceding state (the Markov property). See [](#chapter:mcmc).

**Markov Property**
: The future state depends only on the present state, not on the history: $P(X_{n+1}|X_n, X_{n-1}, \ldots) = P(X_{n+1}|X_n)$. See [](#chapter:mcmc).

**Monte Carlo**
: Any method that uses random sampling to estimate a numerical quantity (e.g. an integral). See [](#chapter:mcmc).

**Markov Chain Monte Carlo (MCMC)**
: A class of algorithms that constructs a Markov chain whose stationary distribution is the target posterior, then uses samples from the chain to estimate properties of that posterior. See [](#chapter:mcmc).

**Metropolis–Hastings Algorithm**
: A widely used MCMC algorithm: proposes moves from a proposal distribution and accepts or rejects them stochastically to ensure the chain converges to the correct target distribution. See [](#chapter:mcmc).

**Stationary Distribution**
: The distribution to which a Markov chain converges after a sufficient number of steps; in MCMC this is the target posterior. See [](#chapter:mcmc).

**Burn-in (Warm-up)**
: The initial portion of an MCMC chain discarded before the chain has converged to the stationary distribution. See [](#chapter:mcmc).

**Mixing**
: The ability of an MCMC chain to explore the full support of the target distribution efficiently. Poor mixing produces highly correlated, slowly moving samples. See [](#chapter:mcmc).

**Autocorrelation Time**
: The number of steps an MCMC chain must take before successive samples are approximately independent. Used to assess mixing and determine effective sample size. See [](#chapter:mcmc).

**Trace Plot**
: A plot of the sampled parameter value against MCMC iteration number; used to assess convergence and mixing. See [](#chapter:mcmc).

---

(sec:vocab_hyp)=
## Hypothesis Testing

**Hypothesis Test**
: A formal procedure for deciding, on the basis of data, whether to reject the null hypothesis $H_0$ in favour of the alternative $H_1$. See [](#chapter:hyp).

**Test Statistic**
: A single number computed from the data that summarises the evidence against $H_0$. Its distribution under $H_0$ is known, enabling inference. See [](#chapter:hyp).

**$p$-value**
: The probability of obtaining a test statistic at least as extreme as the observed value, *assuming $H_0$ is true*. A small $p$-value is evidence against $H_0$. See [](#chapter:hyp).

**Significance Level ($\alpha$)**
: The threshold below which the $p$-value is considered sufficient to reject $H_0$; conventionally $\alpha = 0.05$ (5%). See [](#chapter:hyp).

**Type I Error (False Positive)**
: Rejecting $H_0$ when it is actually true. Its probability equals $\alpha$. See [](#chapter:hyp).

**Type II Error (False Negative)**
: Failing to reject $H_0$ when it is actually false. Its probability is denoted $\beta$. See [](#chapter:hyp).

**Power**
: The probability of correctly rejecting a false $H_0$: $1 - \beta$. A more powerful test is better at detecting true effects. See [](#chapter:hyp).

**$Z$-test**
: A hypothesis test using the standardised Normal statistic $Z = (\overline{x} - \mu_0)/(\sigma/\sqrt{N})$; appropriate when the population variance is known or $N$ is large. See [](#chapter:normt).

**$t$-test**
: A hypothesis test using the $t$-statistic; appropriate when the population variance is unknown and $N$ is small. See [](#chapter:normt).

**Chi-squared Test ($\chi^2$ test)**
: A hypothesis test assessing the compatibility of observed frequencies with expected frequencies, or of a fitted model with data. Uses the $\chi^2$ distribution. See [](#chapter:normt) and [](#chapter:lse).

**Degrees of Freedom ($\nu$)**
: The number of values in a calculation that are free to vary; in a chi-squared test, $\nu = (\text{number of bins}) - (\text{number of fitted parameters}) - 1$. See [](#chapter:normt).

---

(sec:vocab_math)=
## Mathematical Tools

**Gradient ($\nabla f$)**
: The vector of first-order partial derivatives of a scalar function; points in the direction of steepest ascent. See [](#chapter:mma).

**Jacobian ($J$)**
: The matrix of first-order partial derivatives of a vector-valued function; used in multivariate change-of-variables for PDFs. See [](#sec:mmb_multjac).

**Hessian ($H$)**
: The symmetric matrix of second-order partial derivatives of a scalar function; used to characterise local maxima/minima and to approximate the covariance of MLE estimates. See [](#sec:mma_hessian).

**Taylor Expansion**
: Approximates a function near a point using its derivatives: $f(x) \approx f(a) + f'(a)(x-a) + \tfrac{1}{2}f''(a)(x-a)^2 + \cdots$. See [](#sec:mma_taylor).

**Gamma Function ($\Gamma$)**
: The continuous generalisation of the factorial: $\Gamma(n) = (n-1)!$ for positive integers, and $\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}\,dt$ generally. See [](#sec:mmb_gamfun).

**Moment Generating Function (MGF)**
: $M_X(t) = E[e^{tX}]$; uniquely characterises a distribution and allows moments to be computed by differentiating: $E[X^n] = M_X^{(n)}(0)$. See [](#sec:mmb_mgf).

**Delta Method**
: A technique for approximating the variance of a function $g(\hat\theta)$ of an estimator: $\text{Var}[g(\hat\theta)] \approx [g'(\theta)]^2 \text{Var}[\hat\theta]$. See [](#sec:mmb_deltam).

**Change of Variables (1D)**
: If $Y = g(X)$ and $g$ is monotone, $f_Y(y) = f_X(g^{-1}(y))\,|dg^{-1}/dy|$. See [](#sec:mmb_trans1d).

**Kernel**
: The part of a PDF or PMF that depends on the random variable (and parameters), ignoring normalising constants. Identifying the kernel allows a distribution family to be recognised. See [](#chapter:gpdf).

---

(sec:vocab_ml)=
## Machine Learning Terminology

This section maps common machine learning terms to the language used in this module.

**Objective Function**
: The function passed to the optimiser to be minimised (or maximised). In this module: the negative log-likelihood (NLL) in [](#chapter:mle), and $\chi^2$ in [](#chapter:lse). When regularisation is used, the objective function is the loss function plus a penalty term.

**Loss Function**
: A function measuring how wrong the model's predictions are, used to evaluate performance during training. Common examples: mean squared error (MSE) for regression, cross-entropy for classification. In module terms: the NLL is the loss function for maximum likelihood estimation, and $\chi^2$ is the loss function for least squares. When there is no regularisation, the loss function and the objective function are the same thing.

**Cost Function**
: Often used interchangeably with loss function. Some authors reserve "cost" for the total loss summed over the full dataset and "loss" for the per-sample contribution — but this distinction is not universal and should not be relied upon.

**Feature Vector**
: The vector of input variables $\mathbf{x} = (x_1, \ldots, x_d)$ describing a single observation. In this module these are called **measurements** or, in a regression context, **covariates** or **predictors**. See [](#chapter:lse).

**Model Parameters / Weights**
: The quantities to be estimated by minimising the objective function, denoted $\boldsymbol{\theta}$ in this module. In ML these are often called **weights** (denoted $\mathbf{w}$), particularly for linear models and neural networks.

**Training**
: The process of minimising the objective function over the training dataset to find optimal model parameters. In module language: **fitting** or **parameter estimation**. See [](#chapter:mle) and [](#chapter:lse).

**Gradient Descent**
: An iterative optimisation algorithm that updates parameters by stepping in the direction of the negative gradient of the objective function:
$$\theta \leftarrow \theta - \alpha \frac{\partial \mathcal{L}}{\partial \theta}$$
where $\alpha$ is the **learning rate**. The `scipy.optimize.minimize` methods used in this module (e.g. BFGS) are more sophisticated quasi-Newton variants that use curvature information to take larger, better-directed steps. See [](#chapter:mle).

**Epoch**
: One complete pass through the entire training dataset during gradient descent. Training typically runs for many epochs until the loss function converges.

**Batch**
: The subset of training data used to compute a single gradient update. If the batch is the full dataset, the method is **batch gradient descent**; if the batch is a single sample, it is **stochastic gradient descent (SGD)**; in between is **mini-batch gradient descent**.

**Regularisation**
: A penalty added to the objective function to discourage overly complex models, e.g. $\lambda\|\boldsymbol{\theta}\|^2$ (L2 / ridge) or $\lambda\|\boldsymbol{\theta}\|_1$ (L1 / lasso). In Bayesian terms, regularisation is equivalent to imposing a prior on the parameters. See [](#chapter:bay).
