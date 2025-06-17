import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import arviz as az
    import altair as alt
    import numpy as np
    import pymc as pm
    import polars as pl
    return mo, np, pl, pm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    From the [PyMC documentation](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html#a-motivating-example-linear-regression):

    We are interested in predicting outcomes $Y$ as normally-distributed observations with an expected value $\mu$ that is a linear function of two predictor variables, $X_1$ and $X_2$:

    \[
        \begin{align}
            Y  &\sim \mathcal{N}(\mu, \sigma^2) \\
            \mu &= \alpha + \beta_1 X_1 + \beta_2 X_2
        \end{align}
    \]

    where $\alpha$ is the intercept, and $\beta_i$ is the coefficient for covariate $X_i$, while $\sigma$ represents the observation error.  Since we are constructing a Bayesian model, we must assign a prior distribution to the unknown variables in the model. We choose zero-mean normal priors with variance of 100 for both regression coefficients, which corresponds to *weak* information regarding the true parameter values. We choose a half-normal distribution (normal distribution bounded at zero) as the prior for $\sigma$.

    \[
        \begin{align}
            \alpha &\sim \mathcal{N}(0, 100) \\
            \beta_i &\sim \mathcal{N}(0, 100) \\
            \sigma &\sim \lvert\mathcal{N}(0, 1){\rvert}
        \end{align}
    \]
    """
    )
    return


@app.cell
def _(np, pl):
    def generate_data(α=1, β=(1, 2.5), σ=1, size: int = 100) -> pl.DataFrame:
        RANDOM_SEED = 8927
        rng = np.random.default_rng(RANDOM_SEED)

        # Predictor variable
        X1 = np.random.randn(size)
        X2 = np.random.randn(size) * 0.2

        # Simulate outcome variable
        Y = α + β[0] * X1 + β[1] * X2

        # Add noise
        Y += rng.normal(size=size) * σ

        return pl.DataFrame({"X1": X1, "X2": X2, "Y": Y})


    data = generate_data()
    return (data,)


@app.cell
def _(data):
    (data.plot.point(x="X1", y="Y") | data.plot.point(x="X2", y="Y")).resolve_scale(x="shared")
    return


@app.cell
def _(data, pm):
    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Expected value of outcome
        mu = alpha + beta[0] * data["X1"].to_numpy() + beta[1] * data["X2"].to_numpy()

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=data["Y"].to_numpy())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
