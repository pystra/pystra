import pystra as ra
import numpy as np

# Define limit state function
# - case 1: define directly
limit_state = ra.LimitState(
    lambda g, X1, X2, X3: g - X2 * (1000 * X3) ** (-1) - (X1 * (200 * X3) ** (-1)) ** 2
)


# Define limit state function
# - case 2: use predefined function
def example_limitstatefunction(g, X1, X2, X3):
    """
    example limit state function
    """
    return g - X2 * (1000 * X3) ** (-1) - (X1 * (200 * X3) ** (-1)) ** 2


limit_state = ra.LimitState(example_limitstatefunction)

stochastic_model = ra.StochasticModel()

# Define random variables
stochastic_model.add_variable(ra.Lognormal("X1", 500, 100))
stochastic_model.add_variable(ra.Normal("X2", 2000, 400))
stochastic_model.add_variable(ra.Uniform("X3", 5, 0.5))

X3 = ra.Uniform("X3", lower=4.133974596215562, upper=5.866025403784438)

X2 = ra.Normal("X2", *500 * 1.00 * np.array([1, 0.2]))

# Define constants
stochastic_model.add_variable(ra.Constant("g", 1))

# Define Correlation Matrix
stochastic_model.set_correlation(
    ra.CorrelationMatrix([[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]])
)

options = ra.FORMOptions()

# initialize analysis obejct
Analysis = ra.FORM(
    options=options,
    model=stochastic_model,
    limit_state=limit_state,
)

Analysis_result = Analysis.run()  # run analysis

# Some single results:
beta = Analysis_result.beta
failure = Analysis_result.failure_probability

print(Analysis_result.summary())

sorm = ra.SORM(
    options=ra.SORMOptions(),
    model=stochastic_model,
    limit_state=limit_state,
    form=Analysis,
)
sorm_result = sorm.run()

print(sorm_result.summary())

sorm_pf = ra.SORM(
    options=ra.SORMOptions(),
    model=stochastic_model,
    limit_state=limit_state,
    form=Analysis,
)
sorm_pf.options = ra.SORMOptions(fit="point")
sorm_pf_result = sorm_pf.run()
print(sorm_pf_result.summary())

da = ra.DistributionAnalysis(
    rng=20260913,
    options=ra.SimulationOptions(),
    model=stochastic_model,
    limit_state=limit_state,
)
da_result = da.run()

cmc = ra.CrudeMonteCarlo(
    rng=20260913,
    options=ra.SimulationOptions(),
    model=stochastic_model,
    limit_state=limit_state,
)
cmc_result = cmc.run()

ismc = ra.ImportanceSampling(
    rng=20260913,
    options=ra.SimulationOptions(),
    model=stochastic_model,
    limit_state=limit_state,
)
ismc_result = ismc.run()

beta = Analysis_result.beta
failure = Analysis_result.failure_probability

print(f"Beta is {beta}, corresponding to a failure probability of {failure}")
