import pystra as pr
from scipy.stats import genextreme as gev


def lsf(X1, X2, C):
    return X1 - X2 - C


X2 = pr.ScipyDist("X2", gev(c=0.1, loc=200, scale=50))
X2.plot()

limit_state = pr.LimitState(lsf)

model = pr.StochasticModel()
model.add_variable(pr.Normal("X1", 500, 100))
model.add_variable(X2)
model.add_variable(pr.Constant("C", 50))

options = pr.FORMOptions()

form = pr.FORM(model=model, limit_state=limit_state, options=options)
form_result = form.run()
print(form_result.summary())

sorm = pr.SORM(model=model, limit_state=limit_state, form=form)
sorm_result = sorm.run()
print(sorm_result.summary())
