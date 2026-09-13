import numpy as np
import pandas as pd
import pystra as pr


def lsf(R, S):
    return R - S


limit_state = pr.LimitState(lsf)

model = pr.StochasticModel()
model.add_variable(pr.Lognormal("R", 5, 5))
model.add_variable(pr.Lognormal("S", 1, 1))
model.set_correlation(pr.CorrelationMatrix([[1.0, 0.5], [0.5, 1.0]]))

options = pr.FORMOptions()

form = pr.FORM(
    model=model,
    limit_state=limit_state,
    options=options,
)
form_result = form.run()
print(form_result.summary())

sa = pr.SensitivityAnalysis(
    model=model,
    limit_state=limit_state,
    options=options,
)
cf = pr.SensitivityAnalysis(
    model, limit_state, options=options, method="closed_form"
).run()

fd = sa.run()

ref = {
    "R": {"mean": 0.5184, "std": -0.2548},
    "S": {"mean": -1.3629, "std": 0.0445},
}

rows = []
for var in ["R", "S"]:
    for p in ["mean", "std"]:
        sym = "\u03bc" if p == "mean" else "\u03c3"
        rows.append(
            {
                "Parameter": f"\u2202\u03b2/\u2202{sym}_{var}",
                "Reference": ref[var][p],
                "CF": cf.marginal[var][p],
                "FD": fd.marginal[var][p],
            }
        )

df = pd.DataFrame(rows).set_index("Parameter")
df.style.format("{:+.4f}")

pd.DataFrame(
    [
        {
            "Parameter": "\u2202\u03b2/\u2202\u03c1",
            "Reference": 2.4585,
            "CF": cf.correlation[1, 0],
        }
    ]
).set_index("Parameter").style.format("{:+.4f}")

rows = []
for delta in [0.1, 0.01, 0.001, 0.0001]:
    fd_trial = pr.SensitivityAnalysis(
        model, limit_state, options=options, delta=delta
    ).run()
    rows.append(
        {
            "Method": f"FD (\u03b4={delta})",
            "\u2202\u03b2/\u2202\u03c3_S": fd_trial.marginal["S"]["std"],
        }
    )
rows.append({"Method": "CF", "\u2202\u03b2/\u2202\u03c3_S": cf.marginal["S"]["std"]})
rows.append({"Method": "Reference", "\u2202\u03b2/\u2202\u03c3_S": 0.0445})

pd.DataFrame(rows).set_index("Method").style.format("{:+.4f}")


def lsf_calrel(X1, X2, X3):
    return 1 - X2 / (1000 * X3) - (X1 / (200 * X3)) ** 2


ls1 = pr.LimitState(lsf_calrel)

model1 = pr.StochasticModel()
model1.add_variable(pr.Lognormal("X1", 500, 100))
model1.add_variable(pr.Lognormal("X2", 2000, 400))
model1.add_variable(pr.Uniform("X3", 5, 0.5))

R1 = np.array(
    [
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.2],
        [0.2, 0.2, 1.0],
    ]
)
model1.set_correlation(pr.CorrelationMatrix(R1))

opts1 = pr.FORMOptions()

form1 = pr.FORM(
    model=model1,
    limit_state=ls1,
    options=opts1,
)
form1_result = form1.run()
print(form1_result.summary())

sa1 = pr.SensitivityAnalysis(
    model=model1,
    limit_state=ls1,
    options=opts1,
)
cf1 = pr.SensitivityAnalysis(model1, ls1, options=opts1, method="closed_form").run()

# Reference values from Bourinet (2017), Table 2
ref1 = {
    "X1": {"mean": -0.0059, "std": -0.0079},
    "X2": {"mean": -0.0009, "std": -0.0006},
    "X3": {"mean": 1.2602, "std": -1.1942},
}

rows = []
for var in ["X1", "X2", "X3"]:
    for p in ["mean", "std"]:
        sym = "\u03bc" if p == "mean" else "\u03c3"
        rows.append(
            {
                "Parameter": f"\u2202\u03b2/\u2202{sym}_{var}",
                "Reference": ref1[var][p],
                "CF": cf1.marginal[var][p],
            }
        )

pd.DataFrame(rows).set_index("Parameter").style.format("{:+.4f}")

# Reference values from Bourinet (2017), Table 3
ref_corr = {
    (1, 0): -0.5151,
    (2, 0): 0.8916,
    (2, 1): 0.4688,
}

names1 = ["X1", "X2", "X3"]
rows = []
for (i, j), r in ref_corr.items():
    rows.append(
        {
            "Parameter": f"\u2202\u03b2/\u2202\u03c1({names1[i]},{names1[j]})",
            "Reference": r,
            "CF": cf1.correlation[i, j],
        }
    )

pd.DataFrame(rows).set_index("Parameter").style.format("{:+.4f}")
