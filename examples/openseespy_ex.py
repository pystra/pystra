"""Reliability of a portal frame analyzed with OpenSeesPy.

The executed version, with a check by direct simulation, is the
docs/source/notebooks/ex_openseespy.ipynb tutorial.
"""

import numpy as np
import openseespy.opensees as ops

import pystra as ra


def build_frame(E, P, w, x):
    """Create the portal frame in OpenSees' global model (kip, inch)."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, x, 0.0)  # left base, misplaced by x
    ops.node(2, 0.0, 144.0)
    ops.node(3, 240.0, 144.0)
    ops.node(4, 240.0, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.fix(4, 1, 1, 1)
    ops.section("Elastic", 1, E, 25.0, 1500.0)  # girder
    ops.section("Elastic", 2, E, 29.0, 2000.0)  # columns
    ops.geomTransf("Linear", 1)
    ops.beamIntegration("Lobatto", 1, 1, 3)
    ops.beamIntegration("Lobatto", 2, 2, 3)
    ops.element("forceBeamColumn", 1, 1, 2, 1, 2)  # left column
    ops.element("forceBeamColumn", 2, 2, 3, 1, 1)  # girder
    ops.element("forceBeamColumn", 3, 3, 4, 1, 2)  # right column
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, P, 0.0, 0.0)
    ops.eleLoad("-ele", 2, "-type", "beamUniform", -w)


def frame_sway(E, P, w, x):
    """Lateral displacement at the top of the left column (in)."""
    build_frame(E, P, w, x)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-6, 6)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        raise RuntimeError("OpenSees analysis failed")
    return ops.nodeDisp(2, 1)


def sway_limit_state(E, P, w, x):
    return np.array([0.15 - frame_sway(*point) for point in np.broadcast(E, P, w, x)])


model = ra.StochasticModel()
model.add_variable(ra.Lognormal("E", 30e3, 3e3))
model.add_variable(ra.Normal("P", 25.0, 5.0))
model.add_variable(ra.Normal("w", 0.1, 0.02))
model.add_variable(ra.Normal("x", 0.0, 1.0))
limit_state = ra.LimitState(sway_limit_state)

form = ra.FORM(model, limit_state)
print(form.run().summary())
print(ra.SORM(model, limit_state, form=form).run().summary())
