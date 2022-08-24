import openseespy.opensees as ops
import opsvis as opsv
import pystra as ra
import matplotlib.pyplot as plt
import numpy as np


def single_run(E=30e3, P=25.0, w=0.1, x=0.0):
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    ops.node(1, x, 0)
    ops.node(2, 0, 144)
    ops.node(3, 240, 144)
    ops.node(4, 240, 0)

    ops.fix(1, 1, 1, 1)
    ops.fix(4, 1, 1, 1)

    Ag = 25.0
    Ig = 1500.0
    Ac = 29.0
    Ic = 2000.0

    gsecTag = 1
    ops.section("Elastic", gsecTag, E, Ag, Ig)

    csecTag = 2
    ops.section("Elastic", csecTag, E, Ac, Ic)

    transfTag = 1
    ops.geomTransf("Linear", transfTag)

    N = 3

    gbiTag = 1
    ops.beamIntegration("Lobatto", gbiTag, gsecTag, N)
    cbiTag = 2
    ops.beamIntegration("Lobatto", cbiTag, csecTag, N)

    leftColTag = 1
    ops.element("forceBeamColumn", leftColTag, 1, 2, transfTag, cbiTag)
    girderTag = 2
    ops.element("forceBeamColumn", girderTag, 2, 3, transfTag, gbiTag)
    rightColTag = 3
    ops.element("forceBeamColumn", rightColTag, 3, 4, transfTag, cbiTag)

    tsTag = 1
    ops.timeSeries("Constant", tsTag)

    patternTag = 1
    ops.pattern("Plain", patternTag, tsTag)

    ops.load(2, P, 0, 0)
    ops.eleLoad("-ele", girderTag, "-type", "beamUniform", -w)

    # define these to avoid warnings
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-6, 6, 2)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1)
    ops.analysis("Static")
    ops.analyze(1)

    return 0.15 - ops.nodeDisp(2, 1)


def plot_ops_model_results():
    opsv.plot_model()
    opsv.plot_loads_2d(sfac=5)
    opsv.plot_defo()
    sfacN, sfacV, sfacM = 2, 2, 5.0e-2
    opsv.section_force_diagram_2d("N", sfacN)
    plt.title("Axial force distribution")
    opsv.section_force_diagram_2d("T", sfacV)
    plt.title("Shear force distribution")
    opsv.section_force_diagram_2d("M", sfacM)
    plt.title("Bending moment distribution")
    plt.show()


def lsf(E, P, w, x):
    n = len(E)
    g = np.zeros((n, 1))

    for i in range(n):
        g[i] = single_run(E[i], P[i], w[i], x[i])

    return g.T


limit_state = ra.LimitState(lsf)
options = ra.AnalysisOptions()
options.setPrintOutput(True)
stochastic_model = ra.StochasticModel()
stochastic_model.addVariable(ra.Lognormal("E", 30e3, 3e3))
stochastic_model.addVariable(ra.Normal("P", 25, 5))
stochastic_model.addVariable(ra.Normal("x", 0, 1))
stochastic_model.addVariable(ra.Uniform("w", 0.1, 0.02))
form = ra.Form(
    analysis_options=options,
    stochastic_model=stochastic_model,
    limit_state=limit_state,
)
form.run()
form.showDetailedOutput()
