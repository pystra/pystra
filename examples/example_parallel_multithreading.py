"""Evaluate a small FORM batch with one thread per physical point.

Limit-state callbacks receive one-dimensional arrays, one per variable. Threads
can help when each point waits on an external solver; this short algebraic
example demonstrates ordering rather than a speed improvement.
"""

from queue import Queue
from threading import Thread
from time import perf_counter

import numpy as np
import pystra as ra


def example_limit_state(X1, X2, X3):
    """Return limit-state values in the same order as the input points."""
    output = Queue()

    def evaluate(index, x1, x2, x3):
        value = 1 - x2 / (1000 * x3) - (x1 / (200 * x3)) ** 2
        output.put((index, value))

    threads = [
        Thread(target=evaluate, args=(index, *point))
        for index, point in enumerate(zip(X1, X2, X3, strict=True))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ordered = sorted(output.get() for _ in threads)
    return np.array([value for _, value in ordered])


def main():
    """Run FORM for three correlated marginals and return its result."""
    model = ra.StochasticModel()
    model.add_variable(ra.Lognormal("X1", 500, 100))
    model.add_variable(ra.Normal("X2", 2000, 400))
    model.add_variable(ra.Uniform("X3", 5, 0.5))
    model.set_correlation([[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]])
    return ra.FORM(model, ra.LimitState(example_limit_state)).run()


if __name__ == "__main__":
    started = perf_counter()
    result = main()
    print(f"Done in {perf_counter() - started:.3f} seconds")
    print(result.design_point_x)
