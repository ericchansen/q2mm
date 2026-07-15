"""Q2MM benchmark systems and cases.

``q2mm.benchmarks.systems`` is the one registry of scientific benchmark
systems (CH3F, Rh-enamide, Heck relay, ...), each owning its own molecule
loading, force-field assembly, and :class:`~q2mm.models.problem.OptimizationProblem`
construction. :class:`~q2mm.benchmarks.cases.BenchmarkCase` wraps a problem
with the dataset/publication metadata a benchmark runner needs but that
does not belong in the scientific optimization core.
"""
