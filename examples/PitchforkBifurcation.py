import numpy as np

import pycont

def PitchforkTest():
	G = lambda x, r: r*x - x**3
	u0 = np.array([-2.0])
	p0 = 4.0

	ds_max = 0.001
	ds_min = 1.e-6
	ds = 0.001
	n_steps = 10000
	solver_parameters = {"tolerance": 1e-10}
	continuation_result = pycont.pseudoArclengthContinuation(G, u0, p0, ds_min, ds_max, ds, n_steps, solver_parameters=solver_parameters)

	# Print some Internal info
	print('\nNumber of Branches:', len(continuation_result.branches))

	# Plot all branches and interesting points
	pycont.plotBifurcationDiagram(continuation_result)

if __name__ == '__main__':
	PitchforkTest()