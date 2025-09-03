import numpy as np

import pycont

def FoldTest():
	G = lambda x, r: r + x**2

	u0 = np.array([-5.0])
	p0 = -u0[0]**2

	ds_max = 0.01
	ds_min = 1.e-6
	ds = 0.1
	n_steps = 5000
	solver_parameters = {"tolerance": 1e-10}
	continuation_result = pycont.pseudoArclengthContinuation(G, u0, p0, ds_min, ds_max, ds, n_steps, solver_parameters=solver_parameters)

	# Print some Internal info
	print('\nNumber of Branches:', len(continuation_result.branches))

	# Plot the curves
	pycont.plotBifurcationDiagram(continuation_result)

if __name__ == '__main__':
	FoldTest()