import numpy as np

import pycont

def TransCriticalTest():
	G = lambda x, r: r*x - x**2
	
	ds_max = 0.01
	ds_min = 1.e-6
	ds0 = 0.001

	n_steps = 2000
	u0 = np.array([-5.0])
	p0 = -5.0
	solver_parameters = {"tolerance": 1e-10}
	continuation_result = pycont.pseudoArclengthContinuation(G, u0, p0, ds_min, ds_max, ds0, n_steps, solver_parameters=solver_parameters)

	# Print some Internal info
	print('\nNumber of Branches:', len(continuation_result.branches))

	# plot the curves
	pycont.plotBifurcationDiagram(continuation_result)

if __name__ == '__main__':
	TransCriticalTest()