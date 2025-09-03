import numpy as np
import matplotlib.pyplot as plt

import pycont

def TransCriticalTest():
	G = lambda x, r: r*x - x**2
	
	ds_max = 0.001
	ds_min = 1.e-6
	ds0 = 0.1

	n_steps = 10000
	u0 = np.array([-5.0])
	p0 = -5.0
	solver_parameters = {"tolerance": 1e-10}
	continuation_result = pycont.pseudoArclengthContinuation(G, u0, p0, ds_min, ds_max, ds0, n_steps, solver_parameters=solver_parameters)

	# Print some Internal info
	print('\nNumber of Branches:', len(continuation_result.branches))

	x_grid = np.linspace(-10, 10, 1001)
	y_grid = np.linspace(-7.5, 7.5, 1001)
	plt.plot(x_grid, 0.0*x_grid, 'lightgray')
	plt.plot(0.0*y_grid, y_grid, 'lightgray')
	for branch in continuation_result.branches:
		linestyle = '-' if branch.stable else '--'
		plt.plot(branch.p_path, branch.u_path[:,0], color='blue', linestyle=linestyle)
	for event in continuation_result.events:
		if event.kind == "SP":
			plt.plot(event.p, event.u, 'go', label=event.kind)
		elif event.kind == "LP":
			plt.plot(event.p, event.u, 'bo', label=event.kind)
		elif event.kind == "BP":
			plt.plot(event.p, event.u, 'ro', label=event.kind)
	plt.xlabel(r'$r$')
	plt.ylabel(r'$u$')
	plt.legend(loc='upper left')
	plt.show()	

if __name__ == '__main__':
	TransCriticalTest()