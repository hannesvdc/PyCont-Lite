import numpy as np
import matplotlib.pyplot as plt

import pycont

def TransCriticalTest():
	G = lambda x, r: np.array([r*x[0] - x[0]**2])

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

	fig = plt.figure()
	ax = fig.gca()
	x_grid = np.linspace(-10, 10, 1001)
	y_grid = np.linspace(-7.5, 7.5, 1001)
	ax.plot(x_grid, 0.0*x_grid, 'lightgray')
	ax.plot(0.0*y_grid, y_grid, 'lightgray')
	for n in range(len(continuation_result.branches)):
		branch = continuation_result.branches[n]
		ax.plot(branch.p_path, branch.u_path[:,0], 'blue')
	for event in continuation_result.events:
		if event.kind == "SP":
			ax.plot(event.p, event.u, 'go', label=event.kind)
		elif event.kind == "BP":
			ax.plot(event.p, event.u, 'ro', label=event.kind)
	ax.set_xlabel(r'$r$')
	ax.set_ylabel(r'$u$')
	ax.legend(loc='upper left')
	plt.show()	

if __name__ == '__main__':
	TransCriticalTest()