import numpy as np
import matplotlib.pyplot as plt

import pycont

def FoldTest():
	f = lambda x, r: np.array([r + x[0]**2])

	u0 = np.array([-5.0])
	p0 = -u0[0]**2

	ds_max = 0.01
	ds_min = 1.e-6
	ds = 0.1
	n_steps = 5000
	solver_parameters = {"tolerance": 1e-10}
	continuation_result = pycont.pseudoArclengthContinuation(f, u0, p0, ds_min, ds_max, ds, n_steps, solver_parameters=solver_parameters)

	# Print some Internal info
	print('\nNumber of Branches:', len(continuation_result.branches))
	print('Bifurcation Points:', continuation_result.bifurcation_points)

	# Plot the curves
	fig = plt.figure()
	ax = fig.gca()
	x_grid = np.linspace(-80, 8, 1001)
	y_grid = np.linspace(-9, 5, 1001)
	ax.plot(x_grid, 0.0*x_grid, 'lightgray')
	ax.plot(0.0*y_grid, y_grid, 'lightgray')
	for n in range(len(continuation_result.branches)):
		branch = continuation_result.branches[n]
		ax.plot(branch['p'], branch['u'][:,0], 'blue')
	ax.plot(p0, u0, 'go', label='SP')
	ax.set_xlabel(r'$r$')
	ax.set_ylabel(r'$u$', rotation=0)
	ax.legend()
	plt.show()	

if __name__ == '__main__':
	FoldTest()
