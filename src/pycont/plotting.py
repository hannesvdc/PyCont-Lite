import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .Types import ContinuationResult

def plotBifurcationDiagram(cr : ContinuationResult, **kwargs) -> None:
    if len(cr.branches) == 0:
        print('No active branches to plot. Returning')
        return
    M = cr.branches[0].u_path.shape[1]
    
    if "p_label" in kwargs:
        xlabel = kwargs["p_label"]
    else:
        xlabel = r'$p$'

    if "u_label" in kwargs:
        ylabel = kwargs["u_label"]
    else:
        ylabel = r'$u$'

    if "u_transform" in kwargs:
        u_transform = kwargs["u_transform"]
    else:
        M = cr.branches[0].u_path.shape[1]
        if M == 1:
            u_transform = lambda u : u[0]
            ylabel = r'$u$'
        else:
            u_transform = lambda u : lg.norm(u)
            ylabel = r'$||u||$'

    # Plot the regular branches first
    bf_fig, bf_ax = plt.subplots()
    for branch in cr.branches:
        if branch.is_lc:
            # Plot u(t)[0] versus u(t)[1]
            npoints = branch.u_path.shape[0]
            L = (branch.u_path.shape[1]-1) // M
            Q_points = np.reshape(branch.u_path[:,:-1], (npoints, M, L), 'F')
            p_values = branch.p_path
            branch.from_event

            if branch.from_event is not None:
                hopf_location = cr.events[int(branch.from_event)].p
            else:
                hopf_location = None
            _plotLimitCycleFamily(Q_points, p_values, hopf_location)

        else:
            u_vals = np.apply_along_axis(u_transform, 1, branch.u_path)
            linestyle = '--' if not branch.stable else '-'
            bf_ax.plot(branch.p_path, u_vals, color='tab:blue', linestyle=linestyle)
		
    # Plot all interesting points
    style = {"SP" : 'go', "LP" : 'bo', "BP" : 'ro', "HB": 'mo', "DSFLOOR": 'co'}
    for event in cr.events:
        if event.kind in style.keys():
            bf_ax.plot(event.p, u_transform(event.u), style[event.kind], label=event.kind)
    handles, labels = bf_ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    bf_ax.legend(by_label.values(), by_label.keys(), loc='best')

    bf_ax.grid(visible=True)
    bf_ax.set_xlabel(xlabel)
    bf_ax.set_ylabel(ylabel)

    plt.show()	

def _plotLimitCycleFamily(Q_points, p_values, p_hopf):
    u0_t = Q_points[:,0,:]
    u1_t = Q_points[:,1,:]

    fig, ax = plt.subplots()
    norm = Normalize(p_values.min(), p_values.max())
    cmap = 'plasma'
    cm = plt.get_cmap(cmap)

    stride = 10
    for k in range(0, u0_t.shape[0], stride):
        x = u0_t[k]; y = u1_t[k]

        # close the loop for plotting aesthetics
        x_plot = np.r_[x, x[0]]; y_plot = np.r_[y, y[0]]
        color = cm(norm(p_values[k]))
        ax.plot(x_plot, y_plot, color=color)

    # Add a colorbar
    sm = ScalarMappable(norm=norm, cmap=cm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'$p$', rotation=0, labelpad=12, va='top')

    ax.set_xlabel(r'$u_0(t)$')
    ax.set_ylabel(r'$u_1(t)$')
    if p_hopf is not None:
        ax.set_title(rf'Limit Cycles from Hopf Point $p = {round(p_hopf,4)}$')