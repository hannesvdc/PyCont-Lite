import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

from .Types import ContinuationResult

def plotBifurcationDiagram(cr : ContinuationResult, **kwargs) -> None:
    
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

    # Plot the branches
    for branch in cr.branches:
        u_vals = np.apply_along_axis(u_transform, 1, branch.u_path)
        linestyle = '--' if not branch.stable else '-'
        plt.plot(branch.p_path, u_vals, color='tab:blue', linestyle=linestyle)
		
    # Plot all interesting points
    style = {"SP" : 'go', "LP" : 'bo', "BP" : 'ro'}
    for event in cr.events:
        if event.kind in style.keys():
            plt.plot(event.p, u_transform(event.u), style[event.kind], label=event.kind)

    plt.grid(visible=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()	
