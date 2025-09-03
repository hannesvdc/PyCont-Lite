import matplotlib.pyplot as plt

from .Types import ContinuationResult

def plotBifurcationDiagram(cr : ContinuationResult) -> None:

    # Plot the branches
    for branch in cr.branches:
        linestyle = '-' if branch.stable else '--'
        plt.plot(branch.p_path, branch.u_path[:,0], color='tab:blue', linestyle=linestyle)
		
    # Plot all interesting points
    for event in cr.events:
        if event.kind == "SP":
            plt.plot(event.p, event.u, 'go', label=event.kind)
        elif event.kind == "LP":
            plt.plot(event.p, event.u, 'bo', label=event.kind)
        elif event.kind == "BP":
            plt.plot(event.p, event.u, 'ro', label=event.kind)

    plt.grid(visible=True)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$u$', rotation=0)
    plt.legend()
    plt.show()	
