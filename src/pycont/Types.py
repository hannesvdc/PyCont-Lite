import numpy as np

from dataclasses import dataclass, field
from typing import List, Literal, Dict, Optional, Tuple

EventKind = Literal["SP", "LP", "BP", "HB", "DSFLOOR", "MAXSTEPS", "PARAM_MIN", "PARAM_MAX"]

@dataclass
class Event:
	kind: EventKind
	u: np.ndarray
	p: float
	s: float
	info: Dict = field(default_factory=dict)

@dataclass
class Branch:
	id: int
	from_event: Optional[int]
	termination_event: Optional[Event]
	u_path: np.ndarray
	p_path: np.ndarray
	s_path: np.ndarray
	stable: Optional[bool]
	info: Optional[Dict] = field(default_factory=dict)

	_tentativePoints : List[Tuple[np.ndarray, float]] = field(default_factory=list, init=False, repr=False)

	def __init__(self, id : int, n_steps : int, u0 : np.ndarray, p0 : float):
		M = len(u0)
		self.id = id
		self.u_path = np.zeros((n_steps+1, M)); self.u_path[0,:] = u0
		self.p_path = np.zeros(n_steps+1); self.p_path[0] = p0
		self.s_path = np.zeros(n_steps+1)
		self._index = 1
		self._tentativePoints = []

	def addPoint(self, x : np.ndarray, s : float):
		self.u_path[self._index, :] = x[0:-1]
		self.p_path[self._index] = x[-1]
		self.s_path[self._index] = s
		self._index += 1

	def addPointTentative(self, x : np.ndarray, s : float):
		self._tentativePoints.append((np.copy(x), s))

	def commit(self):
		for point in self._tentativePoints:
			self.addPoint(point[0], point[1])
		self._tentativePoints.clear()
		return self

	def trim(self):
		self.u_path = self.u_path[0:self._index,:]
		self.p_path = self.p_path[0:self._index]
		self.s_path = self.s_path[0:self._index]
		return self

@dataclass
class ContinuationResult:
    branches: List[Branch] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)