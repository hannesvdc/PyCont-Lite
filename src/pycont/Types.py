import numpy as np

from dataclasses import dataclass, field
from typing import List, Literal, Dict, Optional

EventKind = Literal["SP", "LP", "BP", "DSFLOOR", "MAXSTEPS"]

@dataclass
class Event:
	kind: EventKind
	u: np.ndarray
	p: float
	info: Dict = field(default_factory=dict)

@dataclass
class Branch:
	id: int
	from_event: Optional[int]
	termination_event: Event
	u_path: np.ndarray
	p_path: np.ndarray
	stable: Optional[bool]
	info: Dict = field(default_factory=dict)

@dataclass
class ContinuationResult:
    branches: List[Branch] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
	
def makeBranch(id, termination_event, u_path, p_path):
	"""
	Internal function to create a Branch dataclass instance from the
	current continuation data.
	"""
	return Branch(id, None, termination_event, u_path, p_path, None)