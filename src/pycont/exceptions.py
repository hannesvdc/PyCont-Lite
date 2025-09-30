class PyContError(Exception):
    """Base exception for PyCont-Lite."""

class InputError(PyContError, ValueError):
    """Invalid user input or options."""
    pass