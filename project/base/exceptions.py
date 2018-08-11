class FEniCSConvergenceException(RuntimeError):
    """Raised when the Newton Solver part of FEniCS 
    fails to converge.
    """
    def __init__(self, msg=None):
        if msg is None:
            msg = 'FEniCS failed to converge.'
        super(FEniCSConvergenceException, self).__init__(msg)


class TerminatedBatchData(RuntimeError):
    """Raised when an instance of BatchData has been
    terminated and is then called again.
    """
    def __init__(self, msg=None):
        if msg is None:
            msg = 'BatchData has already been terminated.'
        super(TerminatedBatchData, self).__init__(msg)
