from fmtorch.path.affine_path import AffinePath
from fmtorch.scheduler import CondOTScheduler

class CondOTProbPath(AffinePath):
    r"""The ``CondOTProbPath`` class represents a conditional optimal transport probability path.

    This class is a specialized version of the ``AffineProbPath`` that uses a conditional optimal transport scheduler to determine the parameters of the affine transformation.

    The parameters :math:`\alpha_t` and :math:`\sigma_t` for the conditional optimal transport path are defined as:

    .. math::

        \alpha_t = t \quad \text{and} \quad \sigma_t = 1 - t.
    """

    def __init__(self):
        self.scheduler = CondOTScheduler()