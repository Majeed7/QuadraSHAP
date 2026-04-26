try:
    from quadrashap._core import (
        PreparedTreeData,
        PreparedTreeQT,
        explain_trees,
        explain_trees_quadrature,
    )

    HAS_CPP_EXT = True
except ImportError:
    HAS_CPP_EXT = False
    explain_trees = None
    PreparedTreeData = None
    PreparedTreeQT = None
    explain_trees_quadrature = None
