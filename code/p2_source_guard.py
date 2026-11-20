# -*- coding: utf-8 -*-
"""Guard for the figure scripts: data-reading figures are built only from this tree's results.

The figure scripts resolve `results/` relative to their own location. The guard stops a run
whose base directory lacks the frozen manifest, which would mean the results being read are
not the ones this tree produced.
"""
from __future__ import annotations

import os


class WrongSourceTree(RuntimeError):
    """A data-reading figure was requested from a tree without this tree's results."""


def require_p2_source(base, what, script):
    """Raise unless `base` (the parent of `code/`) holds this tree's frozen manifest."""
    if os.path.isfile(os.path.join(base, "results", "frozen_split.json")):
        return
    raise WrongSourceTree("%s reads results, but %s has no results/frozen_split.json; run %s "
                          "from the repository root tree." % (what, base, script))
