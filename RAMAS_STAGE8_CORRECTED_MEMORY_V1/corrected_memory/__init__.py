"""Corrected episodic-memory experiment for RAMAS.

This package changes two things relative to the frozen Stage 6.4 memory design
and nothing else:

1. CREDIT LABEL.  The stored per-episode outcome becomes the realized advantage
   of the taken action over the unchanged numerical core, evaluated on the arm's
   OWN pre-trade holdings.  The frozen design compared a full-authority shadow
   portfolio against the archived legacy RAMoE portfolio, which follows a
   different holdings path; that comparison is dominated by the exposure gap
   between the two paths rather than by decision quality.

2. RETRIEVAL.  Episodes are retrieved k-per-action instead of k-nearest overall,
   so the evidence block can express a contrast between candidate actions.

Everything else - inputs, clock, router stream, numerical core, risk projection,
accounting, journal, provider and validation - is imported unchanged from the
verified Stage 6.4 package.  This package never writes to it.
"""

__all__ = ["labels", "retrieval", "engine", "replay"]
