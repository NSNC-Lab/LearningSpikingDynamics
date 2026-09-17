# July 1 local-eligibility runtime backup

Created on 2026-07-15 before restoring the later pre-`dPSC` local-eligibility implementation.

This directory contains a path-preserving copy of the 16 active runtime/configuration files from Git commit `b5da538af6741d1695c2a34ca51b117ec7f8148f`. Every archived file was SHA-256 compared with the active file before replacement and all copies matched.

This is the simplest implementation: STRF gain is applied after convolution and there are no `dPSC_*`, `dV_*`, or `local_pre_derivs` sensitivity paths. The same source remains recoverable directly from Git commit `b5da538`.
