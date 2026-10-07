"""ASAP7 design rules the device generators draw against.

Each number here is one the public KLayout runset (``drc_ASAP7.lydrc`` from
ASAP7_for_KLayout) checks, tagged with the rule it comes from, so a generator
imports the rule instead of restating it.  Placement grids live in
`chipforge_asap7.layout.grid`; choices a cell makes on top of the rules -- the
released cells' ACTIVE overhangs, LIG strap heights, how far a wire runs past
its via -- stay with the generator that makes them.

Several generators sit exactly on these limits with no slack.  `test_finfet`
asserts the drawn margins against them, so a geometry edit fails immediately
instead of waiting for the next machine that has KLayout installed.

All values are in nanometres.
"""

from __future__ import annotations

__all__ = [
    "CONTACT_SIZE",
    "GATE_CUT_MIN_SPACE",
    "M1_MIN_SPACE",
    "M1_V0_ENCLOSURE",
    "M1_WIDTH",
    "M2_V1_ENCLOSURE",
    "M4_PITCH",
    "M4_WIDTH",
    "M4_X_GRID",
    "SHORT_M1_EDGE",
    "TRACK_PITCH",
    "V0_LISD_ENCLOSURE",
    "V3_M3_ENCLOSURE",
    "V3_M4_ENCLOSURE",
]

# ── Front end ─────────────────────────────────────────────────────────────────
GATE_CUT_MIN_SPACE = 35  # GCUT.S.3
V0_LISD_ENCLOSURE = 3  # V0.LISD.EN.2, on at least two opposite sides

# ── Vias, M1-M3 ───────────────────────────────────────────────────────────────
CONTACT_SIZE = 18  # V0.W.1 through V3.W.1: every via up to V3 is 18 nm wide
# V0.M1.AUX.3 requires M1 to be exactly as wide as the V0 landing on it,
# measured across the M1 track, so these two are equal by rule and not by
# coincidence.  Metal shapes are still drawn from M1_WIDTH and vias from
# CONTACT_SIZE, so neither one silently resizes the other.
M1_WIDTH = CONTACT_SIZE
M1_MIN_SPACE = 18  # M1.S.1, both edges > 36 nm
#: M1.S.2: an edge under 36 nm needs 25 nm to its neighbour where a long one
#: needs 18.
SHORT_M1_EDGE = 36
#: M1.S.2 (M2 and M3 have M1's rules): a line end, an edge of 36 nm or less,
#: facing a longer edge -- a via pad's tip toward a rail.
M1_TIP_TO_SIDE = 25
M1_V0_ENCLOSURE = 5  # V0.M1.EN.1: M1 end-cap past V0 along the track
# M2.W.1/M3.W.1 and M2.S.1/M3.S.1 match M1's, so one pitch serves all three.
TRACK_PITCH = M1_WIDTH + M1_MIN_SPACE  # 36
#: M2 end-cap around a V1.  V1.M2.EN.2 asks for 5 nm, but the runset builds
#: its 5 nm ring with ``.sized(-2.5.nm).sized(2.5.nm)``, which erases a strip
#: that is exactly 5 nm wide -- the neighbouring V1.M1.EN.1 keeps its 1 dbu of
#: slack and this one does not.  Landing 8 nm clears the rule as written and as
#: implemented; the released decoder inverter draws exactly 5 and trips it.
M2_V1_ENCLOSURE = 8

# ── V3, M4 ────────────────────────────────────────────────────────────────────
M4_WIDTH = 24  # M4.W.1 (and M4.W.3 forbids even multiples of it)
M4_PITCH = M4_WIDTH + 24  # M4.S.1
M4_X_GRID = 24  # M4.AUX.1: M4 vertices sit on a 24 nm grid in x
V3_M3_ENCLOSURE = 5  # V3.M3.EN.1, on at least two opposite sides
V3_M4_ENCLOSURE = 11  # V3.M4.EN.2, on at least two opposite sides
