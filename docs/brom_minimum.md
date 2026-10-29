# OR-Tools minimum-area proof for a periodic BROM selector core

`chipforge_asap7.verification.brom_minimum` now provides a CP-SAT lower bound
and a separate physical witness for a **specified periodic BROM topology**.
The witness generator is `chipforge_asap7.brom.selector`. The result covers
the periodic array core, not a complete compiled macro.

The proved period is **108 × 432 nm per selector**, storing four logical
bits: **46,656 nm² per selector, or 0.011664 µm² per logical bit**. The result
is the same for one, two and three fins because routing sets the block pitch.
This area does not include taps, array edges, supply closure or periphery.

The topology fixes one live selector and one grounded isolation transistor
per column, with continuous diffusion across columns. An M1 source column
is grounded; each drain reaches its vertical M3 bitline. Each block has a
separate M3 gate stem and gate cuts at its boundaries. Sixteen full-width
candidate conductors occupy M2 and M4. Programming adds exactly one V2 or
V3 to the selected candidate, leaving all base geometry unchanged.

The lower-bound model uses necessary rules, not a predetermined tile size:

- Selector-to-isolation and isolation-to-next-selector gate spacing constrain
  the X period, including the wraparound boundary.
- Fin width, fin spacing, ACTIVE enclosure and separation constrain the Y
  period. Identical blocks preserve the 27 nm global fin lattice.
- M2 candidates need at least 36 nm center spacing; M4 uses 48 nm tracks
  with the pinned deck's 12 nm offset. Spacing includes the next block.
- If any candidate uses M4, the identical block period must preserve both
  the fin and M4 lattices. The model also permits all-M2 routing, so it does
  not assume that a mixed-layer solution wins.
- CP-SAT minimizes the product of width and height on a 0.5 nm coordinate
  grid. A separate feasibility query excludes every strictly smaller area.
  The report demonstrates that the finite 2,048 nm variable domain contains
  every possible smaller-area candidate using independent FEOL bounds.

A second solve adds the actual M2 access obstacles and M3 stem-end spacing.
It attains the lower bound with **eight M2 and eight M4 candidates**. The
runner then measures the generated polygons and binds those measurements
back into a fresh model. A solver solution alone is insufficient.

Each fin count's finite witness contains two blocks and sixteen columns,
covering all sixteen programming choices in each block, with reversed
column order in the second block. Qualification requires:

1. Full gdscheck `main` DRC against the explicitly selected, hash-pinned
   ASAP7 YAML source tree.
2. Unit-fin transistor LVS, top-level pin inventory and pin-name checks.
3. Independent polygon connectivity extraction checking all 32 selectors,
   32 grounded isolation devices, fin counts and all 49 distinct pins.
4. 256 switch-level output observations across every block-enable
   combination, word offset and column.
5. Negative controls: removing gate wraparound, fin-period alignment or
   M4-period alignment must permit a smaller solution; deleting one actual
   programming via must fail both the connectivity audit and the combined
   LVS/pin-inventory check.

The pin-inventory check is essential: KLayout can match a floating gate to
an inferred schematic input even after its physical programming via is
removed. The runner records that raw result and independently rejects the
missing connection; a raw `matched` result alone cannot qualify this proof.

Reproduce with OR-Tools, gdstk, gdspy, KLayout and the pinned gdscheck checkout:

```bash
uv run --with ortools==9.15.6755 --with gdstk python \
  -m chipforge_asap7.verification.brom_minimum \
  --gdscheck-root ../gdscheck --fins 1 2 3 \
  --out build/brom_minimum/qualified
```

Every `fN/` directory retains the CP-SAT model text, solver responses and
logs, rule/source/tool hashes, `plan.json`, witness GDS and LVS reference,
DRC/LVS reports, negative-control artifacts and `report.json`. The runner
replaces any prior success status before starting and fails closed on stale
rule sources, solver timeouts, unattained bounds or failed physical checks.
The observed solver version is recorded; solver statuses are trusted, not
independently checked proof certificates.

The generated plan can also program a different array:

```python
import json
from pathlib import Path
from chipforge_asap7.brom.selector import SelectorPlan, build_selector_array

plan = SelectorPlan(**json.loads(Path(
    "build/brom_minimum/qualified/f2/plan.json"
).read_text()))
mapping = (tuple(range(16)), tuple(reversed(range(16))))
array = build_selector_array(plan, mapping)
```

The claim excludes alternative diffusion-sharing topologies, paired-column
layouts, additional candidate-routing layers, jogged/multiplexed candidates,
and multi-block supercells with different track phases. Those alternatives
could be smaller. It also excludes timing, leakage, analog margins and PVT
qualification. This is not a global minimum across all BROM architectures,
and the finite witnesses do not certify every generated array size/image.
