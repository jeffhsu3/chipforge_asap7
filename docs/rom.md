# NOR-ROM bitcell: `chipforge_asap7.devices.rom_bitcell`

A via-programmed NOR-ROM tile whose wordline is the gate poly itself. With two
fins a bit is 81 x 81 nm.

```python
from chipforge_asap7.devices import (
    RomArraySpec, RomBitcellSpec, build_rom_array, build_rom_bitcell,
)

cell = RomBitcellSpec(fins=2)       # cell "rom_bitcell_f2"
cell.width, cell.height             # 162, 81: two words by one bit
tile = build_rom_bitcell(cell)      # unprogrammed; V1s are added by the array

spec = RomArraySpec(words=8, bits=6, bitcell=cell, strap_every=4)
spec.row_y(4), spec.site(5, 3)      # 540, (427.0, 378): rows step over straps
spec.wordline_x(5)                  # 423: WL<5>'s M3 strap
array = build_rom_array(spec, programmed={(0, 0), (5, 3)})
```

## Why the wordline is the poly

chipforge's `scripts/rom_compiler.py` gives each bitcell its own gate contact
up to a horizontal M2 wordline, on an 81 nm row. ASAP7 can't fit that contact
into any row near 81 nm:

- The LIG may not sit on the channel (LIG.GATE.S.10), so it goes above the
  diffusion.
- It must stay 14 nm from the SDT (LIG.SDT.S.8), and SDT covers the whole
  ACTIVE height (SDT.W.2, SDT.ACTIVE.OV.1).
- The gate cut above it needs 5 nm of clearance (LIG.GCUT.S.11) and 17 nm of
  width (GCUT.W.1).

That adds up to `27 * fins + 57` nm per row: 108 nm for one fin, 135 nm for
two. That compiler's cell also put the source and drain 36 nm from the gate on
a 54 nm gate pitch. That lays the LISD bars over the next gate stripe, so gdscheck
reports about 13 violations per bit on it.

Here the array is turned on its side. Gate poly runs the full height of the
array and is the wordline. Each fin row is one bit, and its bitline runs along
the row on M2:

```text
x:   0    27    54    81   108   135   162
     D    WLa   S    WLb    D    ISO    D ...
     [== one ACTIVE strip per row, never broken ==]   BL<b> (M2)
               VSS (M1 column)
```

- **Source:** the two word transistors share their source. A continuous M1
  column carries it to VSS.
- **Drains:** each word transistor has its own drain, with a 28 x 18 nm M1
  stub along the row (M1.A.1).
- **Programming:** one V1 on the stub ties the drain to the bitline. A clear
  bit leaves the drain floating. Either way it's a one-mask edit, as in
  chipforge's ROM, but on the drain side.
- **Isolation:** `ISO` separates one tile's last drain from the next tile's
  first. Both are the same bitline, so its leakage runs from a net back to
  itself.
- **Diffusion:** it is never broken. A break between different nets costs
  92 nm (ACTIVE.S.2A).
- **Row height:** each row is `fins + 1` fin pitches. The empty pitch is the
  27 nm that ACTIVE.S.1 requires between rows.

| fins | bit (nm) | area (nm²) |
| ---: | --- | ---: |
| 1 | 81 x 54 | 4374 |
| 2 | 81 x 81 | 6561 |
| 3 | 81 x 108 | 8748 |

## Strap row

The poly wordline is strapped to M3 by a 108 nm strap row. The array starts
with one, then adds another under every `strap_every` bit rows. Every 162 nm
tile of it contacts all three gates:

- **Wordlines:** `WLa` and `WLb` each get an LIG reaching from the gate to a
  V0 at its far end. An M1 bar, V1, an M2 landing and V2 then take it to the
  word's M3 wordline, at x = 9 and 99 in the tile.
- **Isolation tie:** `ISO`'s LIG sits on a lower level, with an M1 bar down to
  an M2 VSS track across the row. A V1 joins that track to the VSS column.

The two LIG levels are 31 nm apart (LIG.S.4). On one level, the ISO LIG and
the wordline LIGs on either side would face each other across 16 nm.

The row would be 81 nm if only two-fin tiles used it. It is 108 nm because a
one-fin tile's drain stubs come within 32 nm of its bottom edge, and each M1
bar has to keep 31 nm from them.
At `strap_every=16` the row adds 8% to a two-fin array's height.

```text
y (strap row)
 55   WLa LIG-V0-M1-V1-M2-V2 -> M3        WLb ... -> M3
  9   ======================= M2 VSS track ========================
  8                                                ISO LIG-V0-M1-V1 -^
x:    9    27        54        81    99         135
```

## Status

`build_rom_array` builds the strap rows and closes the array's edges: dummy
gates, ACTIVE overhangs and select. gdscheck reports only ACTIVE.LUP.1 on it,
for 1 to 3 fins with no bits, every bit or a checkerboard programmed.
ACTIVE.LUP.1 fires because the array has no body tap yet.

`test_straps_wire_every_gate` extracts the nets and checks that:

- each word transistor's gate is on its `WL<w>`;
- every source and every `ISO` gate is on `VSS`;
- a drain is on `BL<b>` exactly when its bit is programmed.

It needs gdstk, which isn't a project dependency, and skips without it.

Still to do, in the periphery:

- Add a body tap.
- A transposed decoder: words are columns here, and bits are rows.
- KLayout LVS against `RomBitcellSpec.netlist`.
