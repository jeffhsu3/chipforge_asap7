# NOR-ROM bitcell: `chipforge_asap7.devices.rom_bitcell`

A via-programmed NOR-ROM tile whose wordline is the gate poly itself. With two
fins a bit is 81 x 81 nm.

```python
from chipforge_asap7.devices import RomBitcellSpec, build_rom_array, build_rom_bitcell

spec = RomBitcellSpec(fins=2)       # cell "rom_bitcell_f2"
spec.width, spec.height             # 162, 81: two words by one bit
spec.site(1)                        # (103.0, 27): word 1's programming V1
tile = build_rom_bitcell(spec)      # unprogrammed; V1s are added by the array
array = build_rom_array(spec, words=8, bits=6, programmed={(0, 0), (5, 3)})
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

## Status

`build_rom_array` closes the array's edges: dummy gates, ACTIVE overhangs and
select. gdscheck reports only ACTIVE.LUP.1 on it, for 1 to 3 fins with no bits,
every bit or a checkerboard programmed (`tests/test_rom_bitcell.py`).
ACTIVE.LUP.1 fires because the array has no body tap yet.

Still to do, in the periphery:

- Wordline straps: LIG, V0 and M1 up to M3 every N rows.
- Tie the `ISO` gates to VSS.
- Add a body tap.
- A transposed decoder: words are columns here, and bits are rows.
- LVS against `RomBitcellSpec.netlist`.
