# Register clock closure feedback

The rising-edge output bank now qualifies closure using both controls of the
opposite latch's input transmission gate. The earlier circuit used only its
complement. With routed clock RC, that complement could cross a logic threshold
while the other gate control was still active, permitting master/slave overlap.
Increasing the period did not remove that transition-local problem.

The implemented equations are:

```
REG_M_CLOSED = ~REG_M & REG_MB
REG_S_CLOSED = ~REG_S & REG_SB
REG_M = ~CLK & REG_S_CLOSED
REG_MB = ~REG_M
REG_S = CLK & REG_M_CLOSED
REG_SB = ~REG_S
```

The acknowledgment inverters and AND gates belong to the feedback loop. They
are not a free-running delay line. A stalled or partially closed receiving
control pair withholds permission to open the opposite latch. The master/slave
latches, SAMPLE-controlled input storage, public pins, rising capture edge,
reset behavior and default unregistered mode retain their interface contracts.
The registered bank now adds 16 transistors per bit and 34 shared control
transistors, 16 more shared devices than the earlier implementation.

`output_register` metadata lists the topology, closure signals and clock nets.
The RC transient bench uses that inventory, including all four added
acknowledgment nets. It models the source and receiving sides of CLK separately;
clock-to-Q starts at the public clock pin. Feedback senses the receiving nodes
in the existing Rpath pi model. No threshold was relaxed to obtain a passing run.

## Verification

Logic tests independently stall either receiving control and verify that the
opposite latch stays closed. Physical DRC/LVS covers the standalone bank and
registered macros with sizing, mux4 and keepers. The timing checker retains its
20%/80% gate-control overlap criterion and now also interpolates closure/opening
crossings to report the minimum non-overlap gap. A negative gap fails even when
overlap occurs entirely between saved waveform samples.

Selected 4×2 RVT/two-fin macros, 1 fF output loads, 5 ps input ramps and a
10 ns period, with routed clock/control RC:

| Circuit and condition | Result | Minimum non-overlap | Maximum CLK→valid Q |
|---|---|---:|---:|
| Earlier circuit, TT, 0.5 ps timestep | Fails | Up to 13 ps observed overlap | No passing summary |
| Both-control closure, TT, 2 ps timestep | Passes | 32.5 ps | 393.5 ps |
| Both-control closure, TT, 0.5 ps timestep | Passes | 32.5 ps | 393.2 ps |
| Both-control closure, SS, 2 ps timestep | Passes | 48.7 ps | 721.1 ps |
| Both-control closure, TT, `[0,1,2,3]`, 2 ps timestep | Passes | 32.5 ps | 485.2 ps |
| Both-control closure, FF, 0.5 ps timestep | Register passes; controller fails | 21.1 ps | 338.3 ps (register only) |

TT is 0.70 V / 25 °C; SS is 0.63 V / 125 °C. The passing TT/SS runs verify
read, sampling, shutdown, retention and canceled-read behavior as well as
register phase timing. FF uses 0.77 V / −40 °C, a 25 ps startup ramp, and
SPARSE at 0.5 ps; the full macro fails the controller SAMPLE-closure check,
although register phase handoffs pass. KLU/DC and coarse FF attempts failed
numerically and remain in the evidence directory. The added closure qualification increases output delay;
timing reports from the earlier register are not transferable.

## Report provenance and limits

Macro reports still require the exact transistor-netlist hash and ROM image.
Standalone setup/hold reports now also carry `register_cell_sha256`, identifying
the one-bit characterization cell. Attaching a register report rejects a missing
or mismatched cell hash, even if load, slew, supply and model hashes agree.
This prevents the old register's constraints from being attached to the new
macro. Standalone characterization also supports `--startup-ramp-ps` for an
explicit supply ramp before its conditioning sequence; it never skips the DC
solve or forces transient storage state.

The public `Q[i]` pin rename and this topology change both invalidate older
macro reports. Regenerated reports and the compiler attachment are under
`build/brom_estimates`. Earlier evidence is retained in
`build/brom_estimates_before_bracketed_pins_98bc6d1` and
`build/brom_estimates_before_clock_closure_98bc6d1`. Direct RC comparisons,
including failed numerical attempts, are under `build/brom_clock_closure`.

The pi model still uses a common receiving node for each net. It does not
resolve skew between individual latch gates and acknowledgment gates. Passing
these selected runs does not qualify arbitrary width, duty cycle, load, PVT or
extracted interconnect. Setup/hold and physical timing qualification flags remain
false. Receiver-specific extraction still requires qualification. The historical
controller failures below motivated the subsequent
[local SAMPLE receiver fix](brom_sample_closure.md), which passes the selected
FF 4×2 and TT 4×4 cases with the updated circuit.

Before the local SAMPLE receiver fix, a 4×4 TT stress case had 143.1 ps of register non-overlap but failed the
controller's SAMPLE-closure check. Repeating that case with the pre-fix
`98bc6d1` circuit also fails the same controller check. It is an independent
qualification limit, not evidence that the new register phases overlap.

The standalone sweep now prints stimulus times with enough precision to retain
half-picosecond boundaries late in its 173.6 ns deck. The prior six-significant-
digit formatting could turn a nominal 5 ps ramp into a 4 or 6 ps ramp there.
A regression checks those late timestamps. Sweeps were regenerated with the
corrected stimuli; failed and earlier runs remain archived.

Current standalone TT/SS sweeps pass after regeneration (TT/1 fF/5 ps uses
KLU). The precise-stimulus FF standalone sweep still aborts numerically near a
late input edge, including the finer 0.25 ps retry, and publishes no constraints.
This is distinct from the historical FF full-macro RC result above: its register
phases passed while its controller closure failed. The local SAMPLE receiver
fix has separate full-macro evidence. Failed sweeps are kept explicitly
rather than replaced with old-topology constraints.
