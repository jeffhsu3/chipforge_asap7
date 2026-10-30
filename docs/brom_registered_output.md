# Rising-edge registered BROM output

Set `BromSpec(..., registered_output=True)` with `clocked_macro_circuit()` or
`self_timed_macro_circuit()`, or add `--registered-output` to either CLI mode:

```bash
python -m chipforge_asap7.brom words.json --bits 2 --self-timed \
  --registered-output --out build/registered_brom --drc --lvs
```

The default remains the SAMPLE-controlled output latch. The registered mode
retains that latch on internal `SAMPLED<i>` nets and adds a static master/slave
register driving each public `Q<i>`. The master opens during CLK low; the slave
opens during CLK high. Shared interlocked complementary enables prevent both
input transmission gates from opening together. The added cost is 16 transistors
per output bit plus 18 shared clock-control transistors. Placement uses groups
of at most eight devices. The CLI emits a standalone `output_register` artifact
and records the edge, internal connections and qualification status in the
manifest. External phase-only macros reject this option because they lack CLK.

## Interface contract

Address launched after rising edge t0 is accepted during the high precharge
phase, held for the low evaluation phase, sampled when the replica declares
readiness, and published on Q after rising edge t1. Q then holds until the next
rising capture edge. The low phase must finish sampling before the register's
setup/hold aperture; this is a fixed-cycle interface with no ready/stall output.
Clock-to-Q is finite. The interlock does not eliminate setup/hold requirements
or permit arbitrary short clock pulses.

This matches the synchronous, one-read-stage contract expected by LUnit's
`clocked=True, registered_output=True` interface. An adapter using that contract
should not add another ROM output register. The chipforge backend registration,
pin mapping and reset wiring are separate integration work; they are not added
by this compiler option.

RESETN cancels controller evaluation but does not reset or gate the output
register. Before a first valid capture, Q is unspecified. Cancellation before
sampling preserves the previous sampled word; cancellation after sampling can
still publish the new sampled word at the next rising edge, even with RESETN
low. Reset release and precharge follow the [controller contract](brom_controller.md).

## Validation and limits

Switch-level tests exercise rising capture, falling edges, and data changes
while CLK is high and low. Integrated physical tests cover mux1 and mux4,
one through three fins, and an optional keeper; they check DRC, LVS, exact pin
inventory and connectivity. Synthetic waveform faults verify rejection of
phase overlap, invalid data at capture, and Q glitches between capture edges.

Full self-timed transistor simulations use RVT/two-fin devices, 1 fF output
loads, 5 ps clock ramps, a 1 ns period and 0.5 ps maximum timestep. The checker
also verifies the underlying sample-latch/controller protocol, actual master
closure, sampled-data stability through closure, and canceled-read retention.

| Case | Maximum CLK-to-valid Q | Minimum sampled-data lead before CLK edge |
|---|---:|---:|
| 4×2 mux1, TT / 0.70 V / 25 °C | 48.5 ps | 357.5 ps |
| 4×2 mux1, SS / 0.63 V / 125 °C, one-fin keeper | 69.4 ps | 272.5 ps |
| 16×2 mux4, TT / 0.70 V / 25 °C | 55.4 ps | 309.8 ps |

The TT mux1 case also passes with address changes 10 ps after the rising ramp
starts. Results and reproducible decks are under `build/brom_register/`.
The FF / 0.77 V / −40 °C attempt failed numerical convergence and provides no
fast-corner timing qualification.

These are selected ideal-wire transient measurements, not extracted timing
signoff or measured setup/hold constraints. Metadata explicitly records
`setup_hold_qualified: false`. PVT/setup/hold sweeps, routed clock skew and RC,
width/depth/load coverage, and sized-control combinations still need
qualification. The measured data lead is available slack in this bench, not
a required setup time or a guaranteed maximum operating frequency.
