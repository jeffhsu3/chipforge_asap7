# Sized BROM control distribution

`--sized-controls` inserts load-sized, noninverting tapered buffer chains after
macro construction, including any controller and replica. It works with the
external-phase, `--clocked` and `--self-timed` interfaces. The option is
experimental and opt-in; existing compilation retains its original drivers.

```bash
python -m chipforge_asap7.brom words.json --bits 16 \
  --self-timed --sized-controls --out build/brom_distribution/compiled --drc --lvs
```

The Python API is `size_control_distribution(circuit, DistributionSpec())`.
It modifies and returns the circuit; call it once after all peripheral logic is
present. The macro name includes `_dist4` for the default maximum stage effort.

## Load model and implementation

The pass sums the fin counts of every transistor gate connected to each net.
A unit inverter's input load is the sum of its n- and p-device fins. By default,
a driver may see at most four times its input-fin equivalent. This is a gate
fanout heuristic: it does not assume a calibrated capacitance per fin, balance
n/p mobility, or include wire RC, output diffusion or Miller capacitance.

Eligible internally driven nets have one complementary inverter output and
only gate loads. This includes high-fanout phase controls, latch complements,
block selects, shared decoder signals and candidate outputs. Storage nodes,
bitlines and latch data paths with diffusion connections are excluded.
Heavily loaded external PRECHN/EVAL/SAMPLE inputs instead receive a local
buffer chain; the external pins remain inputs. External input sizing assumes
an upstream source with one base inverter's drive strength.

The final stage uses enough parallel base inverters to meet the fanout target.
Earlier stages taper down toward the original driver. An even number of
inverting stages preserves polarity. The original logic gate remains present
and drives the first stage; downstream consumers use the final stage. The
replica participates in the same load calculation, including its fully loaded
candidate and duplicated phase loads. Array transistors and programming cuts
are unchanged.

Each parallel copy uses the existing single-finger peripheral device layout
and gets its own two-transistor placement group. This avoids introducing
unqualified wider device geometry or exceeding the minimum folded-row capacity.
The manifest's `control_distribution` object records original loads and driver
strengths, each stage's multiplicity and nets, and all added device names.

Drivers are appended to the folded peripheral bank. This is a tapered-driver
implementation using the existing metal spines, not a sink-clustered repeater
tree. Additional buffer placement means the replica observers are no longer
necessarily at the end of the bank. Receiver-specific feedback placement and
wire-aware branch sizing remain work for extracted-RC qualification.

## Verification and limits

Tests check polarity, stage fanout bounds through multi-stage tapers, external
pin direction, unchanged array devices, parameter validation, exact pin
inventory, routed connectivity, DRC and LVS. Physical cases cover one, two and
three fins and mux1/mux4. The transient bench accepts `--sized-controls`:

```bash
PYTHONPATH=src python -m chipforge_asap7.brom.replica_timing \
  --depth 8 --bits 16 --sized-controls --klu \
  --osdi build/brom_timing/model/BSIMCMG107.osdi \
  --out build/brom_distribution/tt_mux1
```

This bench checks real data validity before sampling, latch closure before
evaluation ends, actual candidate deactivation before precharge, read retention
and reset cancellation. Adding buffers changes propagation delays and therefore
requires renewed characterization. It does not establish a higher maximum clock
frequency. Both macro and distribution timing remain unqualified until wire RC,
receiver feedback and the supported configuration/corner range are characterized.

## Recorded validation

Two-fin RVT, 1 fF output loads, ideal wires, 0.5 ps maximum timestep:

| Sized macro / corner | Read margin | Latch closure margin | Candidate-off margin |
|---|---:|---:|---:|
| 8×16 mux1, TT / 0.70 V / 25 °C | 68.0 ps | 50.0 ps | 70.1 ps |
| 16×4 mux4, TT / 0.70 V / 25 °C | 66.8 ps | 50.0 ps | 71.6 ps |
| 4×2 mux1, SS / 0.63 V / 125 °C | 82.9 ps | 69.9 ps | 83.8 ps |

All scheduled reads, hold checks and final reset cancellations passed. These
margins describe the sampling schedule, not improvements in access time.
The wider 8×16 SS KLU run failed at the initial operating point; its sparse
retry was stopped before completing the transient. That shape/corner remains
unverified. FF has not been requalified for sized distribution.

The full 8×16 mux1 CLI build also passes DRC, LVS, pin matching and route
connectivity, alongside all four standalone peripheral artifacts. Reports,
decks, waveforms and physical artifacts are under `build/brom_distribution`.
For that pattern, PRECHN uses parallel-inverter multiplicities [2, 6], SAMPLE
uses [2, 5], and the replica candidate uses [1, 3]. A heavily populated data
candidate is also buffered; its sizing depends on the ROM's programmed pattern.

The combined BROM, controller, replica, distribution, timing, minimum-proof
and packaging regression passed 144 tests. The added sized-bench stimulus
check also passed, and the final sizing-metadata tests passed after reporting
fields were added.

The next [wire-RC-aware sizing and placement pass](brom_wire_rc.md) is available
with `--wire-rc-controls`. It uses routed-length estimates and reports residual
load violations; it does not imply extracted timing closure.
