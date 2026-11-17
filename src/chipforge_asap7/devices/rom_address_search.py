"""Search the NOR-ROM decoder for its slowest address step.

The read bench (:mod:`~chipforge_asap7.devices.rom_read_spice`) reads one
step, ``previous_word -> word``, through a fixed far bitline and slowest
wordline row.  What the step changes is only the decode, from the address
edges to WL: which word's predecoder tree rises, which bits switch, and so
which lines fall beside it.  So the search screens steps with the decoder
alone (`render_decoder_deck`, about 5 s a step instead of 80) and then reads
its winners through the full bench.

1. **Words.**  Every word, from its complement: every bit switching, so the
   word's whole tree rises against the previous word's falling.
2. **Steps.**  For the slowest words, a hill climb over the bits the step
   flips (``previous = word ^ mask``): starting from all of them, toggle one
   bit at a time and keep the slowest, until no single toggle is slower.
3. **Reads.**  The slowest steps, and word 0 from its complement for
   reference, through `run_rom_read`.

Each screen is one simulation: the search finds a slow step, not a proof
that no slower one exists.  The hill climb can stop in a local maximum.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .rom_read_spice import (
    RomReadConditions,
    RomReadResult,
    decoder_spec,
    render_decoder_deck,
    run_rom_read,
)
from .spice import (
    compile_bsimcmg107,
    find_default_bsimcmg107_source,
    find_default_model_card,
    translate_asap7_model,
)
from .sram6t_read_spice import _measure, _simulate

__all__ = ["AddressSearch", "AddressStep", "search_address_steps"]


@dataclass(frozen=True)
class AddressStep:
    """One screened step; times in seconds from the first address edge."""

    word: int
    previous: int
    t_decoded_s: float | None
    t_pa_s: float | None
    t_pb_s: float | None

    @property
    def flips(self) -> int:
        return (self.word ^ self.previous).bit_count()


@dataclass(frozen=True)
class AddressSearch:
    words: tuple[AddressStep, ...]
    climbs: tuple[tuple[AddressStep, ...], ...]
    worst: tuple[AddressStep, ...]
    reads: tuple[tuple[AddressStep, RomReadResult], ...]


def _key(step: AddressStep) -> float:
    return step.t_decoded_s if step.t_decoded_s is not None else -1.0


def search_address_steps(
    *,
    model_card: Path,
    osdi: Path,
    output_dir: Path,
    fins: int = 1,
    words: int = 256,
    conditions: RomReadConditions | None = None,
    top_words: int = 4,
    confirm: int = 2,
    workers: int = 3,
    ngspice: str = "ngspice",
    screen_timeout: float = 300.0,
    log=print,
) -> AddressSearch:
    """Screen every word, climb the slowest, read the winners; see the module."""
    c = conditions or RomReadConditions()
    output_dir = output_dir.resolve()
    screens = output_dir / "screens"
    screens.mkdir(parents=True, exist_ok=True)
    models = screens / f"asap7_{c.vt}_bsimcmg107.pm"
    card = model_card.read_text()
    models.write_text(
        "\n".join(translate_asap7_model(card, f"{t}_{c.vt}") for t in ("nmos", "pmos"))
    )
    spec = decoder_spec(fins, words, c)
    full = (1 << spec.address_bits) - 1
    cache: dict[tuple[int, int], AddressStep] = {}

    def screen(step: tuple[int, int]) -> AddressStep:
        if step not in cache:
            word, previous = step
            deck = render_decoder_deck(
                fins,
                words,
                models=models,
                osdi=osdi,
                conditions=replace(c, word=word, previous_word=previous),
            )
            path = screens / f"w{word}_p{previous}.sp"
            done = path.with_suffix(".log")
            try:
                # A run cut short resumes: a screen whose deck is unchanged
                # and whose log finished reads its log instead of re-running.
                if (
                    path.is_file()
                    and done.is_file()
                    and path.read_text() == deck
                    and "t_decoded" in (old := done.read_text())
                ):
                    text = old
                else:
                    # A screen takes seconds; one that crawls (a step the
                    # solver struggles with) is recorded, not waited on.
                    text = _simulate(path, deck, ngspice, timeout=screen_timeout)
            except RuntimeError as exc:
                # One unsolvable step must not end the search: record it.
                log(f"step {previous} -> {word} failed: {exc}")
                text = ""
            cache[step] = AddressStep(
                word,
                previous,
                _measure(text, "t_decoded"),
                _measure(text, "t_pa"),
                _measure(text, "t_pb"),
            )
        return cache[step]

    def batch(steps: list[tuple[int, int]]) -> list[AddressStep]:
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(screen, steps))

    # 1. Every word from its complement.
    stage1 = batch([(w, w ^ full) for w in range(full + 1)])
    ranked = sorted(stage1, key=_key, reverse=True)
    log(
        f"words: slowest {[(s.word, round(_key(s) * 1e12, 1)) for s in ranked[:top_words]]} ps"
    )

    # 2. Hill-climb the flipped bits of the slowest words.
    climbs = []
    for start in ranked[:top_words]:
        best, path = start, [start]
        while True:
            mask = best.word ^ best.previous
            options = [
                (best.word, best.word ^ (mask ^ (1 << b)))
                for b in range(spec.address_bits)
                if mask ^ (1 << b)  # a step must change at least one bit
            ]
            slowest = max(batch(options), key=_key)
            if _key(slowest) <= _key(best):
                break
            best = slowest
            path.append(best)
        climbs.append(tuple(path))
        log(
            f"word {start.word}: {round(_key(start) * 1e12, 1)} -> "
            f"{round(_key(best) * 1e12, 1)} ps from {best.previous} "
            f"({best.flips} bits)"
        )
    worst = sorted({path[-1] for path in climbs}, key=_key, reverse=True)

    # 3. Read the winners, and the reference step, through the full bench.
    reference = cache.get((0, full)) or screen((0, full))
    chosen = list(dict.fromkeys([*worst[:confirm], reference]))
    reads = []
    for step in chosen:
        (result,) = run_rom_read(
            model_card=model_card,
            osdi=osdi,
            output_dir=output_dir / f"read_w{step.word}_p{step.previous}",
            fins=(fins,),
            words=(words,),
            conditions=replace(c, word=step.word, previous_word=step.previous),
            ngspice=ngspice,
        )
        reads.append((step, result))
        log(
            f"read {step.previous} -> {step.word}: sense "
            f"{round(result.t_sense_s * 1e12, 1) if result.t_sense_s else None} ps"
        )
    search = AddressSearch(tuple(stage1), tuple(climbs), tuple(worst), tuple(reads))
    (output_dir / "search.json").write_text(
        json.dumps(
            {
                "fins": fins,
                "words_per_bitline": words,
                "conditions": asdict(c),
                "words": [asdict(s) for s in stage1],
                "climbs": [[asdict(s) for s in path] for path in climbs],
                "worst": [asdict(s) for s in worst],
                "reads": [{"step": asdict(s), "result": asdict(r)} for s, r in reads],
            },
            indent=1,
        )
        + "\n"
    )
    return search


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``asap7-rom-address-search``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fins", type=int, default=1)
    parser.add_argument("--words", type=int, default=256)
    parser.add_argument("--bits", type=int, default=64)
    parser.add_argument("--temperature-c", type=float, default=25.0)
    parser.add_argument("--driver", default="default")
    parser.add_argument("--address-driver", default="BUFx4")
    parser.add_argument("--top-words", type=int, default=4)
    parser.add_argument("--confirm", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model-card", type=Path)
    parser.add_argument("--osdi", type=Path)
    parser.add_argument("--ngspice", default="ngspice")
    parser.add_argument("--out", type=Path, default=Path("build/rom_address_search"))
    args = parser.parse_args(argv)
    model_card = args.model_card or find_default_model_card()
    if model_card is None or not model_card.is_file():
        parser.error("ASAP7 model card not found; pass --model-card")
    osdi = args.osdi
    if osdi is None:
        source = find_default_bsimcmg107_source()
        if source is None:
            parser.error("BSIM-CMG 107 source not found; pass --osdi")
        osdi = compile_bsimcmg107(source, args.out.resolve() / "BSIMCMG107.osdi")
    conditions = RomReadConditions(
        bits=args.bits,
        temperature_c=args.temperature_c,
        driver=args.driver,
        address_driver=args.address_driver,
    )
    search_address_steps(
        model_card=model_card,
        osdi=osdi,
        output_dir=args.out,
        fins=args.fins,
        words=args.words,
        conditions=conditions,
        top_words=args.top_words,
        confirm=args.confirm,
        workers=args.workers,
        ngspice=args.ngspice,
        log=lambda line: print(line, flush=True),
    )
    print(f"artifacts: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
