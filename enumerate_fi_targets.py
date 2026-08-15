#!/usr/bin/env python3
"""
Computes N for an exhaustive SFI campaign against fi_target_table.csv
(the RangeID-based table from build_fi_target_table.py) — the full
target cartesian product — and reports totals before you commit any
hardware time. No GDB/board connection needed; pure bookkeeping.

TWO FAULT MODELS, TWO DIFFERENT FORMULAS — pick with --fault-model:

  transient (default, matches inject_transient_fault.py):
    n_TOT = sum(layer) sum(register) sum(address-in-range)
            sum(bit, 0..31) sum(input)
    No StuckValue dimension at all — pure bit toggle has no target
    value, so including one would double-count a dimension that doesn't
    exist for this tool.

  permanent (matches inject_permanent_fault.py):
    n_TOT = sum(layer) sum(register) sum(address-in-range)
            sum(bit, 0..31) sum(stuck_value, 0/1) sum(input)
    Stuck-at-0 and stuck-at-1 ARE genuinely different possible defects
    on the same bit, so this dimension is real here.

Using the wrong --fault-model doesn't just miscount N — it also breaks
--state-file subtraction, since transient's injection_state.csv has no
StuckValue column at all (reading row["StuckValue"] against it would
crash), while permanent's does.

If a --state-file (injection_state.csv / injection_state_permanent.csv)
is given, already-completed trials are subtracted, so you get REMAINING
N for picking up an in-progress campaign, not just the theoretical total.

Usage:
    # Transient campaign sizing (default)
    python3 enumerate_fi_targets.py fi_target_table.csv --num-inputs 20
    python3 enumerate_fi_targets.py fi_target_table.csv --state-file injection_state.csv

    # Permanent campaign sizing
    python3 enumerate_fi_targets.py fi_target_table.csv --fault-model permanent \
        --state-file injection_state_permanent.csv

    python3 enumerate_fi_targets.py fi_target_table.csv --layer conv2d_1  # just one layer

Output:
    exhaustive_target_list.csv — one row per remaining trial
    Console summary — total n_TOT, remaining N, breakdown by layer/register,
    and an estimated wall-clock time given a per-trial duration
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Set, Tuple


def z_score(confidence: float) -> float:
    """
    Critical value t (commonly written z) such that
    P(-t < Z < t) = confidence under the standard normal distribution —
    i.e. t = Phi^-1(1 - alpha/2). Enter --confidence as a percentage
    (e.g. 95 for 95%); this converts and looks up the corresponding
    critical value exactly the way you'd read it off a normal
    distribution table, just computed exactly instead of interpolated.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    alpha = 1 - confidence
    return NormalDist().inv_cdf(1 - alpha / 2)


def compute_sample_size(N: int, confidence: float, margin: float, p: float) -> int:
    """
    Standard finite-population sample-size formula:
        n = N / (1 + e^2 * (N-1) / (t^2 * p * (1-p)))
    N = population size, t = critical value from z_score(), e = margin of
    error, p = assumed proportion (0.5 = max-variance, no prior estimate).
    Unlike the per-32-bit-register case this formula was originally built
    for (where it's a no-op — n comes back equal to N), N here is the
    WHOLE campaign's exhaustive trial count, often in the millions, so
    this should produce a genuinely much smaller n.
    """
    if N <= 0:
        return 0
    t = z_score(confidence)
    denom = 1 + (margin ** 2) * (N - 1) / (t ** 2 * p * (1 - p))
    n = N / denom
    return max(1, min(N, math.ceil(n)))


def load_populations(path: Path, include_untouched: bool, layer_filter: Optional[str]) -> List[Dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if layer_filter:
        rows = [r for r in rows if r["Layer"] == layer_filter]
    if not include_untouched:
        before = len(rows)
        rows = [r for r in rows if r.get("DominantUsage", "") != "UNTOUCHED"]
        dropped = before - len(rows)
        if dropped:
            print(f"[i] Skipped {dropped} UNTOUCHED populations (never referenced by any "
                  f"instruction — use --include-untouched to keep them)")
    return rows


def addresses_for(pop: Dict) -> List[str]:
    """Every REAL address this register was hit at within this range —
    not just an envelope min/max, since fi_target_table.csv now stores
    the full PCAddresses list."""
    addrs = [a for a in pop.get("PCAddresses", "").split(";") if a]
    return addrs if addrs else [pop.get("PCAddrMin", "")]


def load_completed_fingerprints(state_path: Optional[Path], fault_model: str) -> Set[Tuple]:
    """
    Reads the ACTUAL columns each tool's state file has — transient's
    schema has no StuckValue column at all; permanent's does. Reading the
    wrong shape crashes (KeyError), which is exactly the bug this rewrite
    fixes: the old version always assumed StuckValue existed.
    """
    if state_path is None or not state_path.exists():
        return set()
    completed = set()
    with open(state_path, newline="") as f:
        for row in csv.DictReader(f):
            if fault_model == "permanent":
                completed.add((row["Layer"], int(row["RangeID"]), row["Register"],
                               row["Address"], int(row["Bit"]), int(row["StuckValue"])))
            else:
                completed.add((row["Layer"], int(row["RangeID"]), row["Register"],
                               row["Address"], int(row["Bit"])))
    return completed


def enumerate_trials(populations: List[Dict], input_ids: List[str], fault_model: str,
                      stuck_values: List[int], completed: Set[Tuple]) -> Tuple[List[Dict], int]:
    trials = []
    theoretical_total = 0
    for pop in populations:
        addrs = addresses_for(pop)
        for addr in addrs:
            for bit in range(32):
                # Transient: no stuck_value dimension at all — iterate
                # once per (layer, register, address, bit, input), not
                # twice, since a pure toggle has no target value.
                stuck_iter = stuck_values if fault_model == "permanent" else [None]
                for stuck in stuck_iter:
                    for input_id in input_ids:
                        theoretical_total += 1
                        if fault_model == "permanent":
                            fp = (pop["Layer"], int(pop["RangeID"]), pop["Register"], addr, bit, stuck)
                        else:
                            fp = (pop["Layer"], int(pop["RangeID"]), pop["Register"], addr, bit)
                        if fp in completed:
                            continue
                        row = {
                            "Layer": pop["Layer"],
                            "Register": pop["Register"],
                            "StaticClass": pop.get("StaticClass", ""),
                            "RangeID": pop["RangeID"],
                            "Address": addr,
                            "Bit": bit,
                            "InputID": input_id,
                        }
                        if fault_model == "permanent":
                            row["StuckValue"] = stuck
                        trials.append(row)
    return trials, theoretical_total


def summarize(trials: List[Dict], theoretical_total: int, seconds_per_trial: float, fault_model: str,
              confidence_pct: float, margin: float, p: float):
    remaining = len(trials)
    already_done = theoretical_total - remaining
    by_layer = defaultdict(int)
    by_register = defaultdict(int)
    for t in trials:
        by_layer[t["Layer"]] += 1
        by_register[t["Register"]] += 1

    print(f"\n=== Exhaustive campaign size ({fault_model}) ===")
    print(f"Theoretical N (n_TOT): {theoretical_total}")
    if already_done:
        print(f"Already completed:     {already_done}")
    print(f"Remaining N:            {remaining}")

    print(f"\nRemaining, by layer:")
    for layer, count in sorted(by_layer.items()):
        print(f"  {layer:<20} {count:>8}")

    print(f"\nRemaining, by register (summed across all layers):")
    for reg, count in sorted(by_register.items(), key=lambda kv: -kv[1]):
        print(f"  {reg:<10} {count:>8}")

    confidence_frac = confidence_pct / 100.0
    t_value = z_score(confidence_frac)
    sample_theoretical = compute_sample_size(theoretical_total, confidence_frac, margin, p)
    sample_remaining = compute_sample_size(remaining, confidence_frac, margin, p)

    print(f"\n=== Statistical sample size (confidence={confidence_pct:.1f}%, "
          f"margin={margin * 100:.2f}%, p={p}) ===")
    print(f"Critical value t (z-score) for {confidence_pct:.1f}% confidence: {t_value:.4f}")
    print(f"Sample size out of theoretical N: {sample_theoretical:,}  "
          f"({100 * (1 - sample_theoretical / theoretical_total):.1f}% fewer trials than exhaustive)"
          if theoretical_total else "")
    print(f"Sample size out of remaining N:   {sample_remaining:,}")
    sample_seconds = sample_theoretical * seconds_per_trial
    print(f"Estimated wall-clock for the SAMPLE at {seconds_per_trial}s/trial: "
          f"{sample_seconds:,.0f}s (~{sample_seconds / 3600:,.1f}h)")

    total_seconds = remaining * seconds_per_trial
    hours = total_seconds / 3600
    print(f"\nEstimated wall-clock for EXHAUSTIVE remaining N at {seconds_per_trial}s/trial: "
          f"{total_seconds:,.0f}s (~{hours:,.1f}h)")
    print("Adjust --seconds-per-trial to match your actual reset+continue+capture round-trip time.")
    if fault_model == "permanent":
        print("NOTE: permanent-fault trials are typically much slower per-trial than transient "
              "(watchpoint re-enforcement overhead) — measure your actual average and pass it "
              "via --seconds-per-trial rather than trusting the 3.0s default here.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path, help="fi_target_table.csv from build_fi_target_table.py")
    parser.add_argument("--fault-model", choices=["transient", "permanent"], default="transient",
                         help="Which tool you're sizing for — transient (inject_transient_fault.py, "
                              "no StuckValue dimension) or permanent (inject_permanent_fault.py, "
                              "StuckValue 0/1 dimension included). Default: transient.")
    parser.add_argument("--layer", type=str, default=None, help="Restrict to one layer")
    parser.add_argument("--include-untouched", action="store_true",
                         help="Include registers with zero usage in the range (default: skip)")
    parser.add_argument("--num-inputs", type=int, default=None,
                         help="Generate placeholder input IDs input_0..input_{N-1}")
    parser.add_argument("--inputs", type=str, default=None,
                         help="Comma-separated real input IDs (overrides --num-inputs)")
    parser.add_argument("--stuck-values", type=str, default="0,1",
                         help="[permanent only] Comma-separated stuck-at values to test (default: both 0 and 1)")
    parser.add_argument("--state-file", type=Path, default=None,
                         help="injection_state.csv (transient) or injection_state_permanent.csv "
                              "(permanent) — if given, already-completed trials are subtracted. "
                              "MUST match --fault-model or column reads will fail.")
    parser.add_argument("--seconds-per-trial", type=float, default=3.0,
                         help="Estimated wall-clock seconds per trial, for the time estimate (default: 3.0)")
    parser.add_argument("--p", type=float, default=0.5,
                         help="Assumed proportion for the sample-size formula — 0.5 is the "
                              "max-variance assumption used when you have no prior estimate "
                              "(default: 0.5)")
    parser.add_argument("--margin", type=float, default=0.005,
                         help="Margin of error for the sample-size formula, e.g. 0.005 = 0.5%% "
                              "(default: 0.005)")
    parser.add_argument("--confidence", type=float, default=95.0,
                         help="Confidence level as a PERCENTAGE (typical range 90-99) — this is "
                              "looked up against the normal distribution to get the critical "
                              "value t used in the sample-size formula (default: 95.0)")
    parser.add_argument("--output", type=Path, default=Path("exhaustive_target_list.csv"))
    args = parser.parse_args()

    if not (0 < args.confidence < 100):
        parser.error(f"--confidence must be between 0 and 100 (got {args.confidence})")
    if not (90 <= args.confidence <= 99):
        print(f"[!] --confidence={args.confidence} is outside the typical 90-99 range — "
              f"still valid, just double-check that's intentional.")

    if args.inputs:
        input_ids = [s.strip() for s in args.inputs.split(",") if s.strip()]
    elif args.num_inputs:
        input_ids = [f"input_{i}" for i in range(args.num_inputs)]
    else:
        input_ids = ["input_0"]
        print("[!] No --inputs or --num-inputs given, defaulting to a single placeholder input. "
              "IMPORTANT: neither inject_transient_fault.py nor inject_permanent_fault.py "
              "actually vary the test input yet (no feed_input() wired up) — every real trial "
              "today uses whatever single input CAMPAIGN_END's inference loop processes. So the "
              "N you can ACTUALLY run right now is the num_inputs=1 number below, regardless of "
              "what --num-inputs you pass here — that flag is for sizing a future campaign once "
              "input-switching exists, not today's achievable N.")

    stuck_values = [int(s) for s in args.stuck_values.split(",")] if args.fault_model == "permanent" else []

    populations = load_populations(args.table, args.include_untouched, args.layer)
    completed = load_completed_fingerprints(args.state_file, args.fault_model)
    if completed:
        print(f"[i] Loaded {len(completed)} completed trial(s) from {args.state_file} "
              f"({args.fault_model} schema) — subtracting from totals")

    trials, theoretical_total = enumerate_trials(populations, input_ids, args.fault_model,
                                                   stuck_values, completed)

    fields = ["Layer", "Register", "StaticClass", "RangeID", "Address", "Bit", "InputID"]
    if args.fault_model == "permanent":
        fields.append("StuckValue")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trials)
    print(f"Saved remaining trial worklist to: {args.output}")

    summarize(trials, theoretical_total, args.seconds_per_trial, args.fault_model,
              args.confidence, args.margin, args.p)


if __name__ == "__main__":
    main()
