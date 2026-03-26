import json
from pathlib import Path
from datasets import load_from_disk

root = Path("/workspace/action_abstraction/outputs/2026-03-22/contrastive_abstraction_prompting/rlad_top4_rlad_solver_template_sweep")
checkpoints = [1953, 1302, 1519, 434]
out = root / "final_report" / "CHECKPOINT_SOURCE_BREAKDOWN.md"

baseline_ds = load_from_disk(str(root / "baseline_solver_rlad_template" / "dataset"))
by_source_baseline = {}
for src in ["aime", "amc"]:
    vals = [float(r["additional_passrate"]) for r in baseline_ds if r["source"] == src]
    by_source_baseline[src] = sum(vals) / len(vals)
overall_baseline = sum(float(r["additional_passrate"]) for r in baseline_ds) / len(baseline_ds)

aime_baseline = by_source_baseline["aime"]
amc_baseline = by_source_baseline["amc"]

lines = [
    "# Checkpoint Source Breakdown",
    "",
    f"- overall baseline: `{overall_baseline:.4f}`",
    f"- AIME baseline: `{aime_baseline:.4f}`",
    f"- AMC baseline: `{amc_baseline:.4f}`",
]

for ckpt in checkpoints:
    p = root / f"checkpoint_{ckpt}_sweep" / "final_summary" / "summary_enriched.json"
    data = json.loads(p.read_text())
    lines += [
        "",
        f"## checkpoint-{ckpt}",
        "",
        "| temp | overall | AIME | Δ AIME | AIME best-of-4 | AIME I/S/R | AMC | Δ AMC | AMC best-of-4 | AMC I/S/R |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in data["temperatures"]:
        aime = row["by_source"]["aime"]
        amc = row["by_source"]["amc"]
        lines.append(
            f"| {row['temperature']:.1f} | {row['mean_conditioned_passrate']:.4f} | {aime['mean_conditioned_passrate']:.4f} | {aime['mean_conditioned_passrate'] - aime_baseline:+.4f} | {aime['mean_best_abstraction_passrate']:.4f} | {aime['improved_count']}/{aime['same_count']}/{aime['regressed_count']} | {amc['mean_conditioned_passrate']:.4f} | {amc['mean_conditioned_passrate'] - amc_baseline:+.4f} | {amc['mean_best_abstraction_passrate']:.4f} | {amc['improved_count']}/{amc['same_count']}/{amc['regressed_count']} |"
        )

out.write_text("\n".join(lines) + "\n")
print(out)
