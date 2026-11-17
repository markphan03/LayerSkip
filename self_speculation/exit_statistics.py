"""Summaries of draft exit depths; percentages use all attempted drafts."""

from tabulate import tabulate


def summarize_draft_exits(counts):
    """Use 1-based layers and round model-depth cutoffs up to whole layers.

    Empty counts mean this strategy does not draft. All-zero counts mean a
    speculative request made no drafts; its percentages are undefined.
    """
    if not counts:
        return None
    total = sum(counts)
    num_layers = len(counts)

    def percentage(count):
        return 100 * count / total if total else None

    layers = []
    cumulative = 0
    for layer, count in enumerate(counts, 1):
        cumulative += count
        layers.append({
            "layer": layer, "model_depth_percent": 100 * layer / num_layers,
            "exit_count": count, "percentage": percentage(count),
            "cumulative_percentage": percentage(cumulative),
        })
    depth_summary = []
    previous = 0
    for depth_percent in (25, 50, 75, 100):
        cutoff = (num_layers * depth_percent + 99) // 100
        count = sum(counts[previous:cutoff])
        depth_summary.append({
            "model_depth_percent": depth_percent,
            "first_layer": previous + 1 if cutoff > previous else None,
            "last_layer": cutoff,
            "exit_count": count, "percentage": percentage(count),
            "cumulative_exit_count": sum(counts[:cutoff]),
            "cumulative_percentage": percentage(sum(counts[:cutoff])),
        })
        previous = cutoff
    return {"num_layers": num_layers, "total_drafts": total,
            "layers": layers, "depth_summary": depth_summary}


def format_draft_exit_statistics(stats, title="Draft exit depths", *, tablefmt="simple_outline"):
    if stats is None:
        return ""

    def percent(value):
        return "N/A" if value is None else f"{value:.2f}%"

    layer_rows = [
        [row["layer"], percent(row["model_depth_percent"]), row["exit_count"],
         percent(row["percentage"]), percent(row["cumulative_percentage"])]
        for row in stats["layers"]
    ]
    summary_rows = [
        [f"{row['model_depth_percent'] - 25}–{row['model_depth_percent']}%",
         f"{row['first_layer']}–{row['last_layer']}" if row["first_layer"] is not None else "—",
         row["exit_count"], percent(row["percentage"]), percent(row["cumulative_percentage"])]
        for row in stats["depth_summary"]
    ]
    return "\n".join([
        f"{title} — {stats['total_drafts']:,} drafts",
        "Includes rejected drafts and depth-limit fallbacks; excludes prefill and verification.",
        "",
        tabulate(layer_rows, headers=["Layer", "Model depth", "Exits", "Share", "Cumulative"],
                 tablefmt=tablefmt, disable_numparse=True, colalign=("right",) * 5),
        "",
        "Model-depth summary (cutoffs rounded up; cumulative = exited by this depth)",
        "",
        tabulate(summary_rows, headers=["Depth band", "Layers", "Exits", "Share", "Cumulative"],
                 tablefmt=tablefmt, disable_numparse=True,
                 colalign=("left", "right", "right", "right", "right")),
        "",
        f"Layer {stats['num_layers']} is full depth, not an early exit.",
    ])
