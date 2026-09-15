"""Redraw report figures from the public aggregate CSVs (requires matplotlib)."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "tables"
BLUE = "#2364aa"
ORANGE = "#c85d18"
GREEN = "#16745e"
GRAY = "#59616c"


def read_rows(name):
    with (TABLES / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(
            HERE / f"{name}.{suffix}",
            dpi=180,
            facecolor="white",
            metadata={"Date": None} if suffix == "svg" else {},
        )
    plt.close(fig)


def pretraining():
    points = [
        {"stage": r["stage"], "step": int(r["step"]), "loss": float(r["val_loss"])}
        for r in read_rows("PRETRAIN_VALIDATION_CURVE.csv")
    ]
    stages = {s: [r for r in points if r["stage"] == s] for s in ("A", "B")}
    handover = stages["A"][-1]["step"]
    decay_start = 44631  # Frozen WSD schedule, report §4.3.
    end = stages["B"][-1]["step"]
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.subplots_adjust(left=.07, right=.98, bottom=.21, top=.79, wspace=.26)
    fig.suptitle(
        "PetitGPT pretraining: reference validation loss",
        x=.07, y=.97, ha="left", fontsize=18, fontweight="bold",
    )
    fig.text(
        .07, .89, f"{len(points)} recorded evaluations across Stage A and Stage B",
        color=GRAY, fontsize=11,
    )
    colors = {"A": BLUE, "B": ORANGE}
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e7eb", linewidth=.8)
        ax.axvspan(decay_start, end, color="#eef0f3", zorder=0)
        ax.axvline(handover, color="#88919d", linestyle="--", linewidth=1)
        ax.axvline(decay_start, color="#88919d", linestyle=":", linewidth=1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xlabel("Global optimizer step", labelpad=9)
        ax.set_ylabel("Reference validation loss")
        for stage, rows in stages.items():
            if ax is axes[1] and stage == "A":
                rows = rows[-1:]
            ax.plot(
                [r["step"] for r in rows], [r["loss"] for r in rows],
                color=colors[stage], marker="o" if stage == "A" else "s",
                markersize=5, linewidth=1.8, label=f"Stage {stage}", zorder=3,
            )

    axes[0].set(title="(a) Full trajectory", xlim=(0, 51000), ylim=(2.35, 5.35))
    axes[0].set_xticks([0, 10000, 20000, 30000, 40000, 50000])
    axes[0].legend(frameon=False, loc="upper right")
    for row, offset in [(stages["A"][0], (10, 7)),
                        (stages["A"][-1], (-53, 14)),
                        (stages["B"][-1], (-50, -3))]:
        axes[0].annotate(
            f"{row['loss']:.4f}", (row["step"], row["loss"]),
            xytext=offset, textcoords="offset points", color=colors[row["stage"]],
        )
    axes[1].set(title="(b) Stage B detail", xlim=(37700, 50300), ylim=(2.43, 2.79))
    axes[1].set_xticks([38000, 41000, 44000, 47000, 50000])
    axes[1].text(38450, 2.777, "A/B handover", va="top", fontsize=9, color=GRAY)
    axes[1].text(
        44850, 2.777, f"WSD decay\nstarts at {decay_start:,}",
        va="top", fontsize=9, color=GRAY,
    )
    start = stages["B"][0]
    axes[1].annotate(
        f"{start['loss']:.4f}\nstep {start['step']:,}", (start["step"], start["loss"]),
        xytext=(14, -54), textcoords="offset points", fontsize=9, color=ORANGE,
    )
    for step, offset in [(decay_start, (8, 9)), (end, (-51, 5))]:
        row = next(r for r in stages["B"] if r["step"] == step)
        axes[1].annotate(
            f"{row['loss']:.4f}", (row["step"], row["loss"]),
            xytext=offset, textcoords="offset points", color=ORANGE,
        )
    fig.text(
        .07, .075,
        "Markers are measured values; lines only connect observations. No smoothing or added evaluation points.",
        fontsize=9.5, color=GRAY,
    )
    fig.text(
        .07, .035,
        f"A ends at {handover:,}; B starts at {start['step']:,} in a separate process. "
        "Shading marks learning-rate decay; panels use different vertical scales.",
        fontsize=9.2, color=GRAY,
    )
    save(fig, "pretraining_validation")


def posttraining():
    rows = read_rows("POSTTRAINING_TRADEOFF.csv")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.subplots_adjust(left=.10, right=.95, bottom=.23, top=.76)
    fig.suptitle(
        "Post-training: reference prediction and procedural transfer",
        x=.10, y=.97, ha="left", fontsize=17, fontweight="bold",
    )
    fig.text(.10, .90, "Five recorded snapshots · lower NLL and more passes are better",
             color=GRAY, fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(color="#e5e7eb", linewidth=.8)
    ax.set(xlim=(1.310, 1.454), ylim=(-20, 558),
           xlabel="P2 val500 reference NLL (lower is better)",
           ylabel="Procedural dev512 passes (out of 512)")
    ax.set_xticks([1.32, 1.34, 1.36, 1.38, 1.40, 1.42, 1.44])
    ax.set_yticks([0, 100, 200, 300, 400, 512])
    ax.axhline(.90 * 512, linestyle="--", linewidth=1, color="#88919d")
    ax.text(1.452, .90 * 512 - 19, "Study dev target: 90%", ha="right",
            va="top", fontsize=9, color=GRAY)
    styles = {
        "P2 step 750": (BLUE, "o", 65, (10, 12), "left"),
        "P3 step 320": (ORANGE, "s", 65, (0, 13), "center"),
        "P3 step 640": (ORANGE, "s", 65, (0, 13), "center"),
        "alpha050": (GREEN, "D", 65, (-6, -44), "right"),
        "alpha075": (GREEN, "*", 230, (0, 15), "center"),
    }
    for r in rows:
        name = r["model"]
        x, y = float(r["val500_nll"]), int(r["dev512_pass"])
        color, marker, size, offset, align = styles[name]
        ax.scatter(x, y, color=color, marker=marker, s=size, zorder=3)
        label = f"{name}\n{x:.4f} · {y}/512"
        if name == "alpha075":
            label = "alpha075 (released)\n" + f"{x:.4f} · {y}/512"
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                    ha=align, color=color, fontsize=9.5,
                    fontweight="bold" if name == "alpha075" else "normal")
    fig.text(.10, .105,
             "alpha050 / alpha075 blend P2 step 750 with P3 step 320. P3 step 640 is a separate endpoint.",
             fontsize=9.5, color=GRAY)
    fig.text(.10, .06,
             "All five use the same diagnostic comparison. dev512 is correlated, reused development data.",
             fontsize=9.5, color=GRAY)
    save(fig, "posttraining_tradeoff")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "path", "svg.hashsalt": "petitgpt-report-v1",
    })
    pretraining()
    posttraining()
    print(f"Wrote two PNG/SVG figure pairs to {HERE}")


if __name__ == "__main__":
    main()
