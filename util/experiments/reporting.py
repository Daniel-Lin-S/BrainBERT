"""Render immutable fit monitors into campaign PNG figures and CSV tables.

Invoke with --artifact-root and --campaign. Input fit references resolve to
scientific.json, checkpoint ancestry and root/resume-segment monitor tables.
Ancestor training and validation are included through the restored step;
ancestor test results are excluded. Outputs are grouped
by split and diagnostic under campaigns/<campaign>/reports/{figures,tables}.
output_index.csv records each product's fit, checkpoint, state and meaning.
No model fitting, feature extraction, or TensorBoard writes occur here.
"""

import argparse
import fcntl
import io
import json
import math
import textwrap
import warnings
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .storage import atomic_write, file_digest
from .report_settings import report_settings, reports_requested
from .monitoring import ACTIONS, STRATEGIES, SUM_FIELDS

DPI = 300
LEGEND_FONT_SIZE = 20
EPOCH_FONT_SIZE = 16
DENSE_EPOCH_FONT_SIZE = 12
DENSE_EPOCH_THRESHOLD = 10
EPOCH_LABEL_HEADROOM = 0.3
EPOCH_LEGEND_COLUMNS = 5
EPOCH_LINE_STYLES = ("-", "--", "-.", ":")
TITLE_FONT_SIZE = 22
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 18
LINE_STYLES = {
    "train": "-",
    "validation": "--",
    "test": "-.",
    "temporal_only": "-",
    "spectral_only": "--",
    "overlap": "-.",
    "zero": "-",
    "random": "--",
    "unchanged": "-.",
    "all": ":",
    "temporal": ":",
    "spectral": ":",
    "union": "--",
}
COLORS = {
    "train": "#0173b2",
    "validation": "#de8f05",
    "test": "#029e73",
    "temporal_only": "#0173b2",
    "spectral_only": "#de8f05",
    "overlap": "#029e73",
    "zero": "#cc78bc",
    "random": "#d55e00",
    "unchanged": "#56b4e9",
    "all": "#333333",
    "temporal": "#56b4e9",
    "spectral": "#f0b35a",
    "union": "#777777",
}
HIDDEN_SUMS = set(SUM_FIELDS) - {"count", "content_count"}
METRICS = ("l1", "content_l1", "relative_l1", "relative_content_l1")
LABELS = {
    "total_loss": "Total loss",
    "l1": "L1",
    "content_l1": "Content L1",
    "relative_l1": "L1 /\nconstant prediction L1",
    "relative_content_l1": "Content L1 /\nconstant prediction L1 (content)",
    "std_ratio": "Prediction / target\nstandard deviation",
    "effective_rank": "Effective rank",
    "effective_rank_fraction": "Effective rank / attainable rank",
    "pca95_components": "Components for 95% variance",
}
TABLES = (
    "epoch_metrics",
    "metrics",
    "mask_groups",
    "frequency_coverage",
    "validation_rank",
    "validation_frequency",
    "test_rank",
    "test_frequency",
)
PHASES = ("train", "validation", "test")
FOLDERS = {"train": "training", "validation": "validation", "test": "test"}
DISPLAY = {
    "gradient_norm_mean": "Interval mean",
    "gradient_norm_max": "Interval maximum",
    "clipping_fraction": "Clipped updates",
    "lr_group_0_last_used": "Learning rate",
    "temporal_only": "Temporal only",
    "spectral_only": "Spectral only",
    "overlap": "Overlap",
    "zero": "Zero",
    "random": "Random",
    "unchanged": "Unchanged",
    "all": "Overall",
    "temporal": "Temporal (incl. overlap)",
    "spectral": "Spectral (incl. overlap)",
    "union": "Any selection",
}


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV, recovering only complete flushed gzip records.

    Parameters
    ----------
    path : Path
        CSV or gzipped CSV. Missing gzip trailers produce a warning.
    """
    if path.suffix == ".gz":
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        data = decoder.decompress(path.read_bytes()).decode("utf-8")
        if not decoder.eof:
            warnings.warn(f"Missing gzip footer; auditing rows: {path}")
            if not data.endswith("\n"):
                raise ValueError(f"Incomplete CSV record: {path}")
    else:
        data = path.read_text()
    frame = pd.read_csv(io.StringIO(data))
    if frame.empty:
        raise ValueError(f"Empty monitor table: {path}")
    if np.isinf(frame.select_dtypes("number").to_numpy()).any():
        raise ValueError(f"Infinite monitor value: {path}")
    return frame


def load_history(
    fit: Path, include_parent: bool = True, ancestry: tuple = (),
) -> dict[str, pd.DataFrame]:
    """Merge restored ancestry and invocation tables at absolute steps.

    Parameters
    ----------
    fit : Path
        Managed fit directory containing immutable provenance and monitors.
    include_parent : bool, optional
        Include ancestor training/validation through the restored checkpoint,
        default True. Test results always belong to the current fit only.
    ancestry : tuple, optional
        Ancestor paths used internally to reject cycles, default empty.
    """
    fit = fit.resolve()
    if fit in ancestry:
        raise ValueError(f"Cyclic checkpoint ancestry at {fit}.")
    root = fit / "monitoring"
    roots = [root] + sorted((root / "segments").glob("*"))
    history = {}
    for segment in roots:
        metrics_path = segment / "metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = read_table(metrics_path)
        training = metrics[metrics.phase == "train"]
        cutoff = int(training.interval_start.min()) if len(training) else None
        for name in TABLES:
            candidates = [segment / f"{name}.csv", segment / f"{name}.csv.gz"]
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                continue
            frame = metrics if name == "metrics" else read_table(path)
            previous = history.get(name)
            if previous is not None and cutoff is not None:
                previous = previous[previous.step < cutoff]
            history[name] = pd.concat(
                [previous, frame] if previous is not None else [frame],
                ignore_index=True,
            )
    if "metrics" not in history:
        raise ValueError(f"No monitor history in {fit}")
    if include_parent:
        history = merge_parent_history(fit, history, (*ancestry, fit))
    overall = history["metrics"]
    if overall.duplicated(["phase", "step"]).any():
        raise ValueError("Conflicting duplicate phase/step monitor rows.")
    overall["total_loss"] = overall.original_loss
    overall["std_ratio"] = overall.prediction_std / overall.target_std
    for phase in ("validation", "test"):
        name = f"{phase}_frequency"
        if name not in history:
            continue
        for step, block in history[name].groupby("step"):
            expected = block.frequency_hz.nunique() * 4
            if (
                len(block) != expected
                or block.duplicated(["strategy", "frequency_hz"]).any()
            ):
                raise ValueError(f"Incomplete {phase} frequency block: {step}")
            match = overall[
                (overall.phase == phase)
                & ((overall.step == step) | (overall.checkpoint_step == step))
            ]
            if len(match) != 1 or int(
                block.loc[block.strategy == "all", "selected_cells"].sum()
            ) != int(match.iloc[0]["count"]):
                raise ValueError(f"Frequency counts disagree at {phase} {step}")
    return history



def merge_parent_history(fit: Path, history: dict, ancestry: tuple) -> dict:
    """Prepend checkpoint-bounded ancestor monitors without ancestor tests."""
    provenance = fit / "provenance.json"
    if not provenance.exists():
        return history
    runner = json.loads(provenance.read_text())["invocation"]["exp"]["runner"]
    source = runner.get("parent_checkpoint")
    if not source:
        return history
    checkpoint = Path(source).resolve()
    expected = runner.get("parent_checkpoint_sha256")
    if not expected or file_digest(checkpoint) != expected:
        raise ValueError("Parent checkpoint checksum differs from provenance.")
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cutoff = int(state["completed_steps"])
    del state
    parent = checkpoint.parent.parent
    previous = load_history(parent, ancestry=ancestry)
    for name, frame in previous.items():
        if name.startswith("test_"):
            continue
        frame = frame[frame.step <= cutoff].copy()
        if "phase" in frame:
            frame = frame[frame.phase != "test"]
        if frame.empty:
            continue
        if "source_fit_id" not in frame:
            frame["source_fit_id"] = parent.name
        current = history.get(name)
        if current is not None:
            current = current.copy()
            current["source_fit_id"] = fit.name
            keys = ["phase", "step"] if "phase" in frame else ["step"]
            replaced = pd.MultiIndex.from_frame(current[keys])
            frame = frame[~pd.MultiIndex.from_frame(frame[keys]).isin(replaced)]
            frame = pd.concat([frame, current], ignore_index=True)
        history[name] = frame.sort_values("step", kind="stable")
    return history

def verify_fit(fit: Path) -> None:
    """Check all immutable completion artifacts without loading a model."""
    record = json.loads((fit / "complete.json").read_text())
    if record["fit_id"] != fit.name:
        raise ValueError("Completion record identifies another fit.")
    for name, expected in record["artifacts"].items():
        path = (fit / name).resolve()
        if not path.is_relative_to(fit.resolve()):
            raise ValueError("Completion artifact escapes its fit directory.")
        if file_digest(path) != expected:
            raise ValueError(f"Completion checksum mismatch: {path}")


def figure_output(name: str) -> str:
    """Group PNGs by split, with one shared mask-condition subfolder.

    Parameters
    ----------
    name : str
        Semantic figure name used by report selection.

    Returns
    -------
    str
        Report-relative PNG path; margins and grids have distinct prefixes.
    """
    name = name.replace("frequency_coverage/coverage", "frequency_coverage")
    parts = name.split("/")
    if parts[0] == "optimization":
        parts = ["training", "optimization", *parts[1:]]
    if len(parts) > 1 and parts[1] == "fixed_masks":
        return f"figures/{'/'.join(parts)}.png"
    if "mask_conditions" in parts:
        folder = "/".join(parts[:2])
        filename = "_".join(parts[2:])
    else:
        folder = parts[0]
        filename = "_".join(parts[1:])
    return f"figures/{folder}/{filename}.png"


class Report:
    """Publish independently usable outputs with one campaign index."""

    def __init__(
        self,
        root: Path,
        fit: Path,
        best: int,
        epochs: list,
        settings: object = None,
    ) -> None:
        self.settings = report_settings(settings)
        self.root = root.resolve()
        self.fit = fit.name
        self.fit_path = fit.resolve()
        self.best = best
        self.epochs = epochs
        self.entries = []
        self.root.mkdir(parents=True, exist_ok=True)
        sns.set_theme(
            style="whitegrid",
            context="talk",
            rc={
                "font.size": LABEL_FONT_SIZE,
                "axes.labelsize": LABEL_FONT_SIZE,
                "axes.titlesize": TITLE_FONT_SIZE,
                "xtick.labelsize": TICK_FONT_SIZE,
                "ytick.labelsize": TICK_FONT_SIZE,
                "legend.fontsize": LEGEND_FONT_SIZE,
            },
        )

    def wants(self, kind: str, name: str) -> bool:
        """Select an output by its explicit format, family, and split."""
        format_name = "png" if kind == "figures" else "csv"
        phase = name.split("/")[0]
        phase = {"training": "train", "optimization": "train"}.get(phase, phase)
        if (
            format_name not in self.settings["formats"]
            or phase not in self.settings["splits"]
        ):
            return False
        if "/margins/" in name:
            family = "mask_margins"
        elif "/full_grid/" in name:
            family = "mask_full_grid"
        elif "frequency_coverage/" in name or "/content_distribution/" in name:
            family = "frequency_coverage"
        elif "/duration/" in name and "/fixed_masks/" in name:
            family = ("frequency_errors" if kind == "figures"
                      else "objective_components")
        elif (
            "/frequency/" in name or "/l1_by_frequency/" in name
            or "/fixed_masks/interpolation/" in name
        ):
            family = "frequency_errors"
        elif "/representation/" in name:
            family = "representation_rank"
        elif name.startswith("optimization/"):
            family = "optimization"
        elif "/provenance/" in name:
            family = "provenance"
        elif "objective" in name:
            family = "objective_components"
        elif kind == "tables":
            family = "overall_metrics"
        else:
            family = "learning_curves"
        return family in self.settings[kind]

    def register(self, name: str, meaning: str, state: str = "ready") -> None:
        """Publish the index after each completed product."""
        if state == "pending":
            kind, relative = name.split("/", 1)
            if not self.wants(kind, relative):
                return
        if name.startswith("figures/"):
            name = figure_output(name[len("figures/") : -len(".png")])
        self.entries = [e for e in self.entries if e["output"] != name]
        self.entries.append(
            {
                "output": name,
                "description": meaning,
                "state": state,
                "fit_id": self.fit,
                "checkpoint_step": self.best,
            }
        )
        frame = pd.DataFrame(self.entries)
        atomic_write(
            self.root / "output_index.csv",
            lambda p: frame.to_csv(p, index=False),
            lambda p: pd.read_csv(p),
        )

    def table(self, name: str, frame: pd.DataFrame, meaning: str) -> None:
        """Save a nonempty derived table, preserving undefined values."""
        if frame.empty or not self.wants("tables", name):
            return
        frame = frame.drop(columns=list(HIDDEN_SUMS), errors="ignore")
        if name.startswith("test/") and hasattr(self, "policy_name"):
            name = name.replace("test/", f"test/{self.policy_name}/", 1)
        if hasattr(self, "fixed_replacement"):
            name = name.replace(
                "test/fixed_masks/",
                f"test/fixed_masks/{self.fixed_replacement}/", 1,
            )
        path = f"tables/{name}.csv"
        atomic_write(
            self.root / path,
            lambda p: frame.to_csv(p, index=False),
            lambda p: pd.read_csv(p),
        )
        self.register(path, meaning)

    def figure(self, name: str, fig: object, meaning: str) -> None:
        """Save and audit a PNG without exposing a partially written image."""
        if not self.wants("figures", name):
            plt.close(fig)
            return
        if meaning.startswith("Test") and hasattr(self, "test_label"):
            meaning = meaning.replace("Test", self.test_label, 1)
        if name.startswith("test/") and hasattr(self, "policy_name"):
            name += f"_{self.policy_name}"
        if hasattr(self, "fixed_replacement"):
            name = name.replace(
                "test/fixed_masks/",
                f"test/fixed_masks/{self.fixed_replacement}/", 1,
            )
            meaning += f"; {self.fixed_replacement} replacement"
        path = figure_output(name)
        for ax in fig.axes:
            ax.set_ylabel(textwrap.fill(ax.get_ylabel().replace("\n", " "), 22))
        has_epochs = any(
            getattr(ax, "_report_epochs", False) for ax in fig.axes
        )
        if len(self.epochs) > DENSE_EPOCH_THRESHOLD:
            for ax in fig.axes:
                if getattr(ax, "_report_epochs", False):
                    lower, upper = ax.get_ylim()
                    ax.set_ylim(
                        lower, upper + (upper - lower) * EPOCH_LABEL_HEADROOM,
                    )
        has_legends = any(ax.get_legend_handles_labels()[1] for ax in fig.axes)
        fig.tight_layout(
            rect=(
                0,
                0.045 if has_epochs else 0,
                1,
                0.9 if has_legends else 0.94,
            ),
            h_pad=5 if has_legends else 2.5,
            w_pad=2,
        )
        shared_legends(fig)
        if has_epochs:
            fig.text(
                0.5,
                0.012,
                f"E1–E{len(self.epochs)}: completed epochs; "
                "dotted line: selected checkpoint",
                ha="center",
                fontsize=LEGEND_FONT_SIZE,
            )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tops = [ax.get_tightbbox(renderer).y1 for ax in fig.axes]
        tops.extend(
            legend.get_window_extent(renderer).y1 for legend in fig.legends
        )
        title_gap = TITLE_FONT_SIZE * fig.dpi / 72
        title_y = (max(tops) + title_gap) / fig.bbox.height
        fig.suptitle(
            textwrap.fill(meaning, 75),
            fontsize=TITLE_FONT_SIZE,
            y=title_y,
            va="bottom",
        )
        try:
            atomic_write(
                self.root / path,
                lambda p: fig.savefig(
                    p, format="png", dpi=DPI, bbox_inches="tight"
                ),
                lambda p: plt.imread(p, format="png"),
            )
        finally:
            plt.close(fig)
        self.register(path, meaning)

    def step_axis(self, ax: object) -> None:
        """Annotate actual optimizer steps and epoch boundaries."""
        ax.set_xlabel("Optimiser steps")
        ax._report_epochs = True
        for epoch, step in enumerate(self.epochs, 1):
            ax.axvline(step, color="0.8", lw=0.8, zorder=0)
            ax.text(
                step,
                0.98,
                f"E{epoch}",
                rotation=(
                    90 if len(self.epochs) > DENSE_EPOCH_THRESHOLD else 0
                ),
                fontsize=(
                    DENSE_EPOCH_FONT_SIZE
                    if len(self.epochs) > DENSE_EPOCH_THRESHOLD
                    else EPOCH_FONT_SIZE
                ),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 1,
                },
            )
        ax.axvline(self.best, color="black", ls=":", lw=1.5)


def shared_legends(fig: object) -> None:
    """Share matching line legends across panels, keeping disjoint rows.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Laid-out figure whose labeled artists define legend semantics.
    """
    groups = []
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        if not labels:
            continue
        entries = dict(zip(labels, handles))
        matching = [g for g in groups if set(g["entries"]) & set(entries)]
        if matching:
            group = matching[0]
            group["entries"].update(entries)
            group["axes"].append(ax)
        else:
            groups.append({"entries": entries, "axes": [ax]})
    for group in groups:
        boxes = [ax.get_position() for ax in group["axes"]]
        left = min(box.x0 for box in boxes)
        right = max(box.x1 for box in boxes)
        top = max(box.y1 for box in boxes)
        entries = group["entries"]
        fig.legend(
            entries.values(),
            entries.keys(),
            loc="lower center",
            bbox_to_anchor=((left + right) / 2, top + 0.01),
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            ncol=min(len(entries), getattr(fig, "_legend_columns", 3)),
        )


def line_panel(
    ax: object,
    frame: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    reference: bool = False,
) -> None:
    """Draw unsmoothed series with consistent color and line encodings."""
    if y not in frame:
        return
    groups = list(frame[group].drop_duplicates())
    order = list(DISPLAY)
    groups.sort(
        key=lambda value: order.index(value) if value in order else len(order)
    )
    colors = sns.color_palette("colorblind", max(len(groups), 3))
    styles = ("-", "--", "-.", ":")
    for index, value in enumerate(groups):
        rows = frame[frame[group] == value].sort_values(x)
        ax.plot(
            rows[x],
            rows[y],
            label=DISPLAY.get(value, str(value)),
            color=COLORS.get(value, colors[index]),
            ls=LINE_STYLES.get(value, styles[index % len(styles)]),
            lw=2,
            marker="o" if value == "validation" else None,
            markersize=3,
        )
    if reference:
        ax.axhline(1, color="gray", ls="--", lw=1)
    ax.set_ylabel(LABELS.get(y, y.replace("_", " ")))


def learning_figures(report: Report, history: dict) -> None:
    """Render overall learning, rank, and optimization histories."""
    metrics = history["metrics"]
    learning = metrics[metrics.phase != "test"]
    keys = ("total_loss", *METRICS, "std_ratio")
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    for ax, key in zip(axes.flat, keys):
        line_panel(
            ax,
            learning,
            "step",
            key,
            "phase",
            key.startswith("relative") or key == "std_ratio",
        )
        report.step_axis(ax)
    report.figure(
        "training/losses/learning_curves",
        fig,
        "Masked reconstruction learning; dotted line: selected "
        "checkpoint; training is an online interval average",
    )
    train = learning[learning.phase == "train"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    groups = (
        [k for k in train if k.startswith("lr_group_")],
        ["gradient_norm_mean", "gradient_norm_max"],
        ["clipping_fraction"],
    )
    for ax, columns in zip(axes, groups):
        long = train.melt(
            id_vars="step",
            value_vars=columns,
            var_name="monitor",
            value_name="value",
        )
        line_panel(ax, long, "step", "value", "monitor")
        report.step_axis(ax)
    axes[0].set_ylabel("Learning rate")
    axes[1].set_ylabel("Gradient norm before clipping")
    threshold = json.loads((report_fit(report) / "scientific.json").read_text())
    axes[1].axhline(threshold["runner"]["grad_clip"], color="gray", ls="--")
    axes[2].set_ylabel("Clipping fraction")
    axes[2].set_ylim(0, 1)
    report.figure("optimization/dynamics", fig, "Optimization dynamics")
    if "validation_rank" in history:
        rank = history["validation_rank"].assign(phase="validation")
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        for ax, key in zip(
            axes,
            ("effective_rank", "effective_rank_fraction", "pca95_components"),
        ):
            line_panel(ax, rank, "step", key, "phase")
            report.step_axis(ax)
        report.figure(
            "validation/representation/rank",
            fig,
            "Diversity of pooled masked-window embeddings",
        )


def report_fit(report: Report) -> Path:
    """Resolve the shared fit associated with the campaign report."""
    return report.fit_path


def condition_figures(report: Report, history: dict) -> None:
    """Keep six marginal totals separate from the nine masking conditions."""
    for phase, rows in history["mask_groups"].groupby("phase", sort=False):
        folder = FOLDERS[phase]
        for family, chosen in (
            (
                "margins",
                rows[(rows.strategy == "all") | (rows.action == "all")],
            ),
            (
                "full_grid",
                rows[(rows.strategy != "all") & (rows.action != "all")],
            ),
        ):
            chosen = chosen[chosen["count"] > 0].copy()
            chosen["condition"] = chosen.strategy + " / " + chosen.action
            report.table(
                f"{folder}/mask_conditions/{family}/metrics",
                chosen,
                f"{phase} {family}: pooled cell-weighted statistics",
            )
            if not report.wants(
                "figures", f"{folder}/mask_conditions/{family}/history"
            ):
                continue
            if phase != "test":
                if family == "margins":
                    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
                    for row_index, (field, pooled) in enumerate(
                        (("strategy", "action"), ("action", "strategy"))
                    ):
                        subset = chosen[chosen[pooled] == "all"]
                        for ax, key in zip(axes[row_index], METRICS):
                            line_panel(
                                ax,
                                subset,
                                "step",
                                key,
                                field,
                                key.startswith("relative"),
                            )
                            report.step_axis(ax)
                    report.figure(
                        f"{folder}/mask_conditions/{family}/history",
                        fig,
                        f"{phase.title()} six mask margins",
                    )
                else:
                    for key in METRICS:
                        fig, axes = plt.subplots(3, 3, figsize=(17, 13))
                        for ax, ((strategy, action), group) in zip(
                            axes.flat,
                            chosen.groupby(["strategy", "action"], sort=False),
                        ):
                            ax.plot(group.step, group[key], lw=2)
                            ax.set_title(
                                textwrap.fill(
                                    f"{DISPLAY[strategy]} / {DISPLAY[action]}",
                                    32,
                                )
                            )
                            ax.set_ylabel("")
                            if key.startswith("relative"):
                                ax.axhline(1, color="gray", ls="--")
                            report.step_axis(ax)
                        fig.supylabel(LABELS[key], fontsize=LABEL_FONT_SIZE)
                        report.figure(
                            f"{folder}/mask_conditions/full_grid/{key}_history",
                            fig,
                            f"{phase.title()} nine conditions: {LABELS[key]}",
                        )
            if phase in ("validation", "test"):
                snapshot = chosen[
                    chosen.step
                    == (
                        report.best
                        if phase == "validation"
                        else chosen.step.max()
                    )
                ]
                fig, axes = plt.subplots(
                    2, 2, figsize=(20 if family == "full_grid" else 16, 12)
                )
                for ax, key in zip(axes.flat, METRICS):
                    if family == "full_grid":
                        grid = snapshot.pivot(
                            index="strategy", columns="action", values=key
                        ).reindex(
                            index=[
                                x
                                for x in STRATEGIES
                                if x in snapshot.strategy.values
                            ],
                            columns=[
                                x
                                for x in ACTIONS
                                if x in snapshot.action.values
                            ],
                        )
                        grid = grid.rename(index=DISPLAY, columns=DISPLAY)
                        sns.heatmap(
                            grid,
                            annot=True,
                            fmt=".3f",
                            cmap="viridis",
                            ax=ax,
                            cbar_kws={"label": LABELS[key]},
                            annot_kws={"fontsize": LABEL_FONT_SIZE},
                        )
                        ax.tick_params(axis="both", labelrotation=0)
                        ax.set_xlabel("Replacement")
                        ax.set_ylabel("Strategy")
                    else:
                        categories = [
                            DISPLAY[strategy if action == "all" else action]
                            for strategy, action in zip(
                                snapshot.strategy, snapshot.action
                            )
                        ]
                        ax.scatter(snapshot[key], categories, s=85)
                        low, high = snapshot[key].min(), snapshot[key].max()
                        if key.startswith("relative"):
                            low, high = min(low, 1), max(high, 1)
                        padding = (high - low) * 0.08
                        if padding > 0:
                            ax.set_xlim(low - padding, high + padding)
                        if key.startswith("relative"):
                            ax.axvline(1, color="gray", ls="--")
                        ax.set_xlabel(LABELS[key])
                    ax.set_title(textwrap.fill(LABELS[key], 35))
                report.figure(
                    f"{folder}/mask_conditions/{family}/selected_checkpoint",
                    fig,
                    f"{phase.title()} mask conditions at validation-selected checkpoint "
                    f"{report.best}",
                )


def frequency_evolution(report: Report, frame: pd.DataFrame) -> None:
    """Plot saved epoch-end frequency profiles with a shared epoch legend.

    Parameters
    ----------
    report : Report
        Output destination and absolute completed-epoch step boundaries.
    frame : pandas.DataFrame
        Validation rows containing step, strategy, frequency_hz,
        l1, content_l1, relative_l1 and relative_content_l1 columns.
        Only the overall
        strategy at exact epoch boundaries is plotted; no interpolation
        between checkpoints is performed.
    """
    overall = frame[frame.strategy == "all"]
    epochs = [
        (epoch, step)
        for epoch, step in enumerate(report.epochs, 1)
        if step in set(overall.step)
    ]
    if not epochs:
        raise ValueError("No completed-epoch frequency diagnostics available.")
    missing = set(report.epochs) - set(overall.step)
    if missing:
        warnings.warn(
            f"Missing frequency diagnostics at epoch-end steps: "
            f"{sorted(missing)}"
        )
    colors = sns.color_palette("viridis", n_colors=len(report.epochs))
    for suffix, keys, description in (
        (
            "", ("relative_l1", "relative_content_l1"),
            "loss normalised by frequency-specific constant prediction L1",
        ),
        (
            "_unnormalised", ("l1", "content_l1"),
            "unnormalised reconstruction loss",
        ),
    ):
        fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        fig._legend_columns = EPOCH_LEGEND_COLUMNS
        for ax, key in zip(axes, keys):
            for epoch, step in epochs:
                rows = overall[overall.step == step].sort_values("frequency_hz")
                if not np.isfinite(rows[["frequency_hz", key]]).all().all():
                    plt.close(fig)
                    raise ValueError(
                        f"Nonfinite {key} frequency profile at epoch {epoch}."
                    )
                ax.plot(
                    rows.frequency_hz, rows[key], color=colors[epoch - 1],
                    linestyle=EPOCH_LINE_STYLES[
                        ((epoch - 1) // EPOCH_LEGEND_COLUMNS)
                        % len(EPOCH_LINE_STYLES)
                    ],
                    linewidth=2, label=f"Epoch {epoch}",
                )
            ax.set_ylabel(LABELS[key])
            ax.set_xlabel("Frequency (Hz)")
        report.figure(
            f"validation/frequency/evolution{suffix}", fig,
            "Validation frequency profiles across completed epochs; "
            + description,
        )


def frequency_figures(
    report: Report,
    history: dict,
    phases: tuple = ("validation", "test"),
    coverage_enabled: bool = True,
) -> None:
    """Plot actual-bin frequency errors and selection rates."""
    if "validation" in phases and "validation_frequency" in history:
        frame = history["validation_frequency"]
        report.table(
            "validation/frequency/reconstruction", frame,
            "Variable-mask validation history for epoch evolution only",
        )
        if report.wants("figures", "validation/frequency/evolution"):
            frequency_evolution(report, frame)
    if not coverage_enabled:
        return
    coverage = history["frequency_coverage"]
    for phase in PHASES:
        frame = coverage[coverage.phase == phase].copy()
        if frame.empty:
            continue
        if phase == "train":
            frame = frame[frame.scope == "interval"]
            frame = frame.groupby(["frequency_hz", "strategy"], as_index=False)[
                ["selected_cells", "valid_cells"]
            ].sum()
            frame["selection_rate"] = frame.selected_cells / frame.valid_cells
        else:
            frame = frame[frame.step == frame.step.max()]
        folder = FOLDERS[phase]
        report.table(
            f"{folder}/frequency_coverage/coverage",
            frame,
            "Training exposure sums canonical intervals across resumes",
        )
        if not report.wants("figures", f"{folder}/frequency_coverage/coverage"):
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        line_panel(ax, frame, "frequency_hz", "selection_rate", "strategy")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Selected / valid cells")
        ax.set_ylim(0, 1)
        report.figure(
            f"{folder}/frequency_coverage/coverage",
            fig,
            f"{phase.title()} frequency masking coverage",
        )


def export_tables(report: Report, history: dict) -> None:
    """Publish numeric results before slower figure rendering."""
    metrics = history["metrics"]
    for phase in PHASES:
        frame = metrics[metrics.phase == phase]
        report.table(
            f"{FOLDERS[phase]}/overall/metrics",
            frame,
            "Original objective is batch-mean; other errors are pooled",
        )
    optimization = metrics[metrics.phase == "train"]
    columns = ["step", "interval_start", "interval_end", "batches"] + [
        c
        for c in optimization
        if c.startswith(("lr_group_", "gradient_")) or c == "clipping_fraction"
    ]
    report.table(
        "optimization/dynamics",
        optimization[columns],
        "Learning rates and pre-clipping gradient diagnostics",
    )
    notes = pd.DataFrame(
        [
            {
                "topic": "aggregation",
                "detail": (
                    "total_loss is batch-mean; l1 "
                    "and content_l1 are cell-pooled"
                ),
            },
            {
                "topic": "epochs",
                "detail": (
                    "Use canonical step intervals; resumed "
                    "epoch summaries can be partial"
                ),
            },
            {
                "topic": "frequency_logs",
                "detail": (
                    "Missing gzip footers are warned; "
                    "only complete audited blocks are used"
                ),
            },
            {
                "topic": "inference",
                "detail": (
                    "Descriptive single-fit results; "
                    "no independent-session significance"
                ),
            },
        ]
    )
    report.table(
        "training/provenance/interpretation",
        notes,
        "Aggregation definitions and interpretation limits",
    )
    for phase in ("validation", "test"):
        name = f"{phase}_rank"
        if name in history:
            report.table(
                f"{phase}/representation/rank",
                history[name],
                "Centered covariance of pooled masked-window embeddings",
            )
        else:
            report.register(
                f"tables/{phase}/representation/rank.csv",
                "Representation diagnostics",
                "pending",
            )
    test = metrics[metrics.phase == "test"]
    if test.empty:
        report.register(
            "tables/test/overall/objective.csv",
            "Final test objective",
            "pending",
        )
        return
    row = test.iloc[-1]
    alpha = json.loads((report_fit(report) / "scientific.json").read_text())[
        "criterion"
    ]["alpha"]
    result = []
    for label, l1, content in (
        ("model", row.original_l1_loss, row.original_content_l1),
        (
            "zero_baseline",
            row.original_zero_baseline_l1,
            row.original_content_zero_baseline_l1,
        ),
    ):
        result.append(
            {
                "predictor": label,
                "total_loss": l1 + alpha * content,
                "l1": l1,
                "content_l1": content,
                "weighted_content_l1": alpha * content,
                "aggregation": "batch_mean",
                "batches": row.batches,
                "checkpoint_step": row.checkpoint_step,
            }
        )
    report.table(
        "test/overall/objective",
        pd.DataFrame(result),
        "Actual objective and matching constant-prediction objective",
    )


def combined_test_conditions(
    reports: Path,
    fit: Path,
    settings: dict,
    checkpoint_step: int,
) -> None:
    """Compare pure replacement evaluations without pooling their losses."""
    sources = list((reports.parent / "evaluations").glob("*/complete.json"))
    sources += list(
        (fit / "monitoring").rglob("test_evaluations/*/policy.json")
    )
    frames = []
    actions = set()
    for source in sources:
        metadata = json.loads(source.read_text())
        policy = metadata.get("policy", metadata)
        selected = [key for key, value in policy.items() if value == 1]
        if len(selected) != 1 or selected[0] not in {"zero", "random"}:
            continue
        action = selected[0]
        if action in actions:
            raise ValueError(
                f"Multiple completed test evaluations for {action}."
            )
        if metadata.get("checkpoint_step", checkpoint_step) != checkpoint_step:
            raise ValueError("Test policies evaluated different checkpoints.")
        frame = read_table(source.parent / "monitoring/mask_groups.csv")
        frame = frame[(frame.action == action) & (frame.strategy != "all")]
        frames.append(frame)
        actions.add(action)
    if actions != {"zero", "random"}:
        return
    selected_settings = {
        **settings,
        "figures": [x for x in settings["figures"] if x == "mask_full_grid"],
        "tables": [x for x in settings["tables"] if x == "mask_full_grid"],
    }
    report = Report(reports, fit, checkpoint_step, [], selected_settings)
    index = reports / "output_index.csv"
    if index.exists():
        report.entries = pd.read_csv(index).to_dict("records")
    report.test_label = "Test (separate 100% zero and 100% random evaluations)"
    condition_figures(report, {"mask_groups": pd.concat(frames)})


def render_test_evaluation(
    evaluation_root: Path,
    fit: Path,
    reports: Path,
    name: str,
    settings: dict,
    checkpoint_step: int | None = None,
) -> None:
    """Publish one named test policy using shared tables and plot styling."""
    if not reports_requested(settings):
        return
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / ".test_report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        settings = {**settings, "splits": ["test"]}
        if (evaluation_root / "complete.json").exists():
            metadata = json.loads(
                (evaluation_root / "complete.json").read_text()
            )
            checkpoint_step = metadata["checkpoint_step"]
            policy = metadata["policy"]
        else:
            policy = json.loads((evaluation_root / "policy.json").read_text())
        if checkpoint_step is None:
            raise ValueError(
                "A named evaluation needs its selected checkpoint step."
            )
        history = load_history(evaluation_root)
        report = Report(
            reports,
            fit,
            checkpoint_step,
            [],
            settings=settings,
        )
        description = ", ".join(
            f"{100 * probability:g}% {action} replacement"
            for action, probability in policy.items()
            if probability
        )
        report.policy_name = name
        index = reports / "output_index.csv"
        if index.exists():
            report.entries = pd.read_csv(index).to_dict("records")
        report.test_label = f"Test ({description})"
        export_tables(report, history)
        frequency_figures(report, history, phases=("test",))
        combined_test_conditions(reports, fit, settings, checkpoint_step)
        print(f"Test reports published: {report.root}", flush=True)


def render(
    artifact_root: Path,
    campaign: str,
    verify: bool = True,
    diagnostics_only: bool = False,
    settings: object = None,
) -> Path:
    """Publish available products without waiting for optional diagnostics."""
    base = artifact_root.resolve() / "campaigns" / campaign
    if settings is not None and not reports_requested(settings):
        return base / "reports"
    references = sorted((base / "fits").glob("*.json"))
    if len(references) != 1:
        raise ValueError(
            "This report requires exactly one campaign fit reference."
        )
    fit_id = json.loads(references[0].read_text())["fit_id"]
    fit = artifact_root.resolve() / "fits" / fit_id
    if settings is None:
        provenance = json.loads((fit / "provenance.json").read_text())
        settings = provenance["invocation"]["exp"]["runner"].get("reports")
    settings = report_settings(settings)
    if not reports_requested(settings):
        return base / "reports"
    if verify:
        verify_fit(fit)
    history = load_history(fit)
    root = base / "reports"
    for name, relative in (
        ("test_rank", "representation/rank.csv"),
        ("test_frequency", "frequency/reconstruction.csv"),
    ):
        path = root / "tables/test" / relative
        if path.exists():
            history[name] = read_table(path)
    validation = load_history(fit, include_parent=False)["metrics"].query(
        "phase == 'validation'"
    )
    best = int(validation.loc[validation.original_loss.idxmin(), "step"])
    cfg = json.loads((fit / "scientific.json").read_text())
    budget = cfg["task"].get("sampling", {}).get("sample_budget")
    total = cfg["runner"]["total_steps"]
    per_epoch = (
        math.ceil(budget / cfg["runner"]["train_batch_size"])
        if budget
        else total
    )
    observed = int(history["metrics"].step.max())
    epoch_steps = list(range(per_epoch, min(total, observed) + 1, per_epoch))
    named_tests = list(
        (fit / "monitoring").rglob("test_evaluations/*/monitoring/metrics.csv")
    )
    external_tests = list((base / "evaluations").glob("*/complete.json"))
    for path in named_tests + external_tests:
        evaluation_root = (
            path.parent if path.name == "complete.json" else path.parent.parent
        )
        render_test_evaluation(
            evaluation_root, fit, root, evaluation_root.name, settings, best
        )
    if named_tests or external_tests:
        settings = {
            **settings,
            "splits": [
                phase for phase in settings["splits"] if phase != "test"
            ],
        }
    report = Report(
        root,
        fit,
        best,
        epoch_steps,
        settings=settings,
    )
    if (named_tests or external_tests) and (root / "output_index.csv").exists():
        report.entries = pd.read_csv(root / "output_index.csv").to_dict(
            "records"
        )
    if diagnostics_only:
        index = root / "output_index.csv"
        if index.exists():
            report.entries = pd.read_csv(index).to_dict("records")
        if "test_rank" in history and report.wants(
            "tables", "test/representation/rank"
        ):
            report.register(
                "tables/test/representation/rank.csv",
                "Test rank uses the validation calculation",
            )
        frequency_figures(
            report, history, phases=("test",), coverage_enabled=False
        )
        return root
    export_tables(report, history)
    for function in (learning_figures, condition_figures, frequency_figures):
        try:
            function(report, history)
        except Exception as error:
            report.register(function.__name__, str(error), "failed")
            warnings.warn(
                f"Report section failed: {function.__name__}: {error}"
            )
    print(f"Saved campaign figures and tables: {root}", flush=True)
    return root


def main() -> None:
    """Resolve campaign references and generate PNG/CSV outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--in-progress", action="store_true")
    config = parser.add_mutually_exclusive_group()
    config.add_argument("--report-config", type=Path)
    config.add_argument("--settings-json")
    args = parser.parse_args()
    settings = None
    if args.report_config:
        from omegaconf import OmegaConf

        loaded = OmegaConf.load(args.report_config)
        settings = loaded
        settings = report_settings(settings)
    elif args.settings_json:
        settings = json.loads(args.settings_json)
    render(
        args.artifact_root,
        args.campaign,
        verify=not args.in_progress,
        settings=settings,
    )


if __name__ == "__main__":
    main()
