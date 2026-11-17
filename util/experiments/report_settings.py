"""Explicit report selections shared by rendering and training integration.

Missing/null settings request no output. Nonempty settings contain formats
(png/csv), splits (train/validation/test), and named figure/table families.
There are no implicit report families, formats, or dataset-specific defaults.
"""

from collections.abc import Mapping, Sequence

FIGURE_FAMILIES = frozenset(
    {
        "learning_curves",
        "mask_margins",
        "mask_full_grid",
        "frequency_errors",
        "frequency_coverage",
        "representation_rank",
        "optimization",
    }
)
TABLE_FAMILIES = frozenset(
    {
        "overall_metrics",
        "objective_components",
        "mask_margins",
        "mask_full_grid",
        "frequency_errors",
        "frequency_coverage",
        "representation_rank",
        "optimization",
        "provenance",
    }
)
ALLOWED = {
    "formats": frozenset({"png", "csv"}),
    "splits": frozenset({"train", "validation", "test"}),
    "figures": FIGURE_FAMILIES,
    "tables": TABLE_FAMILIES,
}


def report_settings(value: object) -> dict:
    """Validate explicit selections, interpreting missing/empty as disabled."""
    if (
        value is None
        or value == ""
        or (isinstance(value, (Mapping, Sequence)) and len(value) == 0)
    ):
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(
            "reports must map formats, splits, figures, and tables."
        )
    if set(value) - set(ALLOWED):
        raise ValueError(
            f"Unknown report settings: {set(value) - set(ALLOWED)}"
        )
    result = {}
    for key, allowed in ALLOWED.items():
        selection = value.get(key)
        if selection is None or selection == "":
            selection = []
        if not isinstance(selection, Sequence) or isinstance(selection, str):
            raise ValueError(f"reports.{key} must be a list of names.")
        unknown = set(selection) - allowed
        if unknown:
            raise ValueError(f"Unknown reports.{key}: {sorted(unknown)}")
        result[key] = list(dict.fromkeys(selection))
    return result


def reports_requested(value: object) -> bool:
    """Whether explicit selections request any generated PNG or CSV output."""
    settings = report_settings(value)
    return bool(
        settings["splits"]
        and (
            ("png" in settings["formats"] and settings["figures"])
            or ("csv" in settings["formats"] and settings["tables"])
        )
    )
