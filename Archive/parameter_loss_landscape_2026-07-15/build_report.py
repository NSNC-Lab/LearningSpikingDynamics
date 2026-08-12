"""Build a paginated PDF report from run_landscape.py outputs.

This script intentionally has no torch dependency so it can run with the
bundled document runtime that provides ReportLab and pypdf.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "output" / "data"
PLOT_DIR = HERE / "output" / "plots"
PDF_DIR = HERE / "output" / "pdf"
PDF_PATH = PDF_DIR / "archived_local_eligibility_parameter_landscapes.pdf"

PARAMETER_ORDER = [
    "Strf_gain",
    "Strf_alpha",
    "output_ad",
    "on_ron_gSYN",
    "off_ron_gSYN",
    "on_sonoff_gSYN",
    "off_sonoff_gSYN",
    "sonoff_ron_gSYN",
    "abs_ref",
    "rel_ref_a",
    "rel_ref_b",
    "rel_ref_c",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: str | float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: str | float, digits: int = 3) -> str:
    number = f(value)
    if not math.isfinite(number):
        return "nan"
    if number == 0:
        return "0"
    magnitude = abs(number)
    if magnitude < 1e-3 or magnitude >= 1e5:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def pct(value: str | float) -> str:
    number = f(value)
    return "N/A" if not math.isfinite(number) else f"{100.0 * number:.1f}%"


def page_decor(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    width, height = landscape(letter)
    canvas.setStrokeColor(colors.HexColor("#d9d9d9"))
    canvas.setLineWidth(0.5)
    canvas.line(0.55 * inch, height - 0.42 * inch, width - 0.55 * inch, height - 0.42 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(0.58 * inch, height - 0.31 * inch, "Archived local-eligibility parameter landscapes")
    canvas.drawRightString(width - 0.58 * inch, 0.28 * inch, f"Page {doc.page}")
    canvas.restoreState()


def styled_table(
    data: list[list[Any]],
    col_widths: list[float],
    font_size: float = 7.0,
    repeat_rows: int = 1,
    alignments: dict[int, str] | None = None,
) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=repeat_rows, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24445c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 1.4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c7c7c7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            style.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f3f6f8")))
    if alignments:
        for column, alignment in alignments.items():
            style.append(("ALIGN", (column, 1), (column, -1), alignment))
    table.setStyle(TableStyle(style))
    return table


def build_report() -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    summary = read_csv(DATA_DIR / "alignment_summary.csv")
    aggregate = read_csv(DATA_DIR / "aggregate_results.csv")
    summary_by_param = {row["parameter"]: row for row in summary}

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#17354b"),
        alignment=TA_CENTER,
        spaceAfter=18,
    )
    h1 = ParagraphStyle(
        "H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#17354b"),
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#24445c"),
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor("#222222"),
        spaceAfter=7,
    )
    small = ParagraphStyle(
        "SmallCustom",
        parent=body,
        fontSize=7.6,
        leading=10,
        spaceAfter=4,
    )
    callout = ParagraphStyle(
        "CalloutCustom",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=15,
        borderWidth=0.8,
        borderColor=colors.HexColor("#9ecae1"),
        borderPadding=9,
        backColor=colors.HexColor("#eef6fb"),
        spaceBefore=7,
        spaceAfter=12,
    )

    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(letter),
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.48 * inch,
        title="Archived local-eligibility parameter landscapes",
        author="Codex analysis for LearningSpikingDynamics",
        subject="One-parameter sampled loss landscapes and archived eligibility-signal consistency checks",
    )
    story: list[Any] = []

    psth_rows = [row for row in summary if row["objective"] == "PSTH SSE"]
    cv_rows = [row for row in summary if row["objective"] == "CV squared error"]

    def family_counts(rows: list[dict[str, str]]) -> tuple[int, int, int, int]:
        green = sum(int(f(row["stable_green_points"])) for row in rows)
        red = sum(int(f(row["stable_red_points"])) for row in rows)
        inconclusive = sum(int(f(row["inconclusive_points"])) for row in rows)
        invalid = sum(int(f(row["invalid_cv_points"])) for row in rows)
        return green, red, inconclusive, invalid

    psth_green, psth_red, psth_inconclusive, _ = family_counts(psth_rows)
    cv_green, cv_red, cv_inconclusive, cv_invalid = family_counts(cv_rows)
    seed_text = ", ".join(str(value) for value in metadata["seeds"])

    story.append(Spacer(1, 0.45 * inch))
    story.append(Paragraph("Archived local-eligibility parameter landscapes", title))
    story.append(
        Paragraph(
            "Full forward sweeps of all 12 learnable parameters in the frozen pre-dPSC July 8/9 implementation",
            ParagraphStyle(
                "Subtitle",
                parent=body,
                fontSize=13,
                leading=18,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#4a4a4a"),
            ),
        )
    )
    story.append(Spacer(1, 0.28 * inch))
    story.append(
        Paragraph(
            f"For the eight PSTH-loss parameters, <b>{psth_green}/{psth_green + psth_red} qualified interior points</b> "
            f"were stable green and {psth_red} were stable red; {psth_inconclusive} additional points were inconclusive. "
            f"For the four CV-loss refractory parameters, <b>{cv_green}/{cv_green + cv_red} qualified points</b> were "
            f"stable green and {cv_red} were stable red; {cv_inconclusive} were inconclusive and {cv_invalid} had an undefined CV stencil.",
            callout,
        )
    )
    story.append(
        Paragraph(
            "This is an exploratory directional-consistency check between the archived eligibility-based update signal and a coarse sampled hard-spike loss curve. "
            "It is not proof of an exact derivative or parameter identifiability. In particular, the archived refractory rule multiplies the CV residual by a refractory accumulator but does not explicitly differentiate CV through interspike intervals and spike times. "
            "Every point is a full 2.9801 s simulation; no Adam update is applied.",
            body,
        )
    )
    overview_data = [
        ["Snapshot", "Saved run", "Cells", "Seeds", "Sweep rows", "Forward cell-runs"],
        [
            "pre-dPSC local snapshot (Jul 9)",
            Path(metadata["checkpoint"]).name,
            ", ".join(str(v) for v in metadata["cells"]),
            ", ".join(str(v) for v in metadata["seeds"]),
            str(metadata["sweep_rows"]),
            str(int(metadata["sweep_rows"]) * len(metadata["seeds"]) * len(metadata["cells"])),
        ],
    ]
    story.append(
        styled_table(
            overview_data,
            [1.85 * inch, 2.05 * inch, 0.55 * inch, 0.95 * inch, 0.72 * inch, 1.05 * inch],
            font_size=7.5,
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph(f"Generated: {metadata['analysis_created']}", small))
    story.append(PageBreak())

    story.append(Paragraph("Methods and interpretation", h1))
    methods = [
        (
            "One parameter at a time",
            "For every parameter, a broad grid spanning its legal or practically relevant range was evaluated. "
            "The other 11 parameters were held at each cell's selected final tracker value from 100_epoch_post_strf_fix_4.mat.",
        ),
        (
            "Fixed cell-specific reference solutions",
            "Cells 1, 2, and 7 use zero-based saved-output batches 6, 3, and 3, respectively. Those batches were selected in-sample from the old run by minimum saved PSTH SSE. "
            "The final tracker slice is used because it is the parameter state that generated the saved raster; the post-final-Adam continuation checkpoint is intentionally not used.",
        ),
        (
            "Common random numbers",
            "Within a stochastic seed, all parameter values receive identical underlying uniforms for onset spikes, offset spikes, spontaneous spikes, and the probabilistic output-spike gate. "
            f"This makes neighboring loss differences much less noisy than independent reruns. Results are averaged over seeds {seed_text}.",
        ),
        (
            "Raw color and qualified criterion",
            "Every retained plotted point has the requested raw face color: green when the mean archived eligibility-based update signal and the adjacent-point sampled slope have the same nonzero sign, red otherwise. "
            "The stricter headline excludes endpoints, zero or turning secants, undefined CV stencils, and any point without unanimous seed-level classification. "
            f"A qualified point is stable green only when every seed has the same nonzero update-signal sign and the same nonzero slope sign and those signs match; stable red requires unanimous but opposite signs. This is descriptive unanimity with {len(metadata['seeds'])} seeds, not a confidence interval.",
        ),
        (
            "Two historical objectives",
            "STRF gain, STRF alpha, output adaptation, and the five conductances use summed PSTH SSE. "
            "The four refractory parameters use the CV squared-error objective because that is how the archived Loss_handler routes their update signals.",
        ),
    ]
    for heading, text in methods:
        story.append(Paragraph(heading, h2))
        story.append(Paragraph(text, body))
    story.append(
        Paragraph(
            "Plot legend: green/red face = the raw sign comparison requested for every point; gray open ring = inconclusive for the qualified analysis; gray X = an undefined CV stencil; purple dashed line = the parameter value that generated the selected saved raster. "
            "The lower panels normalize both signals by the largest absolute value on that cell/parameter curve, so they compare sign and shape rather than units.",
            callout,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Final tracker values held fixed during each sweep", h1))
    baseline_header = ["Parameter", "Cell 1 (batch 6)", "Cell 2 (batch 3)", "Cell 7 (batch 3)"]
    baseline_data = [baseline_header]
    for parameter in PARAMETER_ORDER:
        summary_row = summary_by_param[parameter]
        values = metadata["baseline"][parameter]
        baseline_data.append([summary_row["parameter_label"], *[fmt(value, 5) for value in values]])
    story.append(
        styled_table(
            baseline_data,
            [3.3 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch],
            font_size=8.2,
            alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"},
        )
    )
    story.append(Spacer(1, 0.17 * inch))
    story.append(
        Paragraph(
            "The raw sweep includes each saved-output parameter value itself. A baseline point is omitted from the slope/plot analysis when it is exactly duplicated or lies within 2% of the typical intentional grid spacing from a grid point; every such run remains in raw_results.csv.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Qualified directional-consistency summary", h1))
    summary_data = [["Parameter", "Objective", "Stable G / R", "Qualified %", "Inconcl.", "Invalid CV", "Raw G / all", "Cell 1", "Cell 2", "Cell 7"]]
    for parameter in PARAMETER_ORDER:
        row = summary_by_param[parameter]
        summary_data.append(
            [
                row["parameter_label"],
                row["objective"],
                f"{int(f(row['stable_green_points']))} / {int(f(row['stable_red_points']))}",
                pct(row["qualified_alignment_fraction"]),
                str(int(f(row["inconclusive_points"]))),
                str(int(f(row["invalid_cv_points"]))),
                f"{int(f(row['raw_aligned_points']))} / {int(f(row['raw_points']))}",
                pct(row["cell_1_qualified_alignment_fraction"]),
                pct(row["cell_2_qualified_alignment_fraction"]),
                pct(row["cell_7_qualified_alignment_fraction"]),
            ]
        )
    summary_table = styled_table(
        summary_data,
        [1.75 * inch, 1.15 * inch, 0.72 * inch, 0.72 * inch, 0.55 * inch, 0.62 * inch, 0.72 * inch, 0.58 * inch, 0.58 * inch, 0.58 * inch],
        font_size=7.0,
        alignments={2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT", 7: "RIGHT", 8: "RIGHT", 9: "RIGHT"},
    )
    summary_style = []
    for row_index, row in enumerate(summary[0:] if False else summary_data[1:], start=1):
        overall_text = row[3].rstrip("%")
        if overall_text == "N/A":
            fill = "#e5e5e5"
        else:
            overall = float(overall_text)
            fill = "#e5f5e0" if overall >= 70 else "#fff7bc" if overall >= 50 else "#fee0d2"
        summary_style.append(("BACKGROUND", (3, row_index), (3, row_index), colors.HexColor(fill)))
    summary_table.setStyle(TableStyle(summary_style))
    story.append(summary_table)
    story.append(Spacer(1, 0.16 * inch))
    story.append(
        Paragraph(
            "Qualified percentages use only stable-green plus stable-red points. Inconclusive and invalid-CV points are shown separately and excluded from the denominator. Raw G/all preserves the user's requested binary face-color comparison over every plotted point.",
            body,
        )
    )
    story.append(PageBreak())

    aggregate_by_param: dict[str, list[dict[str, str]]] = {parameter: [] for parameter in PARAMETER_ORDER}
    for row in aggregate:
        aggregate_by_param[row["parameter"]].append(row)

    for parameter in PARAMETER_ORDER:
        summary_row = summary_by_param[parameter]
        plot_path = PLOT_DIR / f"{parameter}_landscape.png"
        if not plot_path.exists():
            raise FileNotFoundError(plot_path)
        story.append(Paragraph(summary_row["parameter_label"], h1))
        story.append(
            Paragraph(
                f"Objective: {summary_row['objective']}. Qualified stable green/red: "
                f"{int(f(summary_row['stable_green_points']))}/{int(f(summary_row['stable_red_points']))}; "
                f"qualified green fraction {pct(summary_row['qualified_alignment_fraction'])}. "
                f"Inconclusive: {int(f(summary_row['inconclusive_points']))}; invalid CV: {int(f(summary_row['invalid_cv_points']))}; "
                f"raw green/all: {int(f(summary_row['raw_aligned_points']))}/{int(f(summary_row['raw_points']))}.",
                small,
            )
        )
        image = Image(str(plot_path))
        image.drawWidth = 9.82 * inch
        image.drawHeight = 6.23 * inch
        story.append(image)
        story.append(PageBreak())

        story.append(Paragraph(f"{summary_row['parameter_label']}: numerical values", h1))
        table_data = [
            [
                "Cell",
                "Parameter value",
                "Loss mean",
                "Update signal",
                "Sampled slope",
                "FR (Hz)",
                "Raw color",
                "Qualified status",
                "CV stencil valid",
            ]
        ]
        detail_rows = sorted(
            aggregate_by_param[parameter],
            key=lambda row: (int(f(row["cell"])), f(row["parameter_value"])),
        )
        for row in detail_rows:
            table_data.append(
                [
                    str(int(f(row["cell"]))),
                    fmt(row["parameter_value"], 5),
                    fmt(row["loss_mean"], 5),
                    fmt(row["gradient_mean"], 4),
                    fmt(row["slope_mean"], 4),
                    fmt(row["firing_rate_mean_hz"], 4),
                    row["classification"].upper(),
                    row["qualified_status"].replace("_", " ").upper(),
                    pct(row["cv_stencil_valid_fraction"]),
                ]
            )
        detail_table = styled_table(
            table_data,
            [0.42 * inch, 1.0 * inch, 0.9 * inch, 1.02 * inch, 1.02 * inch, 0.7 * inch, 0.68 * inch, 1.05 * inch, 0.85 * inch],
            font_size=5.8,
            alignments={0: "RIGHT", 1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT", 7: "RIGHT"},
        )
        class_styles = []
        for row_index, row in enumerate(detail_rows, start=1):
            raw_fill = colors.HexColor(
                "#d9f0d3" if row["classification"] == "green" else "#fdd0c4"
            )
            status_fill = {
                "stable_green": colors.HexColor("#d9f0d3"),
                "stable_red": colors.HexColor("#fdd0c4"),
                "invalid_cv": colors.HexColor("#d9d9d9"),
                "inconclusive": colors.HexColor("#fff7bc"),
            }[row["qualified_status"]]
            class_styles.extend(
                [
                    ("BACKGROUND", (6, row_index), (6, row_index), raw_fill),
                    ("FONTNAME", (6, row_index), (6, row_index), "Helvetica-Bold"),
                    ("BACKGROUND", (7, row_index), (7, row_index), status_fill),
                    ("FONTNAME", (7, row_index), (7, row_index), "Helvetica-Bold"),
                ]
            )
        detail_table.setStyle(TableStyle(class_styles))
        story.append(detail_table)
        story.append(PageBreak())

    story.append(Paragraph("Limitations and what this test can establish", h1))
    limitations = [
        (
            "Sampled hard-spike loss",
            "The forward model contains threshold and stochastic spike events, so the loss is a staircase-like sampled function rather than a globally smooth curve. Common random numbers reduce Monte Carlo noise but do not make the hard-event map differentiable.",
        ),
        (
            "Consistency check, not derivative validation",
            "A stable-green point means the archived update signal has the same direction as the coarse sampled curve under the strict descriptive rule used here. It does not establish an exact derivative or magnitude, nor guarantee that a finite Adam step will lower loss after momentum, normalization, clipping, or parameter interactions.",
        ),
        (
            "One-dimensional slices",
            "Each curve holds the other parameters fixed. It can expose a wrong local direction, flat regions, and boundaries, but it cannot detect compensating parameter combinations or establish joint identifiability of PV/E interaction parameters.",
        ),
        (
            f"Only three cells and {len(metadata['seeds'])} seeds",
            f"The test intentionally uses cells 1, 2, and 7 and {len(metadata['seeds'])} independent stochastic seeds to keep a full 12-parameter analysis tractable. Broader biological claims would require more cells, checkpoint basins, seeds, and held-out stimuli.",
        ),
        (
            "Historical loss split",
            "The refractory update signals are attached to CV squared error in this archived code, whereas all other update signals are attached to PSTH SSE. Comparing a refractory signal to PSTH loss would answer the wrong question for this implementation.",
        ),
        (
            "Refractory signal is heuristic",
            "The archived refractory update is proportional to 2(CVsim - CVdata) times a refractory eligibility accumulator. It does not explicitly carry the derivative of CV through ISIs and event times. Therefore even a stable-green CV point is evidence of directional consistency for this heuristic, not validation of the exact CV derivative.",
        ),
        (
            "Archived bin convention preserved",
            "At a PSTH boundary the archived code updates eligibility at the closing timestep but counts spikes from the preceding interval, excluding that closing timestep. This report preserves that implementation exactly instead of silently fixing it.",
        ),
        (
            "Sparse CV convention",
            "When too few simulated interspike intervals exist, the archived refractory objective contributes zero loss and zero update signal. This report marks an interior point invalid when its center or either neighboring CV is undefined and excludes it from the qualified percentage.",
        ),
        (
            "Coarse adjacent-point slopes",
            "The broad grids are intended to reveal loss-landscape shape. Their adjacent-point slopes are coarse secant diagnostics, especially for logarithmic STRF and conductance grids; they are not converged local finite differences. Endpoints and turning/flat secants are therefore excluded from the qualified result.",
        ),
        (
            "Constrained boundaries",
            "Several selected solutions are at allowed boundaries, including some conductances, absolute refractory periods, relative refractory b values, and relative refractory c values. Boundary points are endpoints of these slices and are excluded from the two-sided qualified comparison.",
        ),
        (
            "In-sample batch selection",
            "Batches were selected using PSTH SSE on the same recording analyzed here. Neither the selected basin nor the directional results are a held-out generalization test.",
        ),
    ]
    for heading, text in limitations:
        story.append(KeepTogether([Paragraph(heading, h2), Paragraph(text, body)]))
    story.append(
        Paragraph(
            "Files accompanying this PDF: raw_results.csv contains every seed/run/cell observation; per_seed_slopes.csv contains adjacent-point slopes and qualification fields before seed averaging; aggregate_results.csv contains plotted values, raw colors, and qualified statuses; alignment_summary.csv contains the report summary; metadata.json contains provenance, grids, baseline values, and classification rules.",
            callout,
        )
    )

    document.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return PDF_PATH


if __name__ == "__main__":
    result = build_report()
    print(result)
