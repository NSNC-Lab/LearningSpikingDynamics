"""Build the PDF report for the archived STRF-gain local-FD pilot."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
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
PDF_PATH = PDF_DIR / "archived_strf_gain_local_finite_difference_pilot.pdf"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: Any, digits: int = 3) -> str:
    value = number(value)
    if not math.isfinite(value):
        return "N/A"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def pct(value: Any, digits: int = 1) -> str:
    value = number(value)
    return "N/A" if not math.isfinite(value) else f"{100.0 * value:.{digits}f}%"


def page_decor(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    width, height = landscape(letter)
    canvas.setStrokeColor(colors.HexColor("#d9d9d9"))
    canvas.setLineWidth(0.5)
    canvas.line(0.55 * inch, height - 0.42 * inch, width - 0.55 * inch, height - 0.42 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(0.58 * inch, height - 0.31 * inch, "Archived STRF-gain local finite-difference pilot")
    canvas.drawRightString(width - 0.58 * inch, 0.28 * inch, f"Page {doc.page}")
    canvas.restoreState()


def styled_table(
    data: list[list[Any]],
    col_widths: list[float],
    font_size: float = 7.0,
    status_column: int | None = None,
) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    style: list[tuple[Any, ...]] = [
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
        if status_column is not None:
            status = str(data[row_index][status_column]).lower()
            fill = {
                "green": colors.HexColor("#d9f0d3"),
                "red": colors.HexColor("#fcbba1"),
                "gray": colors.HexColor("#e5e5e5"),
            }.get(status, colors.white)
            style.append(("BACKGROUND", (status_column, row_index), (status_column, row_index), fill))
            style.append(("FONTNAME", (status_column, row_index), (status_column, row_index), "Helvetica-Bold"))
    table.setStyle(TableStyle(style))
    return table


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return image


def median(values: list[float]) -> float:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return math.nan
    midpoint = len(values) // 2
    return values[midpoint] if len(values) % 2 else 0.5 * (values[midpoint - 1] + values[midpoint])


def build_report() -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    epsilon_summary = read_csv(DATA_DIR / "epsilon_summary.csv")
    aggregate = read_csv(DATA_DIR / "aggregate_finite_differences.csv")
    primary_epsilon = float(metadata["predeclared_primary_log_epsilon"])
    primary = [row for row in aggregate if math.isclose(number(row["epsilon"]), primary_epsilon)]
    primary_summary = next(
        row for row in epsilon_summary if math.isclose(number(row["epsilon"]), primary_epsilon)
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=27,
        textColor=colors.HexColor("#17354b"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    subtitle = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        textColor=colors.HexColor("#4a4a4a"),
        alignment=TA_CENTER,
        spaceAfter=10,
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
        spaceBefore=6,
        spaceAfter=10,
    )

    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(letter),
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.48 * inch,
        title="Archived STRF-gain local finite-difference pilot",
        author="Codex analysis for LearningSpikingDynamics",
        subject="Local paired finite differences for the archived STRF-gain eligibility signal",
    )
    story: list[Any] = []

    primary_green = sum(row["status"] == "green" for row in primary)
    primary_red = sum(row["status"] == "red" for row in primary)
    primary_gray = sum(row["status"] == "gray" for row in primary)

    # Cover.
    story.append(Spacer(1, 0.42 * inch))
    story.append(Paragraph("Archived STRF-gain local finite-difference pilot", title))
    story.append(
        Paragraph(
            "Random full-parameter contexts, small paired gain perturbations, and expected sampled-loss slopes",
            subtitle,
        )
    )
    story.append(
        Paragraph(
            f"At the predeclared <b>+/-{100 * primary_epsilon:g}% log-gain perturbation</b>, "
            f"<b>{primary_green}/24 contexts were direction-qualified green</b>, {primary_red} were red, "
            f"and {primary_gray} were unresolved. All 24 mean eligibility and finite-difference signs "
            f"agreed, but only the 21 uncertainty-resolved, stable-sign-across-epsilon comparisons count in the headline.",
            callout,
        )
    )
    story.append(
        Paragraph(
            f"Direction was much stronger than magnitude calibration. At 2%, Spearman association was "
            f"<b>{fmt(primary_summary['spearman_all'])}</b> and cosine alignment was "
            f"<b>{fmt(primary_summary['cosine_all'])}</b>. The median resolved FD/eligibility magnitude ratio "
            f"was <b>{fmt(primary_summary['median_fd_to_analytic_ratio_both_resolved'])}</b>, while a through-origin "
            f"fit gave <b>{fmt(primary_summary['through_origin_fd_on_analytic'])}</b>. Thus this pilot supports the "
            f"archived STRF-gain update's direction in the sampled regime, but not equality to the numerical gradient.",
            body,
        )
    )
    overview = [
        ["Snapshot", "Contexts", "Seeds", "Trials/context", "Gain rows", "Forward cell-runs"],
        [
            "pre-dPSC local snapshot (Jul 9)",
            "8 LHS batches x 3 cells = 24",
            ", ".join(str(value) for value in metadata["seeds"]),
            str(len(metadata["seeds"]) * metadata["trials_per_seed"]),
            "center + 12 endpoints + duplicate",
            str(len(metadata["seeds"]) * metadata["batch_rows"] * len(metadata["cells"])),
        ],
    ]
    story.append(
        styled_table(
            overview,
            [1.85 * inch, 2.0 * inch, 2.2 * inch, 1.0 * inch, 1.8 * inch, 0.95 * inch],
            7.3,
        )
    )
    story.append(Spacer(1, 0.10 * inch))
    story.append(
        Paragraph(
            "Interpretation: each hard-spike trajectory has a discontinuous sampled loss. The numerical quantity here is a paired Monte Carlo central secant of expected sampled PSTH SSE at finite epsilon. It approaches an ordinary derivative only if that expectation is differentiable and the estimates converge as epsilon shrinks.",
            small,
        )
    )
    story.append(Paragraph(f"Generated: {metadata['generated_at']}", small))
    story.append(PageBreak())

    # Methods.
    story.append(Paragraph("What was tested", h1))
    story.append(
        Paragraph(
            "For each cell-specific Latin-hypercube parameter context, all 11 nuisance parameters were held fixed and STRF gain was evaluated at g0 and at symmetric log-space probes:",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>g+ = g0 exp(h), &nbsp;&nbsp; g- = g0 exp(-h)</b>, where h = 0.0025, 0.005, 0.01, 0.02, 0.05, or 0.10.",
            callout,
        )
    )
    story.append(
        Paragraph(
            "For seed s, the paired numerical log-gain slope is D(s,h) = [L(g+;s) - L(g-;s)] / log(g+/g-). The archived center eligibility is expressed in the same coordinate as A(s) = g0 x dLelig/dg. Means and nominal 95% t intervals are calculated across eight seed-level paired estimates; each seed contains ten model trials.",
            body,
        )
    )
    method_rows = [
        ["Design element", "Implementation"],
        ["Parameter contexts", "Separate scrambled Latin-hypercube design for cells 1, 2, and 7 over archived initializer ranges"],
        ["Gain boundary guard", "Gain centers lower-truncated at 0.001 x exp(0.10), so every -10% log probe remains >= 0.001"],
        ["Paired randomness", "Onset, offset, and output-gate torch draws shared only within a center's 14 rows"],
        ["Independent stochastic replication", "Eight sequential seeds; 10 trials per seed"],
        ["Spontaneous input", "Archived numpy spontaneous stream has no batch axis and is shared across centers within each seed"],
        ["CRN integrity control", "A duplicate center row must match loss, eligibility, spike count, and firing rate bit-for-bit"],
        ["Direction qualification", "Eligibility interval excludes zero and FD belongs to a resolved same-sign run across 3 adjacent epsilons"],
        ["Magnitude stability", "Reported separately when 3 adjacent resolved FD intervals have a common intersection"],
        ["Primary analysis", "+/-2% log gain, predeclared before final aggregation; all other epsilons are sensitivity checks"],
    ]
    story.append(styled_table(method_rows, [2.0 * inch, 8.0 * inch], 7.4))
    story.append(Spacer(1, 0.10 * inch))
    story.append(
        Paragraph(
            "Green/red is an exploratory directional label, not a proof of exact differentiation. The t intervals are nominal, the hard-event differences need not be Gaussian, and no multiple-testing correction is applied.",
            small,
        )
    )
    story.append(PageBreak())

    # Epsilon summary.
    story.append(Paragraph("Epsilon sensitivity", h1))
    story.append(scaled_image(PLOT_DIR / "epsilon_summary.png", 9.8 * inch, 5.2 * inch))
    summary_table: list[list[Any]] = [
        ["h", "FD resolved", "Stable sign", "FD magnitude-stable", "Green", "Red", "Gray", "Raw sign", "Spearman", "FD/elig median"],
    ]
    for row in epsilon_summary:
        summary_table.append(
            [
                f"{number(row['epsilon_percent_log']):g}%",
                f"{row['fd_resolved']}/24",
                f"{row['stable_sign_window']}/24",
                f"{row['magnitude_stable_window']}/24",
                row["green"],
                row["red"],
                row["gray"],
                pct(row["raw_alignment_fraction"]),
                fmt(row["spearman_all"]),
                fmt(row["median_fd_to_analytic_ratio_both_resolved"]),
            ]
        )
    story.append(
        styled_table(
            summary_table,
            [0.55 * inch, 0.85 * inch, 0.9 * inch, 1.15 * inch, 0.55 * inch, 0.45 * inch, 0.5 * inch, 0.75 * inch, 0.7 * inch, 0.9 * inch],
            6.5,
        )
    )
    story.append(PageBreak())

    # Per-cell primary summary.
    story.append(Paragraph("Predeclared 2% result by cell", h1))
    per_cell_rows: list[list[Any]] = [
        ["Cell", "Contexts", "Green", "Red", "Gray", "Median FD/elig", "FR min", "FR median", "FR max"]
    ]
    for cell in metadata["cells"]:
        rows = [row for row in primary if int(number(row["cell"])) == int(cell)]
        ratios = [
            number(row["fd_log_mean"]) / number(row["analytic_log_mean"])
            for row in rows
            if number(row["analytic_log_mean"]) != 0
        ]
        firing_rates = [number(row["center_firing_rate_mean_hz"]) for row in rows]
        per_cell_rows.append(
            [
                str(cell),
                str(len(rows)),
                str(sum(row["status"] == "green" for row in rows)),
                str(sum(row["status"] == "red" for row in rows)),
                str(sum(row["status"] == "gray" for row in rows)),
                fmt(median(ratios)),
                f"{min(firing_rates):.1f} Hz",
                f"{median(firing_rates):.1f} Hz",
                f"{max(firing_rates):.1f} Hz",
            ]
        )
    story.append(
        styled_table(
            per_cell_rows,
            [0.55 * inch, 0.75 * inch, 0.6 * inch, 0.5 * inch, 0.55 * inch, 1.15 * inch, 0.85 * inch, 0.9 * inch, 0.85 * inch],
            7.5,
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("Interpretation", h2))
    story.append(
        Paragraph(
            "Cells 2 and 7 resolved all eight parameter contexts as green at 2%. Cell 1 resolved five as green and left three gray; none were direction-qualified red. The sole negative-gradient context had a negative finite-difference slope as well. This is substantially more informative than the broad sweep because each comparison is local and paired.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The random initializer produced high firing rates in this pilot (about 24-305 Hz across centers). These are legitimate tests of the original initialization distribution, but they are not a representative sample of fitted or biologically plausible regimes. A trained-state stratum remains necessary before generalizing the result to final inverse-model parameters.",
            callout,
        )
    )
    story.append(PageBreak())

    def add_plot_page(title_text: str, filename: str, caption: str, max_height: float = 5.75 * inch) -> None:
        story.append(Paragraph(title_text, h1))
        story.append(scaled_image(PLOT_DIR / filename, 9.9 * inch, max_height))
        story.append(Spacer(1, 0.05 * inch))
        story.append(Paragraph(caption, small))
        story.append(PageBreak())

    add_plot_page(
        "Directional agreement map",
        "status_heatmap.png",
        "Green requires an uncertainty-resolved archived eligibility and a numerical sign belonging to a three-epsilon resolved same-sign window. Gray means unresolved; it is not counted as incorrect. No red context was observed.",
        6.0 * inch,
    )
    add_plot_page(
        "Eligibility versus finite-difference slope",
        "gradient_scatter_by_epsilon.png",
        "The panels retain all 24 contexts. The identity line tests magnitude as well as direction. Most resolved points lie in the same sign quadrant but well below the identity line, demonstrating directional agreement without magnitude equality.",
    )
    add_plot_page(
        "Magnitude calibration",
        "magnitude_calibration.png",
        "The context-level ratio is heterogeneous. At the primary 2% scale, the median resolved FD/eligibility ratio is about 0.16, but the through-origin scale is about 0.019 because the largest archived eligibility values dominate that fit. Neither statistic is close to one.",
    )
    add_plot_page(
        "Ensemble of conditional local landscapes",
        "ensemble_local_landscapes.png",
        "Each curve is one conditional one-dimensional slice: only gain changes, while the other 11 parameters remain fixed. Curves are expressed as loss change relative to their own center loss; different centers are not connected into a false global landscape.",
    )

    for cell in metadata["cells"]:
        add_plot_page(
            f"Cell {cell}: individual local loss slices",
            f"local_landscapes_cell_{cell}.png",
            "The shaded band is the nominal 95% seed-level interval for paired loss change. Point colors apply to the symmetric +/- pair at that epsilon. Orange text gives only the archived center direction so the actual loss curve remains visually readable.",
        )

    for cell in metadata["cells"]:
        add_plot_page(
            f"Cell {cell}: finite-difference epsilon sensitivity",
            f"fd_convergence_cell_{cell}.png",
            "Blue points and intervals are paired central secants of expected sampled loss. The orange dashed line and band show archived center eligibility. Stable numerical slopes across epsilon support a local directional conclusion, while vertical separation from orange shows magnitude miscalibration.",
        )

    # Full primary table.
    story.append(Paragraph("All 24 primary comparisons", h1))
    primary_rows: list[list[Any]] = [
        ["Cell", "Init", "g0", "FR", "FD dL/dlog(g)", "Elig dL/dlog(g)", "FD/elig", "FD 95% CI", "Elig 95% CI", "Status"],
    ]
    for row in sorted(primary, key=lambda value: (int(number(value["cell"])), int(number(value["init_id"])))):
        analytic = number(row["analytic_log_mean"])
        ratio = number(row["fd_log_mean"]) / analytic if analytic != 0 else math.nan
        primary_rows.append(
            [
                str(int(number(row["cell"]))),
                str(int(number(row["init_id"]))),
                fmt(row["center_gain"], 4),
                f"{number(row['center_firing_rate_mean_hz']):.1f}",
                fmt(row["fd_log_mean"], 4),
                fmt(row["analytic_log_mean"], 4),
                fmt(ratio, 4),
                f"[{fmt(row['fd_log_ci_low'], 3)}, {fmt(row['fd_log_ci_high'], 3)}]",
                f"[{fmt(row['analytic_log_ci_low'], 3)}, {fmt(row['analytic_log_ci_high'], 3)}]",
                row["status"],
            ]
        )
    story.append(
        styled_table(
            primary_rows,
            [0.38 * inch, 0.38 * inch, 0.62 * inch, 0.48 * inch, 0.92 * inch, 0.98 * inch, 0.62 * inch, 1.38 * inch, 1.38 * inch, 0.68 * inch],
            5.9,
            status_column=9,
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            "FR is the center firing rate in Hz. Gradients use the log-gain coordinate so signs and units are directly comparable.",
            small,
        )
    )
    story.append(PageBreak())

    # Conclusions.
    story.append(Paragraph("Conclusions and limitations", h1))
    conclusions = [
        "<b>The concern about the first report was valid.</b> Its broad neighboring points measured long secants and created many inconclusive classifications. Small paired perturbations provide much stronger local evidence for STRF gain.",
        "<b>Direction is robust in this pilot.</b> At the predeclared 2% scale, 21/24 contexts were qualified green, none were qualified red, and all 24 mean signs agreed. Results were similar from 1% through 10%.",
        "<b>Magnitude is not validated.</b> Numerical FD values are generally smaller than archived eligibility values and the calibration ratio varies substantially by context. This can arise from surrogate-rule scaling, missing normalization, omitted paths, or hard-event/expected-loss differences; this experiment alone does not identify which.",
        "<b>This is an initialization-distribution test.</b> Eight Latin-hypercube batches sparsely cover a 12-dimensional space, and their firing rates are often high. The test should next be repeated around fitted checkpoints and biologically relevant firing-rate regimes.",
        "<b>The uncertainty analysis is exploratory.</b> Eight seed-level observations give low-power nominal t intervals for a discrete, potentially heavy-tailed difference. Epsilon tests are correlated, contexts share the seed's spontaneous input, and no multiplicity correction is applied.",
        "<b>This does not test a frozen-event state derivative.</b> A separate voltage/rate sensitivity finite difference with event sequences held fixed is still the cleanest way to isolate chain-rule implementation from the stochastic spike objective.",
    ]
    for index, text in enumerate(conclusions, start=1):
        story.append(Paragraph(f"{index}. {text}", body))
    story.append(Spacer(1, 0.10 * inch))
    story.append(
        Paragraph(
            "Bottom line: the archived STRF-gain eligibility appears to point downhill/uphill correctly across the sampled local contexts, but the analysis does not support calling it the exact gradient of expected PSTH SSE. The next decisive experiment is the same paired epsilon ladder around trained parameter states, followed by an explicit investigation of the magnitude scale discrepancy.",
            callout,
        )
    )

    document.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return PDF_PATH


if __name__ == "__main__":
    print(build_report())
