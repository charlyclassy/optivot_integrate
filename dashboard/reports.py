"""Report exports: Excel workbook, CSV bundle and PDF summary."""
from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import config

SECTION_TITLES = {
    "forecast_summary": "Forecast summary",
    "capacity_risk": "Capacity-risk summary",
    "model_performance": "Model performance",
    "scenario_summary": "Scenario summary",
    "history": "Forecast history and outcomes",
    "forecast_detail": "Forecast detail",
}
SHEET_NAMES = {
    "forecast_summary": "Forecast summary", "capacity_risk": "Capacity risk", "model_performance": "Model performance",
    "scenario_summary": "Scenario summary", "history": "History", "forecast_detail": "Forecast detail",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Make a frame export-friendly: readable timestamps and rounded numbers."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        elif pd.api.types.is_timedelta64_dtype(out[col]):
            out[col] = (out[col].dt.total_seconds() / 60).round(0)
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(3)
    return out


def preview(df: pd.DataFrame) -> pd.DataFrame:
    """On-screen version of a report table: same content, numbers to two decimals."""
    out = _clean(df)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(2)
    return out


def _cover(meta: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [(k, str(v)) for k, v in meta.items()], columns=["Item", "Value"]
    )


def build_excel(sections: dict[str, pd.DataFrame], meta: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        _cover(meta).to_excel(writer, sheet_name="Report", index=False)
        for key, frame in sections.items():
            _clean(frame).to_excel(writer, sheet_name=SHEET_NAMES.get(key, key)[:31], index=False)
        for sheet in writer.sheets.values():
            for column in sheet.columns:
                width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
                sheet.column_dimensions[column[0].column_letter].width = min(max(12, width + 2), 48)
            sheet.freeze_panes = "A2"
    return buffer.getvalue()


def build_csv_zip(sections: dict[str, pd.DataFrame], meta: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report_info.csv", _cover(meta).to_csv(index=False))
        for key, frame in sections.items():
            archive.writestr(f"{key}.csv", _clean(frame).to_csv(index=False))
    return buffer.getvalue()


def build_pdf(sections: dict[str, pd.DataFrame], meta: dict, max_rows: int = 40) -> bytes:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=18, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=12, spaceAfter=5,
                        textColor=colors.HexColor("#16202c"))
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#4b5766"))
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=7.5, leading=9)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"{config.PRODUCT_NAME} report", author=config.PRODUCT_NAME)
    story = [Paragraph(f"{config.PRODUCT_NAME}: depot demand and capacity report", title),
             Paragraph(" &nbsp;|&nbsp; ".join(f"<b>{k}:</b> {v}" for k, v in meta.items()), small),
             Spacer(1, 6)]
    width = A4[0] - 30 * mm
    for key, frame in sections.items():
        if key == "forecast_detail":
            continue  # interval-level detail belongs in the Excel / CSV export
        story.append(Paragraph(SECTION_TITLES.get(key, key), h2))
        if frame.empty:
            story.append(Paragraph("No records for the selected depots and dates.", small))
            continue
        shown = _clean(frame).head(max_rows)
        data = [[Paragraph(f"<b>{c}</b>", cell) for c in shown.columns]]
        for row in shown.itertuples(index=False):
            data.append([Paragraph("" if pd.isna(v) else f"{v:,.2f}" if isinstance(v, float) else str(v), cell) for v in row])
        table = Table(data, colWidths=[width / len(shown.columns)] * len(shown.columns), repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5dae1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(table)
        if len(frame) > max_rows:
            story.append(Paragraph(f"Showing the first {max_rows} of {len(frame)} rows. The Excel export contains all rows.", small))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Observed values are real metered depot data. Forecast values are produced by the XGBoost model named above. "
        "Scenario values are hypothetical adjustments of that forecast.", small))
    doc.build(story)
    return buffer.getvalue()


def save_export(content: bytes, stem: str, extension: str, folder: Path | None = None) -> Path:
    folder = folder or config.REPORT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.{extension}"
    path.write_bytes(content)
    return path
