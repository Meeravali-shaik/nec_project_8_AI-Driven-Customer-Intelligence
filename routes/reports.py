from io import BytesIO

import pandas as pd
from flask import render_template, request, send_file

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except ImportError:  # pragma: no cover - optional runtime dependency
    A4 = None
    canvas = None

from app import app
from analytics.data_manager import get_active_dataset_label
from analytics.dynamic_engine import build_market_snapshot, get_form_options, parse_filter_input


@app.route("/reports", methods=["GET", "POST"])
def reports():
    filters = parse_filter_input(request.values)
    snapshot = build_market_snapshot(filters)

    return render_template(
        "reports.html",
        filters=filters,
        customer_count=snapshot["customer_count"],
        active_dataset=get_active_dataset_label(),
        options=get_form_options(),
    )


@app.route("/export-csv")
def export_csv():
    filters = parse_filter_input(request.args)
    snapshot = build_market_snapshot(filters)
    csv_data = snapshot["filtered_df"].to_csv(index=False).encode("utf-8")
    buffer = BytesIO(csv_data)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="dynamic_customer_report.csv",
    )


@app.route("/export-excel")
def export_excel():
    filters = parse_filter_input(request.args)
    snapshot = build_market_snapshot(filters)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        snapshot["filtered_df"].to_excel(writer, index=False, sheet_name="Customers")
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="dynamic_customer_report.xlsx",
    )


@app.route("/export-pdf")
def export_pdf():
    filters = parse_filter_input(request.args)
    snapshot = build_market_snapshot(filters)
    dataframe = snapshot["filtered_df"]

    if canvas is None:
        fallback_lines = [
            "AI-Driven Customer Intelligence Report",
            f"Dataset: {get_active_dataset_label()}",
            f"Filtered customers: {snapshot['customer_count']}",
            f"Average income: Rs. {snapshot['avg_income']}",
            f"Average purchase: Rs. {snapshot['avg_purchase']}",
        ]
        buffer = BytesIO("\n".join(fallback_lines).encode("utf-8"))
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype="text/plain",
            as_attachment=True,
            download_name="dynamic_customer_report.txt",
        )

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "AI-Driven Customer Intelligence Report")
    y -= 25
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Dataset: {get_active_dataset_label()}")
    y -= 18
    pdf.drawString(40, y, f"Filtered customers: {snapshot['customer_count']}")
    y -= 18
    pdf.drawString(40, y, f"Average income: Rs. {snapshot['avg_income']}")
    y -= 18
    pdf.drawString(40, y, f"Average purchase: Rs. {snapshot['avg_purchase']}")
    y -= 30

    columns = [column for column in ["Customer_ID", "Name", "City", "Segment", "Purchase_Amount", "Churn_Label"] if column in dataframe.columns]
    preview = dataframe[columns].head(12) if columns else dataframe.head(12)

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(40, y, "Customer Preview")
    y -= 18
    pdf.setFont("Helvetica", 8)
    for row in preview.astype(str).itertuples(index=False):
        line = " | ".join(row)
        pdf.drawString(40, y, line[:110])
        y -= 14
        if y < 60:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 8)

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="dynamic_customer_report.pdf",
    )
