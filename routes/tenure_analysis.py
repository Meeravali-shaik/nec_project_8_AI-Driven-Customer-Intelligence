from flask import render_template, request

from app import app
from analytics.dynamic_engine import build_market_snapshot, get_form_options, parse_filter_input


@app.route("/tenure-analysis", methods=["GET", "POST"])
def tenure_analysis():
    filters = parse_filter_input(request.values)
    snapshot = build_market_snapshot(filters)

    return render_template(
        "tenure_analysis.html",
        filters=filters,
        charts=snapshot["tenure_charts"],
        summary=snapshot["tenure_summary"],
        customer_count=snapshot["customer_count"],
        options=get_form_options(),
    )
