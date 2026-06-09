from flask import render_template, request

from app import app
from analytics.dynamic_engine import build_market_snapshot, get_form_options, parse_filter_input


@app.route("/market-analysis", methods=["GET", "POST"])
def market_analysis():
    filters = parse_filter_input(request.values)
    snapshot = build_market_snapshot(filters)

    return render_template(
        "market_analysis.html",
        filters=filters,
        charts=snapshot["charts"],
        customer_count=snapshot["customer_count"],
        options=get_form_options(),
    )
