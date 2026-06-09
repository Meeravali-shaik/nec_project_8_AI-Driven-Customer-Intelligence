from flask import render_template, request

from app import app
from analytics.dynamic_engine import (
    build_customer_summary,
    build_market_snapshot,
    get_form_options,
    parse_filter_input,
    parse_profile_input,
)


@app.route("/behavior-analysis", methods=["GET", "POST"])
def behavior_analysis():
    source = request.form if request.method == "POST" else request.args
    profile = parse_profile_input(source)
    filters = parse_filter_input(request.values)
    snapshot = build_market_snapshot(filters)
    summary = build_customer_summary(profile)
    filtered = snapshot["filtered_df"]

    avg_spending = round(filtered["Purchase_Amount"].mean(), 2) if not filtered.empty else 0
    avg_frequency = round(filtered["Purchase_Frequency"].mean(), 2) if not filtered.empty else 0
    popular_product = (
        filtered["Product_Category"].mode().iloc[0] if not filtered.empty else "No Data"
    )
    active_customer = (
        filtered.loc[filtered["Purchase_Frequency"].idxmax(), "Name"]
        if not filtered.empty
        else "No Data"
    )

    return render_template(
        "behavior_analysis.html",
        profile=profile,
        filters=filters,
        summary=summary,
        avg_spending=avg_spending,
        avg_frequency=avg_frequency,
        popular_product=popular_product,
        active_customer=active_customer,
        charts=snapshot["behavior_charts"],
        options=get_form_options(),
    )
