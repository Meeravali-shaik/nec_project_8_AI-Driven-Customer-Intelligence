from flask import render_template, request

from app import app
from analytics.data_manager import build_dataset_profile, get_active_dataset_label, load_active_dataset_raw
from analytics.dynamic_engine import (
    build_customer_summary,
    build_market_snapshot,
    get_form_options,
    parse_filter_input,
    parse_profile_input,
)


@app.route("/", methods=["GET", "POST"])
def dashboard():
    source = request.form if request.method == "POST" else request.args
    profile = parse_profile_input(source)
    filters = parse_filter_input(request.values)
    snapshot = build_market_snapshot(filters)
    summary = build_customer_summary(profile)
    dataset_profile = build_dataset_profile(load_active_dataset_raw())

    return render_template(
        "dashboard.html",
        profile=profile,
        filters=filters,
        summary=summary,
        charts=snapshot["charts"],
        customer_count=snapshot["customer_count"],
        avg_income=snapshot["avg_income"],
        avg_purchase=snapshot["avg_purchase"],
        avg_emi=snapshot["avg_emi"],
        high_risk_count=snapshot["high_risk_count"],
        highest_revenue_group=snapshot["highest_revenue_group"],
        most_active_segment=snapshot["most_active_segment"],
        fastest_growing_segment=snapshot["fastest_growing_segment"],
        active_dataset=get_active_dataset_label(),
        dataset_profile=dataset_profile,
        options=get_form_options(),
    )
