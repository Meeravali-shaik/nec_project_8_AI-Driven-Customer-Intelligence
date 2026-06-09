from flask import render_template, request

from app import app
from analytics.dynamic_engine import analyze_expiry, get_form_options, parse_profile_input


@app.route("/expiry-monitor", methods=["GET", "POST"])
@app.route("/expiry", methods=["GET", "POST"])
def expiry_monitor():
    source = request.form if request.method == "POST" else request.args
    profile = parse_profile_input(source)
    result = analyze_expiry(profile)

    return render_template(
        "expiry_monitor.html",
        profile=profile,
        result=result,
        options=get_form_options(),
    )
