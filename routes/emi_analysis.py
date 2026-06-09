from flask import render_template, request

from app import app
from analytics.dynamic_engine import analyze_emi, get_form_options, parse_profile_input


@app.route("/emi-analysis", methods=["GET", "POST"])
def emi_analysis():
    source = request.form if request.method == "POST" else request.args
    profile = parse_profile_input(source)
    result = analyze_emi(profile)

    return render_template(
        "emi_analysis.html",
        profile=profile,
        result=result,
        options=get_form_options(),
    )
