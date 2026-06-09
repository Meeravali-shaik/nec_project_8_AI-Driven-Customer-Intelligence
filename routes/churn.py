from flask import render_template, request

from app import app
from analytics.dynamic_engine import get_form_options, parse_profile_input, predict_churn


@app.route("/churn", methods=["GET", "POST"])
def churn():
    source = request.form if request.method == "POST" else request.args
    profile = parse_profile_input(source)
    result = predict_churn(profile)

    return render_template(
        "churn.html",
        profile=profile,
        result=result,
        options=get_form_options(),
    )
