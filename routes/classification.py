from flask import render_template, request

from app import app
from analytics.dynamic_engine import (
    classify_customer,
    get_form_options,
    parse_profile_input,
    predict_churn,
)


@app.route("/classification", methods=["GET", "POST"])
def classification():
    source = request.form if request.method == "POST" else request.args
    profile = parse_profile_input(source)
    result = classify_customer(profile)
    churn = predict_churn(profile)

    return render_template(
        "classification.html",
        profile=profile,
        result=result,
        churn=churn,
        options=get_form_options(),
    )
