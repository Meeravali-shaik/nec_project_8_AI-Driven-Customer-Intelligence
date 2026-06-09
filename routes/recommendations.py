from flask import render_template, request

from app import app
from analytics.dynamic_engine import (
    generate_recommendations,
    get_form_options,
    parse_profile_input,
    predict_segment,
)


@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    source = request.form if request.method == "POST" else request.args
    profile = parse_profile_input(source)
    segment = predict_segment(profile)
    products = generate_recommendations(profile, segment["segment"])

    return render_template(
        "recommendations.html",
        profile=profile,
        products=products,
        segment=segment,
        options=get_form_options(),
    )
