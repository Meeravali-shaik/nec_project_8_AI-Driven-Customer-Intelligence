from flask import render_template, request

from app import app
from analytics.dynamic_engine import get_form_options, find_similar_customers, parse_profile_input


@app.route("/customers", methods=["GET", "POST"])
def customers():
    source = request.form if request.method == "POST" else request.args
    profile = parse_profile_input(source)
    similar_customers = find_similar_customers(profile)

    return render_template(
        "customer_explorer.html",
        profile=profile,
        customer_count=len(similar_customers),
        customers=similar_customers.to_dict("records"),
        options=get_form_options(),
    )
