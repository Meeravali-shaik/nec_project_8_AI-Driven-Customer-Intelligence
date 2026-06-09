from flask import render_template, request

from app import app
from analytics.dynamic_engine import (
    generate_cluster_distribution_chart,
    generate_segmentation_statistics,
    get_available_segmentation_features,
    find_similar_customers,
    get_form_options,
    parse_segmentation_config,
    parse_profile_input,
    predict_custom_segment,
    generate_segmentation_chart,
)


@app.route("/segmentation", methods=["GET", "POST"])
def segmentation():
    source = request.form if request.method == "POST" else request.args
    profile = parse_profile_input(source)
    segmentation_config = parse_segmentation_config(source)
    segment_result = predict_custom_segment(
        profile,
        segmentation_config["cluster_features"],
        segmentation_config["cluster_count"],
    )
    similar_customers = find_similar_customers(profile, limit=8)

    return render_template(
        "segmentation.html",
        profile=profile,
        result=segment_result,
        customers=similar_customers.to_dict("records"),
        segment_graph=generate_segmentation_chart(
            profile,
            segmentation_config["cluster_features"],
            segmentation_config["cluster_count"],
        ),
        cluster_distribution=generate_cluster_distribution_chart(
            segmentation_config["cluster_features"],
            segmentation_config["cluster_count"],
        ),
        cluster_stats=generate_segmentation_statistics(
            segmentation_config["cluster_features"],
            segmentation_config["cluster_count"],
        ),
        segmentation_config=segmentation_config,
        available_features=get_available_segmentation_features(),
        options=get_form_options(),
    )
