from flask import flash, redirect, render_template, request, url_for

from app import app
from analytics.data_manager import (
    build_dataset_profile,
    get_active_dataset_label,
    load_active_dataset_raw,
    reset_to_default_dataset,
    save_uploaded_dataset,
)


@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "reset":
            reset_to_default_dataset()
            flash("Switched back to the default sample dataset.", "success")
            return redirect(url_for("dataset"))

        upload = request.files.get("dataset_file")
        if not upload or not upload.filename:
            flash("Choose a CSV or XLSX file before uploading.", "danger")
            return redirect(url_for("dataset"))

        try:
            saved_path = save_uploaded_dataset(upload)
        except ValueError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("dataset"))

        flash(f"Dataset uploaded successfully: {saved_path.name}", "success")
        return redirect(url_for("dataset"))

    raw_df = load_active_dataset_raw()
    profile = build_dataset_profile(raw_df)
    return render_template(
        "dataset.html",
        dataset_profile=profile,
        active_dataset=get_active_dataset_label(),
    )
