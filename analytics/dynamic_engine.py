from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from analytics.data_manager import build_dataset_profile, get_dataset_signature, load_active_dataset_raw

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - optional runtime dependency
    px = None


PROFILE_DEFAULTS = {
    "Age": 30,
    "Annual_Income": 700000,
    "Monthly_Income": 58000,
    "EMI": 15000,
    "Gender": "Unknown",
    "City": "Unknown",
    "Occupation": "Unknown",
    "Product_Category": "General",
    "Purchase_Frequency": 6,
    "Tenure_Months": 24,
    "Product_Expiry_Days": 30,
    "Customer_Satisfaction": 7,
    "Website_Visits": 12,
    "Last_Purchase_Days": 45,
}

FILTER_DEFAULTS = {
    "age_min": "",
    "age_max": "",
    "income_min": "",
    "income_max": "",
    "city": "",
    "gender": "",
}

CATEGORICAL_FIELDS = [
    "Gender",
    "City",
    "Occupation",
    "Product_Category",
    "Discount_Usage",
    "Last_Product",
]

NUMERIC_FIELDS = [
    "Age",
    "Annual_Income",
    "Monthly_Income",
    "EMI",
    "Dependents",
    "Tenure_Months",
    "Purchase_Amount",
    "Purchase_Frequency",
    "Total_Purchases",
    "Avg_Order_Value",
    "Last_Purchase_Days",
    "Product_Expiry_Days",
    "Customer_Satisfaction",
    "Website_Visits",
    "Email_Clicks",
    "Churn_Status",
]

SEGMENT_FEATURES = [
    "Age",
    "Annual_Income",
    "Purchase_Amount",
    "Purchase_Frequency",
]

PURCHASE_FEATURES = [
    "Age",
    "Annual_Income",
    "EMI",
    "Purchase_Frequency",
    "Website_Visits",
    "Gender",
    "City",
    "Occupation",
    "Product_Category",
    "Customer_Satisfaction",
    "Tenure_Months",
]

CHURN_FEATURES = [
    "Tenure_Months",
    "Last_Purchase_Days",
    "Customer_Satisfaction",
    "Purchase_Frequency",
    "Product_Expiry_Days",
    "EMI",
    "Annual_Income",
    "Gender",
    "City",
    "Occupation",
    "Product_Category",
]

CLASSIFICATION_FEATURES = [
    "Age",
    "Annual_Income",
    "EMI",
    "Purchase_Frequency",
    "Customer_Satisfaction",
    "Tenure_Months",
    "Last_Purchase_Days",
    "Gender",
    "City",
    "Occupation",
    "Product_Category",
]


def classify_emi_ratio(ratio: float) -> str:
    if ratio >= 50:
        return "High Risk"
    if ratio >= 30:
        return "Medium Risk"
    return "Low Risk"


def classify_expiry(days: float) -> str:
    if days <= 15:
        return "Critical"
    if days <= 30:
        return "Warning"
    return "Safe"


def classify_risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def _safe_mode(series: pd.Series, fallback: str) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return fallback
    return str(non_null.mode().iloc[0])


def _safe_numeric(series: pd.Series, fallback: int | float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float(fallback)
    return float(numeric.median())


def _chart_placeholder(title: str, subtitle: str = "") -> str:
    detail = f"<p>{subtitle}</p>" if subtitle else ""
    return (
        '<div class="plotly-fallback" style="padding:1.25rem;border:1px solid #2b3442;'
        'border-radius:16px;background:linear-gradient(180deg,#0f172a,#111827);color:#e5e7eb;">'
        f"<h4 style='margin:0 0 .5rem 0;'>{title}</h4>{detail}"
        "</div>"
    )


def _to_chart_html(builder, title: str, subtitle: str = "") -> str:
    if px is None:
        return _chart_placeholder(title, subtitle)
    return builder().to_html(full_html=False)


def _coerce_binary(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(
        {
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "at risk": 1,
            "stable": 0,
            "high": 1,
            "low": 0,
        }
    )
    return mapped.fillna(0).astype(int)


def _ensure_column(df: pd.DataFrame, column: str, value) -> None:
    if column not in df.columns:
        df[column] = value


@lru_cache(maxsize=4)
def load_dataset(signature: str | None = None) -> pd.DataFrame:
    _ = signature or get_dataset_signature()
    raw = load_active_dataset_raw().copy()
    profile = build_dataset_profile(raw)
    mapping = profile["field_mapping"]

    for canonical, source in mapping.items():
        if canonical not in raw.columns and source in raw.columns:
            raw[canonical] = raw[source]

    _ensure_column(raw, "Customer_ID", [f"C{index + 1:03d}" for index in range(len(raw))])
    _ensure_column(raw, "Name", [f"Customer {index + 1}" for index in range(len(raw))])
    _ensure_column(raw, "Age", PROFILE_DEFAULTS["Age"])
    _ensure_column(raw, "Annual_Income", PROFILE_DEFAULTS["Annual_Income"])
    _ensure_column(raw, "Monthly_Income", np.nan)
    _ensure_column(raw, "EMI", 0)
    _ensure_column(raw, "Dependents", 0)
    _ensure_column(raw, "Tenure_Months", PROFILE_DEFAULTS["Tenure_Months"])
    _ensure_column(raw, "Product_Category", PROFILE_DEFAULTS["Product_Category"])
    _ensure_column(raw, "Last_Product", raw["Product_Category"])
    _ensure_column(raw, "Purchase_Amount", 0)
    _ensure_column(raw, "Purchase_Frequency", PROFILE_DEFAULTS["Purchase_Frequency"])
    _ensure_column(raw, "Total_Purchases", raw["Purchase_Frequency"] * 4)
    _ensure_column(raw, "Avg_Order_Value", np.nan)
    _ensure_column(raw, "Last_Purchase_Days", PROFILE_DEFAULTS["Last_Purchase_Days"])
    _ensure_column(raw, "Product_Expiry_Days", PROFILE_DEFAULTS["Product_Expiry_Days"])
    _ensure_column(raw, "Customer_Satisfaction", PROFILE_DEFAULTS["Customer_Satisfaction"])
    _ensure_column(raw, "Website_Visits", PROFILE_DEFAULTS["Website_Visits"])
    _ensure_column(raw, "Email_Clicks", 0)
    _ensure_column(raw, "Discount_Usage", "No")
    _ensure_column(raw, "Churn_Status", 0)
    _ensure_column(raw, "Next_Purchase_Category", raw["Product_Category"])
    _ensure_column(raw, "Gender", PROFILE_DEFAULTS["Gender"])
    _ensure_column(raw, "City", PROFILE_DEFAULTS["City"])
    _ensure_column(raw, "Occupation", PROFILE_DEFAULTS["Occupation"])

    for column in NUMERIC_FIELDS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw["Annual_Income"] = raw["Annual_Income"].fillna(_safe_numeric(raw["Annual_Income"], 700000))
    raw["Monthly_Income"] = raw["Monthly_Income"].fillna(raw["Annual_Income"] / 12)
    raw["EMI"] = raw["EMI"].fillna(0)
    raw["Age"] = raw["Age"].fillna(_safe_numeric(raw["Age"], 30))
    raw["Purchase_Frequency"] = raw["Purchase_Frequency"].fillna(_safe_numeric(raw["Purchase_Frequency"], 4))
    raw["Tenure_Months"] = raw["Tenure_Months"].fillna(_safe_numeric(raw["Tenure_Months"], 12))
    raw["Product_Expiry_Days"] = raw["Product_Expiry_Days"].fillna(
        _safe_numeric(raw["Product_Expiry_Days"], 30)
    )
    raw["Customer_Satisfaction"] = raw["Customer_Satisfaction"].fillna(
        _safe_numeric(raw["Customer_Satisfaction"], 7)
    )
    raw["Website_Visits"] = raw["Website_Visits"].fillna(_safe_numeric(raw["Website_Visits"], 10))
    raw["Last_Purchase_Days"] = raw["Last_Purchase_Days"].fillna(
        _safe_numeric(raw["Last_Purchase_Days"], 45)
    )
    raw["Purchase_Amount"] = raw["Purchase_Amount"].fillna(
        _safe_numeric(raw["Purchase_Amount"], 0)
    )
    raw["Total_Purchases"] = raw["Total_Purchases"].fillna(raw["Purchase_Frequency"] * 4)
    raw["Avg_Order_Value"] = raw["Avg_Order_Value"].fillna(
        raw["Purchase_Amount"] / np.where(raw["Purchase_Frequency"] > 0, raw["Purchase_Frequency"], 1)
    )
    raw["Dependents"] = raw["Dependents"].fillna(0)
    raw["Email_Clicks"] = raw["Email_Clicks"].fillna(0)
    raw["Churn_Status"] = _coerce_binary(raw["Churn_Status"])

    for column in CATEGORICAL_FIELDS + ["Next_Purchase_Category"]:
        raw[column] = raw[column].fillna("Unknown").astype(str)

    raw["Gender"] = raw["Gender"].replace({"nan": "Unknown"})
    raw["City"] = raw["City"].replace({"nan": "Unknown"})
    raw["Occupation"] = raw["Occupation"].replace({"nan": "Unknown"})

    raw["Monthly_Income"] = raw["Monthly_Income"].mask(
        raw["Monthly_Income"].isna() | (raw["Monthly_Income"] == 0),
        raw["Annual_Income"] / 12,
    )
    raw["EMI_Ratio"] = np.where(
        raw["Monthly_Income"] > 0,
        (raw["EMI"] / raw["Monthly_Income"]) * 100,
        0,
    )
    raw["EMI_Risk"] = raw["EMI_Ratio"].apply(classify_emi_ratio)
    raw["Expiry_Status"] = raw["Product_Expiry_Days"].apply(classify_expiry)
    raw["Churn_Label"] = raw["Churn_Status"].map({1: "At Risk", 0: "Stable"})
    raw["Income_Group"] = pd.cut(
        raw["Annual_Income"],
        bins=[-np.inf, 400000, 900000, np.inf],
        labels=["Budget", "Core", "Premium"],
    ).astype(str)
    raw["Risk_Category"] = np.select(
        [
            (raw["Churn_Status"] == 1) | (raw["EMI_Ratio"] >= 50),
            (raw["EMI_Ratio"] >= 30) | (raw["Product_Expiry_Days"] <= 30),
        ],
        ["High", "Medium"],
        default="Low",
    )
    return raw


def get_form_options() -> dict[str, list[str]]:
    df = load_dataset(get_dataset_signature())
    return {
        "cities": sorted(df["City"].dropna().astype(str).unique().tolist()),
        "genders": sorted(df["Gender"].dropna().astype(str).unique().tolist()),
        "occupations": sorted(df["Occupation"].dropna().astype(str).unique().tolist()),
        "categories": sorted(df["Product_Category"].dropna().astype(str).unique().tolist()),
    }


def get_available_segmentation_features() -> list[str]:
    df = load_dataset(get_dataset_signature())
    candidates = [
        "Age",
        "Annual_Income",
        "Monthly_Income",
        "EMI",
        "Purchase_Amount",
        "Purchase_Frequency",
        "Tenure_Months",
        "Customer_Satisfaction",
        "Website_Visits",
        "Last_Purchase_Days",
    ]
    return [column for column in candidates if column in df.columns]


def get_default_profile() -> dict[str, object]:
    df = load_dataset(get_dataset_signature())
    profile = PROFILE_DEFAULTS.copy()
    for key in [
        "Age",
        "Annual_Income",
        "Monthly_Income",
        "EMI",
        "Purchase_Frequency",
        "Tenure_Months",
        "Product_Expiry_Days",
        "Customer_Satisfaction",
        "Website_Visits",
        "Last_Purchase_Days",
    ]:
        profile[key] = int(_safe_numeric(df[key], profile[key]))

    for key in ["Gender", "City", "Occupation", "Product_Category"]:
        profile[key] = _safe_mode(df[key], str(profile[key]))
    return profile


def _read_number(source, key: str, default: int | float) -> int:
    raw_value = source.get(key, default)
    if raw_value in (None, ""):
        return int(default)
    try:
        return int(float(raw_value))
    except (TypeError, ValueError):
        return int(default)


def parse_profile_input(source) -> dict[str, object]:
    defaults = get_default_profile()
    profile = {
        "Age": _read_number(source, "Age", defaults["Age"]),
        "Annual_Income": _read_number(source, "Annual_Income", defaults["Annual_Income"]),
        "Monthly_Income": _read_number(source, "Monthly_Income", defaults["Monthly_Income"]),
        "EMI": _read_number(source, "EMI", defaults["EMI"]),
        "Gender": source.get("Gender", defaults["Gender"]) or defaults["Gender"],
        "City": source.get("City", defaults["City"]) or defaults["City"],
        "Occupation": source.get("Occupation", defaults["Occupation"]) or defaults["Occupation"],
        "Product_Category": source.get("Product_Category", defaults["Product_Category"])
        or defaults["Product_Category"],
        "Purchase_Frequency": _read_number(
            source, "Purchase_Frequency", defaults["Purchase_Frequency"]
        ),
        "Tenure_Months": _read_number(source, "Tenure_Months", defaults["Tenure_Months"]),
        "Product_Expiry_Days": _read_number(
            source, "Product_Expiry_Days", defaults["Product_Expiry_Days"]
        ),
        "Customer_Satisfaction": _read_number(
            source, "Customer_Satisfaction", defaults["Customer_Satisfaction"]
        ),
        "Website_Visits": _read_number(source, "Website_Visits", defaults["Website_Visits"]),
        "Last_Purchase_Days": _read_number(
            source, "Last_Purchase_Days", defaults["Last_Purchase_Days"]
        ),
    }
    if not profile["Monthly_Income"]:
        profile["Monthly_Income"] = int(profile["Annual_Income"] / 12)
    return profile


def parse_filter_input(source) -> dict[str, object]:
    filters = FILTER_DEFAULTS.copy()
    for key in ["age_min", "age_max", "income_min", "income_max"]:
        raw_value = source.get(key, "")
        if raw_value in ("", None):
            filters[key] = ""
            continue
        try:
            filters[key] = int(float(raw_value))
        except (TypeError, ValueError):
            filters[key] = ""
    filters["city"] = source.get("city", "") or ""
    filters["gender"] = source.get("gender", "") or ""
    return filters


def parse_segmentation_config(source) -> dict[str, object]:
    available = get_available_segmentation_features()
    selected = source.getlist("cluster_features") if hasattr(source, "getlist") else None
    if not selected:
        raw_selected = source.get("cluster_features", "")
        if raw_selected:
            selected = [item for item in str(raw_selected).split(",") if item]
    selected = [feature for feature in (selected or SEGMENT_FEATURES) if feature in available]
    if len(selected) < 2:
        selected = available[: min(4, len(available))]
    cluster_count = _read_number(source, "cluster_count", 4)
    cluster_count = max(2, min(cluster_count, 6))
    return {"cluster_features": selected, "cluster_count": cluster_count}


def filter_customers(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    filtered = df.copy()
    if filters.get("age_min") != "":
        filtered = filtered[filtered["Age"] >= filters["age_min"]]
    if filters.get("age_max") != "":
        filtered = filtered[filtered["Age"] <= filters["age_max"]]
    if filters.get("income_min") != "":
        filtered = filtered[filtered["Annual_Income"] >= filters["income_min"]]
    if filters.get("income_max") != "":
        filtered = filtered[filtered["Annual_Income"] <= filters["income_max"]]
    if filters.get("city"):
        filtered = filtered[filtered["City"] == filters["city"]]
    if filters.get("gender"):
        filtered = filtered[filtered["Gender"] == filters["gender"]]
    return filtered


def _build_preprocessor(features: list[str], df: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    categorical_features = [column for column in features if column not in numeric_features]
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def _fit_classifier(df: pd.DataFrame, features: list[str], target: str, estimator) -> Pipeline:
    preprocess = _build_preprocessor(features, df)
    model = estimator if df[target].nunique() > 1 else DummyClassifier(strategy="most_frequent")
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipeline.fit(df[features], df[target])
    return pipeline


def _derive_segment_labels(centers: pd.DataFrame) -> dict[int, str]:
    if "Annual_Income" in centers.columns:
        ordered = centers["Annual_Income"].sort_values().index.tolist()
        labels = {
            ordered[0]: "Budget Customer",
            ordered[-1]: "Premium Customer",
        }
        remaining = [cluster for cluster in centers.index if cluster not in labels]
        for cluster in remaining:
            labels[cluster] = "Regular Customer" if len(remaining) == 1 else "Potential Customer"
        return labels
    return {int(cluster): f"Cluster {int(cluster) + 1}" for cluster in centers.index}


def build_segmentation_model(
    features: list[str] | None = None,
    cluster_count: int = 4,
    signature: str | None = None,
) -> dict[str, object]:
    df = load_dataset(signature or get_dataset_signature())
    if len(df) < 2:
        df = pd.concat([df, df], ignore_index=True)
    features = features or [feature for feature in SEGMENT_FEATURES if feature in df.columns]
    if len(features) < 2:
        features = get_available_segmentation_features()[:2]

    model_input = df[features].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_input)
    n_clusters = min(max(2, cluster_count), min(6, len(df)))

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = model.fit_predict(scaled)

    enriched = df.copy()
    enriched["Cluster"] = labels
    centers = enriched.groupby("Cluster")[features].mean()
    label_map = _derive_segment_labels(centers)
    enriched["Segment"] = enriched["Cluster"].map(label_map)

    return {
        "dataframe": enriched,
        "features": features,
        "cluster_count": n_clusters,
        "scaler": scaler,
        "model": model,
        "label_map": label_map,
    }


@lru_cache(maxsize=4)
def get_models(signature: str | None = None) -> dict[str, object]:
    actual_signature = signature or get_dataset_signature()
    df = load_dataset(actual_signature)
    segment_bundle = build_segmentation_model(signature=actual_signature)

    purchase_pipeline = _fit_classifier(
        df,
        [feature for feature in PURCHASE_FEATURES if feature in df.columns],
        "Next_Purchase_Category",
        RandomForestClassifier(n_estimators=400, random_state=42, min_samples_leaf=2),
    )
    churn_pipeline = _fit_classifier(
        df,
        [feature for feature in CHURN_FEATURES if feature in df.columns],
        "Churn_Status",
        RandomForestClassifier(n_estimators=300, random_state=42, min_samples_leaf=2),
    )

    class_df = segment_bundle["dataframe"].copy()
    customer_classifier = _fit_classifier(
        class_df,
        [feature for feature in CLASSIFICATION_FEATURES if feature in class_df.columns],
        "Segment",
        RandomForestClassifier(n_estimators=300, random_state=42, min_samples_leaf=2),
    )
    risk_classifier = _fit_classifier(
        class_df,
        [feature for feature in CLASSIFICATION_FEATURES if feature in class_df.columns],
        "Risk_Category",
        DecisionTreeClassifier(max_depth=5, random_state=42),
    )

    return {
        "segment_bundle": segment_bundle,
        "purchase_pipeline": purchase_pipeline,
        "churn_pipeline": churn_pipeline,
        "customer_classifier": customer_classifier,
        "risk_classifier": risk_classifier,
    }


def dataset_with_segments(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is not None:
        signature = get_dataset_signature()
        bundle = build_segmentation_model(signature=signature)
        joined = df.copy()
        if len(joined) == len(bundle["dataframe"]):
            return bundle["dataframe"]
    return get_models(get_dataset_signature())["segment_bundle"]["dataframe"].copy()


def _estimate_purchase_amount(profile: dict[str, object]) -> int:
    df = load_dataset(get_dataset_signature())
    narrowed = df[
        (df["Product_Category"] == profile["Product_Category"])
        & (df["Occupation"] == profile["Occupation"])
    ]
    if narrowed.empty:
        narrowed = df[df["Product_Category"] == profile["Product_Category"]]
    if narrowed.empty:
        narrowed = df
    return int(narrowed["Purchase_Amount"].median())


def find_similar_customers(profile: dict[str, object], limit: int = 10) -> pd.DataFrame:
    df = dataset_with_segments()
    age_window = max(3, int(profile["Age"] * 0.15))
    income_window = max(100000, int(profile["Annual_Income"] * 0.2))

    filtered = df[
        (df["Age"].between(profile["Age"] - age_window, profile["Age"] + age_window))
        & (
            df["Annual_Income"].between(
                profile["Annual_Income"] - income_window,
                profile["Annual_Income"] + income_window,
            )
        )
    ]

    for column in ["Gender", "City", "Occupation", "Product_Category"]:
        matched = filtered[filtered[column] == profile[column]]
        if not matched.empty:
            filtered = matched

    numeric_features = [
        "Age",
        "Annual_Income",
        "Monthly_Income",
        "EMI",
        "Purchase_Frequency",
        "Tenure_Months",
        "Product_Expiry_Days",
        "Customer_Satisfaction",
    ]
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df[numeric_features])
    scaled_profile = scaler.transform(
        pd.DataFrame([{feature: profile[feature] for feature in numeric_features}])
    )[0]

    distances = np.linalg.norm(scaled_df - scaled_profile, axis=1)
    ranked = df.copy()
    ranked["Similarity_Score"] = (1 / (1 + distances)) * 100

    if filtered.empty:
        filtered = ranked.nlargest(limit, "Similarity_Score")
    else:
        filtered = ranked.loc[filtered.index].sort_values("Similarity_Score", ascending=False).head(limit)
    return filtered.round({"Similarity_Score": 2})


def predict_segment(profile: dict[str, object]) -> dict[str, object]:
    bundle = get_models(get_dataset_signature())["segment_bundle"]
    row = pd.DataFrame([{feature: profile.get(feature, 0) for feature in bundle["features"]}])
    scaled = bundle["scaler"].transform(row)
    cluster = int(bundle["model"].predict(scaled)[0])
    distances = bundle["model"].transform(scaled)[0]
    total_distance = float(distances.sum()) or 1.0
    confidence = float((1 - (distances.min() / total_distance)) * 100)
    return {
        "cluster": cluster,
        "segment": bundle["label_map"][cluster],
        "confidence": round(confidence, 2),
    }


def predict_purchase(profile: dict[str, object]) -> dict[str, object]:
    pipeline = get_models(get_dataset_signature())["purchase_pipeline"]
    features = pipeline.feature_names_in_
    row = pd.DataFrame([{feature: profile.get(feature, 0) for feature in features}])
    probabilities = pipeline.predict_proba(row)[0]
    classes = pipeline.classes_
    ranked_indices = np.argsort(probabilities)[::-1][:3]
    ranked_products = [
        {
            "product": str(classes[index]),
            "probability": round(float(probabilities[index]) * 100, 2),
        }
        for index in ranked_indices
    ]
    return {
        "prediction": ranked_products[0]["product"],
        "probability": ranked_products[0]["probability"],
        "top_recommendations": ranked_products,
    }


def predict_churn(profile: dict[str, object]) -> dict[str, object]:
    pipeline = get_models(get_dataset_signature())["churn_pipeline"]
    features = pipeline.feature_names_in_
    row = pd.DataFrame([{feature: profile.get(feature, 0) for feature in features}])
    probabilities = pipeline.predict_proba(row)[0]
    classes = list(pipeline.classes_)
    if 1 in classes:
        probability = float(probabilities[classes.index(1)])
    else:
        probability = 0.0
    risk_level = classify_risk_level(probability)

    expiry_status = classify_expiry(profile["Product_Expiry_Days"])
    if risk_level == "High" and profile["Customer_Satisfaction"] <= 4:
        recommendation = "Assign immediate support outreach with a service recovery offer."
    elif expiry_status == "Critical":
        recommendation = "Trigger a renewal alert and limited-time retention incentive."
    elif profile["Last_Purchase_Days"] >= 90:
        recommendation = "Run a re-engagement campaign with category-specific reminders."
    else:
        recommendation = "Maintain engagement with loyalty rewards and proactive follow-up."

    return {
        "probability": round(probability * 100, 2),
        "risk_level": risk_level,
        "recommendation": recommendation,
    }


def classify_customer(profile: dict[str, object]) -> dict[str, object]:
    models = get_models(get_dataset_signature())
    customer_classifier = models["customer_classifier"]
    risk_classifier = models["risk_classifier"]

    customer_features = customer_classifier.feature_names_in_
    risk_features = risk_classifier.feature_names_in_
    customer_row = pd.DataFrame([{feature: profile.get(feature, 0) for feature in customer_features}])
    risk_row = pd.DataFrame([{feature: profile.get(feature, 0) for feature in risk_features}])

    customer_probabilities = customer_classifier.predict_proba(customer_row)[0]
    customer_classes = customer_classifier.classes_
    best_customer_index = int(np.argmax(customer_probabilities))

    risk_probabilities = risk_classifier.predict_proba(risk_row)[0]
    risk_classes = risk_classifier.classes_
    best_risk_index = int(np.argmax(risk_probabilities))

    return {
        "customer_type": str(customer_classes[best_customer_index]),
        "customer_probability": round(float(customer_probabilities[best_customer_index]) * 100, 2),
        "risk_category": str(risk_classes[best_risk_index]),
        "risk_probability": round(float(risk_probabilities[best_risk_index]) * 100, 2),
    }


def analyze_emi(profile: dict[str, object]) -> dict[str, object]:
    monthly_income = max(int(profile["Monthly_Income"]), 1)
    ratio = round((profile["EMI"] / monthly_income) * 100, 2)
    risk = classify_emi_ratio(ratio)
    stress_score = round(min(100, ratio * 1.4), 2)
    if risk == "High Risk":
        insight = "Debt servicing is consuming a large share of monthly income."
    elif risk == "Medium Risk":
        insight = "Cash flow is manageable but should be monitored for stress."
    else:
        insight = "Monthly obligations are well aligned with current income."
    return {"ratio": ratio, "risk": risk, "stress_score": stress_score, "insight": insight}


def analyze_expiry(profile: dict[str, object]) -> dict[str, object]:
    status = classify_expiry(profile["Product_Expiry_Days"])
    if status == "Critical":
        action = "Contact the customer now for renewal or replacement."
    elif status == "Warning":
        action = "Send a reminder campaign and targeted service offer."
    else:
        action = "Keep the customer in a routine lifecycle reminder journey."
    return {"status": status, "action": action}


def generate_recommendations(profile: dict[str, object], segment: str | None = None) -> list[str]:
    purchase = predict_purchase(profile)
    base_items = [item["product"] for item in purchase["top_recommendations"]]

    df = dataset_with_segments()
    filtered = df[df["Product_Category"] == profile["Product_Category"]]
    if segment:
        matching_segment = filtered[filtered["Segment"] == segment]
        if not matching_segment.empty:
            filtered = matching_segment

    popular_items = filtered["Next_Purchase_Category"].value_counts().index.tolist()
    recommendations = []
    for item in base_items + popular_items:
        if item not in recommendations:
            recommendations.append(item)
        if len(recommendations) == 5:
            break
    return recommendations


def describe_behavior(profile: dict[str, object], similar_customers: pd.DataFrame) -> str:
    if similar_customers.empty:
        return "Limited comparable history was found, so the profile is based on model expectations."

    avg_frequency = similar_customers["Purchase_Frequency"].mean()
    avg_satisfaction = similar_customers["Customer_Satisfaction"].mean()
    avg_spend = similar_customers["Purchase_Amount"].mean()

    if profile["Purchase_Frequency"] > avg_frequency and profile["Customer_Satisfaction"] >= avg_satisfaction:
        return "High-engagement profile with strong satisfaction and repeat buying intent."
    if profile["Purchase_Frequency"] <= avg_frequency and profile["Customer_Satisfaction"] < avg_satisfaction:
        return "Needs nurturing: lower engagement and lower satisfaction than similar customers."
    if _estimate_purchase_amount(profile) > avg_spend:
        return "High-value profile with cross-sell potential above the similar-customer baseline."
    return "Moderately engaged profile with room for targeted cross-sell and loyalty actions."


def build_customer_summary(profile: dict[str, object]) -> dict[str, object]:
    similar_customers = find_similar_customers(profile)
    segment_result = predict_segment(profile)
    purchase_result = predict_purchase(profile)
    churn_result = predict_churn(profile)
    emi_result = analyze_emi(profile)
    expiry_result = analyze_expiry(profile)
    classification = classify_customer(profile)
    recommendations = generate_recommendations(profile, segment_result["segment"])
    behavior = describe_behavior(profile, similar_customers)

    return {
        "profile": profile,
        "similar_customers": similar_customers.to_dict("records"),
        "similar_count": int(len(similar_customers)),
        "segment": segment_result,
        "purchase": purchase_result,
        "churn": churn_result,
        "emi": emi_result,
        "expiry": expiry_result,
        "classification": classification,
        "recommendations": recommendations,
        "behavior": behavior,
    }


def _empty_figure(title: str, x_label: str = "Category", y_label: str = "Count") -> str:
    if px is None:
        return _chart_placeholder(title, "Chart rendering is unavailable in this environment.")
    fig = px.bar(pd.DataFrame({x_label: ["No Data"], y_label: [0]}), x=x_label, y=y_label, title=title)
    return fig.to_html(full_html=False)


def generate_market_charts(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {
            "income_distribution": _empty_figure("Income Distribution", "Range", "Count"),
            "age_distribution": _empty_figure("Age Distribution", "Range", "Count"),
            "churn_distribution": _empty_figure("Churn Distribution", "Status", "Count"),
            "segment_distribution": _empty_figure("Segment Distribution", "Segment", "Count"),
            "city_analysis": _empty_figure("City Analysis", "City", "Customers"),
            "gender_analysis": _empty_figure("Gender Analysis", "Gender", "Customers"),
            "emi_risk_analysis": _empty_figure("EMI Risk Analysis", "Risk", "Count"),
            "purchase_frequency_analysis": _empty_figure("Purchase Frequency Analysis", "Frequency", "Count"),
            "product_popularity": _empty_figure("Product Popularity", "Product", "Count"),
            "revenue_analysis": _empty_figure("Revenue by Segment", "Segment", "Revenue"),
        }

    churn_counts = df["Churn_Label"].value_counts().reset_index()
    churn_counts.columns = ["Status", "Count"]
    segment_counts = df["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]
    city_counts = df["City"].value_counts().reset_index()
    city_counts.columns = ["City", "Customers"]
    gender_counts = df["Gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Customers"]
    emi_counts = df["EMI_Risk"].value_counts().reset_index()
    emi_counts.columns = ["Risk", "Count"]
    product_counts = df["Product_Category"].value_counts().head(8).reset_index()
    product_counts.columns = ["Product", "Count"]
    revenue_by_segment = df.groupby("Segment", as_index=False)["Purchase_Amount"].sum()

    if px is None:
        return {
            "income_distribution": _chart_placeholder("Income Distribution", "Plotly is not installed."),
            "age_distribution": _chart_placeholder("Age Distribution", "Plotly is not installed."),
            "churn_distribution": _chart_placeholder("Churn Distribution", "Plotly is not installed."),
            "segment_distribution": _chart_placeholder("Segment Distribution", "Plotly is not installed."),
            "city_analysis": _chart_placeholder("City Analysis", "Plotly is not installed."),
            "gender_analysis": _chart_placeholder("Gender Analysis", "Plotly is not installed."),
            "emi_risk_analysis": _chart_placeholder("EMI Risk Analysis", "Plotly is not installed."),
            "purchase_frequency_analysis": _chart_placeholder(
                "Purchase Frequency Analysis", "Plotly is not installed."
            ),
            "product_popularity": _chart_placeholder("Product Popularity", "Plotly is not installed."),
            "revenue_analysis": _chart_placeholder("Revenue Analysis", "Plotly is not installed."),
        }

    income_fig = px.histogram(df, x="Annual_Income", nbins=12, title="Income Distribution")
    age_fig = px.histogram(df, x="Age", nbins=12, title="Age Distribution")
    churn_fig = px.bar(churn_counts, x="Status", y="Count", color="Status", title="Churn Distribution")
    segment_fig = px.bar(segment_counts, x="Segment", y="Count", color="Segment", title="Segment Distribution")
    city_fig = px.bar(city_counts, x="City", y="Customers", color="City", title="City Analysis")
    gender_fig = px.pie(gender_counts, names="Gender", values="Customers", title="Gender Analysis")
    emi_fig = px.pie(emi_counts, names="Risk", values="Count", title="EMI Risk Analysis")
    frequency_fig = px.histogram(
        df,
        x="Purchase_Frequency",
        nbins=10,
        color="Segment",
        title="Purchase Frequency Analysis",
    )
    product_fig = px.bar(product_counts, x="Product", y="Count", color="Product", title="Product Popularity")
    revenue_fig = px.bar(
        revenue_by_segment,
        x="Segment",
        y="Purchase_Amount",
        color="Segment",
        title="Revenue Analysis",
    )

    return {
        "income_distribution": income_fig.to_html(full_html=False),
        "age_distribution": age_fig.to_html(full_html=False),
        "churn_distribution": churn_fig.to_html(full_html=False),
        "segment_distribution": segment_fig.to_html(full_html=False),
        "city_analysis": city_fig.to_html(full_html=False),
        "gender_analysis": gender_fig.to_html(full_html=False),
        "emi_risk_analysis": emi_fig.to_html(full_html=False),
        "purchase_frequency_analysis": frequency_fig.to_html(full_html=False),
        "product_popularity": product_fig.to_html(full_html=False),
        "revenue_analysis": revenue_fig.to_html(full_html=False),
    }


def generate_behavior_charts(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {
            "age_spend": _empty_figure("Age vs Spend", "Age", "Purchase"),
            "income_spend": _empty_figure("Income vs Spend", "Income", "Purchase"),
            "frequency_distribution": _empty_figure("Purchase Frequency Distribution", "Frequency", "Count"),
        }

    if px is None:
        return {
            "age_spend": _chart_placeholder("Age vs Spend", "Plotly is not installed."),
            "income_spend": _chart_placeholder("Income vs Spend", "Plotly is not installed."),
            "frequency_distribution": _chart_placeholder(
                "Purchase Frequency Distribution", "Plotly is not installed."
            ),
        }

    age_spend = px.scatter(
        df,
        x="Age",
        y="Purchase_Amount",
        color="Gender",
        title="Age vs Spend",
        hover_data=["Name", "City", "Segment"],
    )
    income_spend = px.scatter(
        df,
        x="Annual_Income",
        y="Purchase_Amount",
        color="Segment",
        title="Income vs Spend",
        hover_data=["Name", "Product_Category"],
    )
    frequency_distribution = px.histogram(
        df,
        x="Purchase_Frequency",
        nbins=10,
        color="Segment",
        title="Purchase Frequency Distribution",
    )
    return {
        "age_spend": age_spend.to_html(full_html=False),
        "income_spend": income_spend.to_html(full_html=False),
        "frequency_distribution": frequency_distribution.to_html(full_html=False),
    }


def generate_segmentation_chart(profile: dict[str, object], features: list[str] | None = None, cluster_count: int = 4) -> str:
    bundle = build_segmentation_model(features=features, cluster_count=cluster_count, signature=get_dataset_signature())
    df = bundle["dataframe"]
    chart_features = bundle["features"]
    segment_info = predict_segment(profile) if chart_features == get_models(get_dataset_signature())["segment_bundle"]["features"] else predict_custom_segment(profile, chart_features, cluster_count)

    x_feature = chart_features[0]
    y_feature = chart_features[1] if len(chart_features) > 1 else "Purchase_Amount"

    chart_base = df[[x_feature, y_feature, "Segment", "Name"]].copy()
    chart_base["Source"] = "Historical Customers"
    customer_marker = pd.DataFrame(
        [
            {
                x_feature: profile.get(x_feature, 0),
                y_feature: profile.get(y_feature, _estimate_purchase_amount(profile)),
                "Segment": segment_info["segment"],
                "Name": "Current Input",
                "Source": "Current Input",
            }
        ]
    )

    chart_data = pd.concat([chart_base, customer_marker], ignore_index=True)
    if px is None:
        return _chart_placeholder("Dynamic Customer Segmentation", "Plotly is not installed.")

    fig = px.scatter(
        chart_data,
        x=x_feature,
        y=y_feature,
        color="Segment",
        symbol="Source",
        title="Dynamic Customer Segmentation",
        hover_data=["Name"],
    )
    return fig.to_html(full_html=False)


def predict_custom_segment(profile: dict[str, object], features: list[str], cluster_count: int) -> dict[str, object]:
    bundle = build_segmentation_model(features=features, cluster_count=cluster_count, signature=get_dataset_signature())
    row = pd.DataFrame([{feature: profile.get(feature, 0) for feature in bundle["features"]}])
    scaled = bundle["scaler"].transform(row)
    cluster = int(bundle["model"].predict(scaled)[0])
    distances = bundle["model"].transform(scaled)[0]
    total_distance = float(distances.sum()) or 1.0
    confidence = float((1 - (distances.min() / total_distance)) * 100)
    return {
        "cluster": cluster,
        "segment": bundle["label_map"][cluster],
        "confidence": round(confidence, 2),
    }


def generate_segmentation_statistics(features: list[str], cluster_count: int) -> list[dict[str, object]]:
    bundle = build_segmentation_model(features=features, cluster_count=cluster_count, signature=get_dataset_signature())
    df = bundle["dataframe"]
    stats = []
    for cluster, group in df.groupby("Cluster"):
        stats.append(
            {
                "cluster_name": bundle["label_map"][int(cluster)],
                "size": int(len(group)),
                "avg_income": round(float(group["Annual_Income"].mean()), 2),
                "avg_spending": round(float(group["Purchase_Amount"].mean()), 2),
                "churn_risk": round(float(group["Churn_Status"].mean() * 100), 2),
                "preferred_product": group["Product_Category"].mode().iloc[0],
            }
        )
    return stats


def generate_cluster_distribution_chart(features: list[str], cluster_count: int) -> str:
    bundle = build_segmentation_model(features=features, cluster_count=cluster_count, signature=get_dataset_signature())
    cluster_counts = bundle["dataframe"]["Segment"].value_counts().reset_index()
    cluster_counts.columns = ["Segment", "Count"]
    if px is None:
        return _chart_placeholder("Cluster Size Chart", "Plotly is not installed.")
    fig = px.bar(cluster_counts, x="Count", y="Segment", orientation="h", color="Segment", title="Cluster Size Chart")
    return fig.to_html(full_html=False)


def generate_tenure_charts(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {
            "tenure_distribution": _empty_figure("Tenure Distribution", "Tenure", "Count"),
            "tenure_vs_spending": _empty_figure("Tenure vs Spending", "Tenure", "Spend"),
        }
    if px is None:
        return {
            "tenure_distribution": _chart_placeholder("Tenure Distribution", "Plotly is not installed."),
            "tenure_vs_spending": _chart_placeholder("Tenure vs Spending", "Plotly is not installed."),
        }
    distribution = px.histogram(df, x="Tenure_Months", nbins=12, title="Tenure Distribution")
    spend = px.scatter(
        df,
        x="Tenure_Months",
        y="Purchase_Amount",
        color="Segment",
        title="Tenure vs Spending",
        hover_data=["Name", "City"],
    )
    return {
        "tenure_distribution": distribution.to_html(full_html=False),
        "tenure_vs_spending": spend.to_html(full_html=False),
    }


def build_tenure_summary(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "avg_tenure": 0,
            "loyal_customers": 0,
            "new_customers": 0,
            "loyalty_insight": "No customers match the current filters.",
        }
    avg_tenure = round(float(df["Tenure_Months"].mean()), 2)
    loyal_customers = int((df["Tenure_Months"] >= 24).sum())
    new_customers = int((df["Tenure_Months"] <= 6).sum())
    loyalty_insight = (
        "Long-term customers are driving repeat value."
        if loyal_customers >= new_customers
        else "The portfolio has a large share of newly acquired customers."
    )
    return {
        "avg_tenure": avg_tenure,
        "loyal_customers": loyal_customers,
        "new_customers": new_customers,
        "loyalty_insight": loyalty_insight,
    }


def build_market_snapshot(filters: dict[str, object]) -> dict[str, object]:
    df = dataset_with_segments()
    filtered = filter_customers(df, filters)
    revenue_group = filtered.groupby("Income_Group", as_index=False)["Purchase_Amount"].sum() if not filtered.empty else pd.DataFrame(columns=["Income_Group", "Purchase_Amount"])
    highest_revenue_group = (
        revenue_group.sort_values("Purchase_Amount", ascending=False).iloc[0]["Income_Group"]
        if not revenue_group.empty
        else "No Data"
    )
    active_segment = filtered["Segment"].mode().iloc[0] if not filtered.empty else "No Data"
    fastest_growing_segment = (
        filtered.groupby("Segment")["Tenure_Months"].mean().sort_values().index[0]
        if not filtered.empty
        else "No Data"
    )
    return {
        "filtered_df": filtered,
        "customer_count": int(len(filtered)),
        "avg_income": int(filtered["Annual_Income"].mean()) if not filtered.empty else 0,
        "avg_purchase": int(filtered["Purchase_Amount"].mean()) if not filtered.empty else 0,
        "avg_emi": int(filtered["EMI"].mean()) if not filtered.empty else 0,
        "high_risk_count": int((filtered["Churn_Status"] == 1).sum()) if not filtered.empty else 0,
        "highest_revenue_group": highest_revenue_group,
        "most_active_segment": active_segment,
        "fastest_growing_segment": fastest_growing_segment,
        "charts": generate_market_charts(filtered),
        "behavior_charts": generate_behavior_charts(filtered),
        "tenure_charts": generate_tenure_charts(filtered),
        "tenure_summary": build_tenure_summary(filtered),
    }
