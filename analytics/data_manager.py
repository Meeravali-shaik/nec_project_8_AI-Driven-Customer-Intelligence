from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = BASE_DIR / "customer_dataset_100.csv"
UPLOAD_DIR = BASE_DIR / "uploads"
STATE_PATH = BASE_DIR / "active_dataset.json"

UPLOAD_DIR.mkdir(exist_ok=True)

CANONICAL_ALIASES = {
    "Customer_ID": ["customerid", "customer_id", "custid", "clientid", "id"],
    "Name": ["name", "customername", "clientname", "full_name"],
    "Age": ["age", "customerage"],
    "Gender": ["gender", "sex"],
    "City": ["city", "location", "town"],
    "Occupation": ["occupation", "job", "profession"],
    "Annual_Income": ["annualincome", "income", "yearlyincome", "salary", "annual_salary"],
    "Monthly_Income": ["monthlyincome", "monthly_salary", "monthlysalary"],
    "EMI": ["emi", "monthlyemi", "loanemi"],
    "Dependents": ["dependents", "familymembers"],
    "Tenure_Months": ["tenuremonths", "tenure", "loyaltymonths", "monthsactive"],
    "Product_Category": ["productcategory", "category", "product_type"],
    "Last_Product": ["lastproduct", "previousproduct", "latestproduct"],
    "Purchase_Amount": ["purchaseamount", "amountspent", "spend", "spending", "revenue"],
    "Purchase_Frequency": ["purchasefrequency", "frequency", "ordersfrequency"],
    "Total_Purchases": ["totalpurchases", "totalorders", "orderscount"],
    "Avg_Order_Value": ["avgordervalue", "averageordervalue", "basketvalue"],
    "Last_Purchase_Days": ["lastpurchasedays", "dayssincelastpurchase", "recency"],
    "Product_Expiry_Days": ["productexpirydays", "expirydays", "days_to_expiry"],
    "Customer_Satisfaction": ["customersatisfaction", "satisfaction", "satisfactionscore"],
    "Website_Visits": ["websitevisits", "visits", "sitevisits"],
    "Email_Clicks": ["emailclicks", "campaignclicks"],
    "Discount_Usage": ["discountusage", "couponusage"],
    "Churn_Status": ["churnstatus", "churn", "is_churned"],
    "Next_Purchase_Category": ["nextpurchasecategory", "nextproduct", "next_purchase"],
}


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def infer_field_mapping(columns: list[str]) -> dict[str, str]:
    normalized = {normalize_column_name(column): column for column in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        if canonical in columns:
            mapping[canonical] = canonical
            continue
        for alias in aliases:
            matched = normalized.get(alias)
            if matched:
                mapping[canonical] = matched
                break
    return mapping


def _read_state() -> dict[str, str]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_state(payload: dict[str, str]) -> None:
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_active_dataset_path() -> Path:
    state = _read_state()
    dataset_path = Path(state.get("path", DEFAULT_DATASET_PATH))
    if not dataset_path.exists():
        return DEFAULT_DATASET_PATH
    return dataset_path


def get_active_dataset_label() -> str:
    path = get_active_dataset_path()
    if path == DEFAULT_DATASET_PATH:
        return "Default Sample Dataset"
    return path.name


def get_dataset_signature() -> str:
    path = get_active_dataset_path()
    stat = path.stat()
    return f"{path.resolve()}::{int(stat.st_mtime_ns)}::{stat.st_size}"


def load_active_dataset_raw() -> pd.DataFrame:
    path = get_active_dataset_path()
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


def save_uploaded_dataset(upload) -> Path:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in {".csv", ".xlsx"}:
        raise ValueError("Please upload a CSV or XLSX file.")

    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(upload.filename).stem).strip("_")
    if not safe_stem:
        safe_stem = "customer_dataset"

    target = UPLOAD_DIR / f"{safe_stem}_{uuid4().hex[:8]}{extension}"
    upload.save(target)
    _write_state({"path": str(target)})
    return target


def reset_to_default_dataset() -> None:
    _write_state({"path": str(DEFAULT_DATASET_PATH)})


def build_dataset_profile(df: pd.DataFrame | None = None) -> dict[str, object]:
    if df is None:
        df = load_active_dataset_raw()

    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numerical_columns]
    missing = df.isna().sum()
    mapping = infer_field_mapping(df.columns.tolist())

    numeric_preview = {}
    for column in numerical_columns[:8]:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        numeric_preview[column] = {
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "mean": round(float(series.mean()), 2),
        }

    return {
        "dataset_name": get_active_dataset_label(),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "missing_columns": [
            {"name": name, "count": int(count)}
            for name, count in missing.sort_values(ascending=False).items()
            if int(count) > 0
        ],
        "field_mapping": mapping,
        "numeric_preview": numeric_preview,
    }

