"""
Exploratory extraction for the openFDA Drug Event API.

This script consumes nested adverse event safety reports from:
https://api.fda.gov/drug/event.json

It converts the raw JSON into separate pandas DataFrames with clear units of
observation: one row per report, one row per patient, one row per drug, and one
row per reaction. The resulting tables can be joined with safetyreportid.

Warning:
These are adverse event reports, not confirmed causal events. A report may
contain multiple drugs and multiple reactions. Do not interpret drug-reaction
pairs as causal relationships, incidence rates, or proof that a drug caused an
event.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


BASE_URL = "https://api.fda.gov/drug/event.json"
DEFAULT_REACTION_TERM = "headache"
DEFAULT_MAX_RECORDS = 10000
DEFAULT_OUTPUT_DIR = Path(".")


REPORT_FIELDS = [
    "safetyreportid",
    "safetyreportversion",
    "receivedate",
    "receiptdate",
    "transmissiondate",
    "serious",
    "seriousnessdeath",
    "seriousnesshospitalization",
    "seriousnesslifethreatening",
    "seriousnessdisabling",
    "seriousnessother",
    "primarysourcecountry",
    "occurcountry",
    "reporttype",
    "duplicate",
    "companynumb",
]

PATIENT_FIELDS = [
    "safetyreportid",
    "patientonsetage",
    "patientonsetageunit",
    "patientsex",
    "patientweight",
]

DRUG_FIELDS = [
    "safetyreportid",
    "drug_sequence",
    "medicinalproduct",
    "drugcharacterization",
    "drugadministrationroute",
    "drugindication",
    "drugdosagetext",
    "drugstartdate",
    "drugauthorizationnumb",
    "drugbatchnumb",
]

REACTION_FIELDS = [
    "safetyreportid",
    "reaction_sequence",
    "reactionmeddrapt",
    "reactionmeddraversionpt",
]

SOURCE_FIELDS = [
    "safetyreportid",
    "primarysource_qualification",
    "primarysource_reportercountry",
    "primarysourcecountry",
    "sender_sendertype",
    "sender_senderorganization",
    "receiver_receivertype",
    "receiver_receiverorganization",
]

DUPLICATE_FIELDS = [
    "safetyreportid",
    "duplicate",
    "reportduplicate_duplicatenumb",
    "reportduplicate_duplicatesource",
]


def load_env_file(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from .env without requiring python-dotenv."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def call_openfda(
    search: str | None = None,
    limit: int = 100,
    skip: int = 0,
) -> dict[str, Any]:
    """Call the openFDA Drug Event API and return the decoded JSON response."""
    params: dict[str, Any] = {
        "limit": limit,
        "skip": skip,
    }

    api_key = os.getenv("OPENFDA_API_KEY")
    if api_key:
        params["api_key"] = api_key

    if search:
        params["search"] = search

    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "openfda-eda-python/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "API request failed.\n"
            f"Status code: {error.code}\n"
            f"URL: {url}\n"
            f"Response: {body[:500]}"
        ) from error


def fetch_reports(
    reaction_term: str = DEFAULT_REACTION_TERM,
    max_records: int = 100,
    page_size: int = 100,
    pause_seconds: float = 0.25,
) -> list[dict[str, Any]]:
    """Fetch raw safety reports from openFDA using simple skip pagination."""
    search_query = f'patient.reaction.reactionmeddrapt:"{reaction_term}"'
    page_size = min(page_size, 1000)
    reports: list[dict[str, Any]] = []

    for skip in range(0, max_records, page_size):
        limit = min(page_size, max_records - len(reports))
        if limit <= 0:
            break

        data = call_openfda(search=search_query, limit=limit, skip=skip)
        page_results = data.get("results", [])
        reports.extend(page_results)

        if len(page_results) < limit or len(reports) >= max_records:
            break

        time.sleep(pause_seconds)

    return reports


def value_from(source: dict[str, Any], key: str) -> Any:
    """Return a top-level value, preserving missing fields as None."""
    return source.get(key)


def first_dict(value: Any) -> dict[str, Any]:
    """Return a dictionary from values that may arrive as dict, list, or None."""
    if isinstance(value, dict):
        return value

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item

    return {}


def joined_field(value: Any, key: str) -> Any:
    """Extract a field from a dict or join the field values from a list of dicts."""
    if isinstance(value, dict):
        return value.get(key)

    if isinstance(value, list):
        values = [
            str(item.get(key))
            for item in value
            if isinstance(item, dict) and item.get(key) is not None
        ]
        return " | ".join(values) if values else None

    return None


def create_reports_df(raw_reports: list[dict[str, Any]]) -> pd.DataFrame:
    # Unit of observation: one safety report.
    rows = [
        {field: value_from(report, field) for field in REPORT_FIELDS}
        for report in raw_reports
    ]
    return pd.DataFrame(rows, columns=REPORT_FIELDS)


def create_patients_df(raw_reports: list[dict[str, Any]]) -> pd.DataFrame:
    # Unit of observation: one patient per safety report.
    rows = []
    for report in raw_reports:
        patient = report.get("patient", {})
        rows.append(
            {
                "safetyreportid": report.get("safetyreportid"),
                "patientonsetage": patient.get("patientonsetage"),
                "patientonsetageunit": patient.get("patientonsetageunit"),
                "patientsex": patient.get("patientsex"),
                "patientweight": patient.get("patientweight"),
            }
        )

    return pd.DataFrame(rows, columns=PATIENT_FIELDS)


def create_drugs_df(raw_reports: list[dict[str, Any]]) -> pd.DataFrame:
    # Unit of observation: one drug inside one safety report.
    rows = []
    for report in raw_reports:
        safetyreportid = report.get("safetyreportid")
        patient = report.get("patient", {})
        drugs = patient.get("drug", [])

        for sequence, drug in enumerate(drugs, start=1):
            rows.append(
                {
                    "safetyreportid": safetyreportid,
                    "drug_sequence": sequence,
                    "medicinalproduct": drug.get("medicinalproduct"),
                    "drugcharacterization": drug.get("drugcharacterization"),
                    "drugadministrationroute": drug.get("drugadministrationroute"),
                    "drugindication": drug.get("drugindication"),
                    "drugdosagetext": drug.get("drugdosagetext"),
                    "drugstartdate": drug.get("drugstartdate"),
                    "drugauthorizationnumb": drug.get("drugauthorizationnumb"),
                    "drugbatchnumb": drug.get("drugbatchnumb"),
                }
            )

    return pd.DataFrame(rows, columns=DRUG_FIELDS)


def create_reactions_df(raw_reports: list[dict[str, Any]]) -> pd.DataFrame:
    # Unit of observation: one reaction inside one safety report.
    rows = []
    for report in raw_reports:
        safetyreportid = report.get("safetyreportid")
        patient = report.get("patient", {})
        reactions = patient.get("reaction", [])

        for sequence, reaction in enumerate(reactions, start=1):
            rows.append(
                {
                    "safetyreportid": safetyreportid,
                    "reaction_sequence": sequence,
                    "reactionmeddrapt": reaction.get("reactionmeddrapt"),
                    "reactionmeddraversionpt": reaction.get(
                        "reactionmeddraversionpt"
                    ),
                }
            )

    return pd.DataFrame(rows, columns=REACTION_FIELDS)


def create_sources_df(raw_reports: list[dict[str, Any]]) -> pd.DataFrame:
    # Unit of observation: source metadata for one safety report.
    rows = []
    for report in raw_reports:
        primarysource = first_dict(report.get("primarysource"))
        sender = first_dict(report.get("sender"))
        receiver = first_dict(report.get("receiver"))

        rows.append(
            {
                "safetyreportid": report.get("safetyreportid"),
                "primarysource_qualification": primarysource.get("qualification"),
                "primarysource_reportercountry": primarysource.get(
                    "reportercountry"
                ),
                "primarysourcecountry": report.get("primarysourcecountry"),
                "sender_sendertype": sender.get("sendertype"),
                "sender_senderorganization": sender.get("senderorganization"),
                "receiver_receivertype": receiver.get("receivertype"),
                "receiver_receiverorganization": receiver.get(
                    "receiverorganization"
                ),
            }
        )

    return pd.DataFrame(rows, columns=SOURCE_FIELDS)


def create_duplicates_df(raw_reports: list[dict[str, Any]]) -> pd.DataFrame:
    # Unit of observation: duplicate metadata for one safety report.
    rows = []
    for report in raw_reports:
        reportduplicate = report.get("reportduplicate")

        rows.append(
            {
                "safetyreportid": report.get("safetyreportid"),
                "duplicate": report.get("duplicate"),
                "reportduplicate_duplicatenumb": joined_field(
                    reportduplicate,
                    "duplicatenumb"
                ),
                "reportduplicate_duplicatesource": joined_field(
                    reportduplicate,
                    "duplicatesource"
                ),
            }
        )

    return pd.DataFrame(rows, columns=DUPLICATE_FIELDS)


def create_dataframes(raw_reports: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Create all EDA DataFrames from raw openFDA reports."""
    return {
        "reports_df": create_reports_df(raw_reports),
        "patients_df": create_patients_df(raw_reports),
        "drugs_df": create_drugs_df(raw_reports),
        "reactions_df": create_reactions_df(raw_reports),
        "sources_df": create_sources_df(raw_reports),
        "duplicates_df": create_duplicates_df(raw_reports),
    }


def add_report_year(reports_df: pd.DataFrame) -> pd.DataFrame:
    """Add a numeric year from receivedate formatted as YYYYMMDD."""
    reports_df = reports_df.copy()
    reports_df["year"] = pd.to_datetime(
        reports_df["receivedate"],
        format="%Y%m%d",
        errors="coerce",
    ).dt.year
    return reports_df


def print_missing_summary(name: str, df: pd.DataFrame) -> None:
    """Print count and percentage of missing values for each field."""
    print(f"\nMissing values: {name}")
    if df.empty:
        print("DataFrame is empty.")
        return

    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
        }
    )
    print(missing)


def export_dataframes(
    dataframes: dict[str, pd.DataFrame],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Export every DataFrame to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in dataframes.items():
        output_path = output_dir / f"{name}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Exported: {output_path}")


def print_shapes(dataframes: dict[str, pd.DataFrame]) -> None:
    """Print the shape of each DataFrame."""
    print("\nDataFrame shapes")
    for name, df in dataframes.items():
        print(f"- {name}: {df.shape}")


def print_exploratory_summaries(dataframes: dict[str, pd.DataFrame]) -> None:
    """Generate basic exploratory summaries from the normalized tables."""
    reports_df = add_report_year(dataframes["reports_df"])
    patients_df = dataframes["patients_df"]
    drugs_df = dataframes["drugs_df"]
    reactions_df = dataframes["reactions_df"]

    print("\nExploratory summaries")
    print(f"Number of safety reports: {len(reports_df):,}")
    print(f"Number of drugs: {len(drugs_df):,}")
    print(f"Number of reactions: {len(reactions_df):,}")

    print("\nTop 20 medicinal products")
    print(
        drugs_df["medicinalproduct"]
        .value_counts(dropna=True)
        .head(20)
        .rename_axis("medicinalproduct")
        .reset_index(name="count")
    )

    print("\nTop 20 reaction terms")
    print(
        reactions_df["reactionmeddrapt"]
        .value_counts(dropna=True)
        .head(20)
        .rename_axis("reactionmeddrapt")
        .reset_index(name="count")
    )

    print("\nReports by year")
    print(
        reports_df["year"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("year")
        .reset_index(name="report_count")
    )

    print("\nSerious vs non-serious reports")
    print(
        reports_df["serious"]
        .value_counts(dropna=False)
        .rename_axis("serious")
        .reset_index(name="report_count")
    )

    print("\nMissing values in patient age and sex")
    print(patients_df[["patientonsetage", "patientsex"]].isna().sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create exploratory pandas DataFrames from openFDA Drug Event reports."
    )
    parser.add_argument(
        "--reaction-term",
        default=DEFAULT_REACTION_TERM,
        help="Reaction term used in the openFDA search query.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=DEFAULT_MAX_RECORDS,
        help="Maximum number of safety reports to fetch.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of safety reports per API page. openFDA limit is capped at 1000.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV files will be exported.",
    )
    return parser.parse_args()


def main() -> dict[str, pd.DataFrame]:
    args = parse_args()
    load_env_file()

    print(
        "WARNING: These are adverse event reports, not confirmed causal events. "
        "Do not treat drug-reaction pairs as causal relationships."
    )
    print(f"\nFetching openFDA reports for reaction term: {args.reaction_term}")

    raw_reports = fetch_reports(
        reaction_term=args.reaction_term,
        max_records=args.max_records,
        page_size=args.page_size,
    )

    dataframes = create_dataframes(raw_reports)

    print_shapes(dataframes)

    for name, df in dataframes.items():
        print_missing_summary(name, df)

    export_dataframes(dataframes, args.output_dir)
    print_exploratory_summaries(dataframes)

    return dataframes


if __name__ == "__main__":
    main()
