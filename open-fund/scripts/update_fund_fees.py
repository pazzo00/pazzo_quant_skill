"""Fetch public fund subscription/redemption fees and fill missing rates by sample means."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


SOURCE_TEMPLATE = "https://fundf10.eastmoney.com/jjfl_{code}.html"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; factor-premium-portfolio/1.0)",
    "Accept-Encoding": "identity",
}


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def normalize_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(character for character in text if character.isdigit())
    return digits.zfill(6)[-6:]


def percent_values(text: str) -> list[float]:
    return [float(value) / 100.0 for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]


def duration_to_days(value: str, unit: str) -> float:
    amount = float(value)
    if unit == "年":
        return amount * 365.0
    if unit in {"月", "个月"}:
        return amount * 30.0
    return amount


def duration_matches(label: str, holding_days: int) -> bool:
    text = re.sub(r"[\s（）()持有期限N]", "", label)
    recognized = False
    lower, lower_inclusive = -np.inf, True
    upper, upper_inclusive = np.inf, True
    patterns = [
        (r"(?:大于等于|不少于|≥|>=)(\d+(?:\.\d+)?)(年|个月|月|天|日)", "lower", True),
        (r"(?:大于|超过|>)(\d+(?:\.\d+)?)(年|个月|月|天|日)", "lower", False),
        (r"(\d+(?:\.\d+)?)(年|个月|月|天|日)(?:以上|及以上)", "lower", True),
        (r"(?:小于等于|不超过|至多|≤|<=)(\d+(?:\.\d+)?)(年|个月|月|天|日)", "upper", True),
        (r"(?:小于|不满|<)(\d+(?:\.\d+)?)(年|个月|月|天|日)", "upper", False),
        (r"(\d+(?:\.\d+)?)(年|个月|月|天|日)(?:以内|及以内)", "upper", True),
    ]
    for pattern, boundary, inclusive in patterns:
        for match in re.finditer(pattern, text):
            recognized = True
            days = duration_to_days(match.group(1), match.group(2))
            if boundary == "lower" and days >= lower:
                lower, lower_inclusive = days, inclusive
            if boundary == "upper" and days <= upper:
                upper, upper_inclusive = days, inclusive
    if not recognized:
        return False
    lower_ok = holding_days >= lower if lower_inclusive else holding_days > lower
    upper_ok = holding_days <= upper if upper_inclusive else holding_days < upper
    return lower_ok and upper_ok


def find_fee_table(soup: BeautifulSoup, heading: str):
    for header in soup.find_all(["h3", "h4"]):
        if heading in header.get_text(" ", strip=True):
            return header.find_next("table")
    return None


def parse_fee_page(code: str, html: str, holding_days: int) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    fund_type_match = re.search(r"类型：\s*([^\s]+)", page_text)
    row = {
        "code": code,
        "source_url": SOURCE_TEMPLATE.format(code=code),
        "fund_type": fund_type_match.group(1) if fund_type_match else None,
        "purchase_rate": np.nan,
        "purchase_original_rate": np.nan,
        "redemption_rate": np.nan,
        "redemption_holding_days": holding_days,
        "redemption_schedule_json": None,
        "status": "no_fee_data",
        "error": None,
    }

    purchase_table = find_fee_table(soup, "申购费率")
    if purchase_table is not None:
        for table_row in purchase_table.select("tbody tr"):
            cells = table_row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            values = percent_values(cells[-1].get_text(" ", strip=True))
            if values:
                row["purchase_original_rate"] = values[0]
                row["purchase_rate"] = values[-1]
                break

    redemption_table = find_fee_table(soup, "赎回费率")
    schedule = []
    if redemption_table is not None:
        for table_row in redemption_table.select("tbody tr"):
            cells = table_row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            duration = cells[0].get_text(" ", strip=True)
            values = percent_values(cells[-1].get_text(" ", strip=True))
            if not values:
                continue
            schedule.append({"duration": duration, "rate": values[-1]})
            if duration_matches(duration, holding_days):
                row["redemption_rate"] = values[-1]
        if schedule and not np.isfinite(row["redemption_rate"]):
            row["redemption_rate"] = schedule[-1]["rate"]
    row["redemption_schedule_json"] = json.dumps(schedule, ensure_ascii=False)
    if np.isfinite(row["purchase_rate"]) or np.isfinite(row["redemption_rate"]):
        row["status"] = "ok"
    return row


def fetch_one(code: str, holding_days: int, timeout: float) -> dict:
    url = SOURCE_TEMPLATE.format(code=code)
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        response.encoding = "utf-8"
        return parse_fee_page(code, response.text, holding_days)
    except Exception as error:  # individual failures are retained and imputed later
        return {
            "code": code,
            "source_url": url,
            "fund_type": None,
            "purchase_rate": np.nan,
            "purchase_original_rate": np.nan,
            "redemption_rate": np.nan,
            "redemption_holding_days": holding_days,
            "redemption_schedule_json": None,
            "status": "fetch_failed",
            "error": str(error),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--code-column", default="code")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holding-days", type=int, default=365)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    source = read_table(args.input)
    if args.code_column not in source.columns:
        raise ValueError(f"Missing code column: {args.code_column}")
    codes = sorted({normalize_code(value) for value in source[args.code_column].dropna()})
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(fetch_one, code, args.holding_days, args.timeout): code for code in codes}
        for job in as_completed(jobs):
            rows.append(job.result())
    fees = pd.DataFrame(rows).sort_values("code").reset_index(drop=True)

    purchase_mean = float(fees["purchase_rate"].mean(skipna=True))
    redemption_mean = float(fees["redemption_rate"].mean(skipna=True))
    if not np.isfinite(purchase_mean) or not np.isfinite(redemption_mean):
        raise RuntimeError("No usable fee observations were found; cannot apply mean-rate fallback")
    fees["purchase_rate_imputed"] = fees["purchase_rate"].isna()
    fees["redemption_rate_imputed"] = fees["redemption_rate"].isna()
    fees["purchase_rate"] = fees["purchase_rate"].fillna(purchase_mean)
    fees["redemption_rate"] = fees["redemption_rate"].fillna(redemption_mean)
    fees["fee_as_of"] = pd.Timestamp.now().normalize()
    fees["source"] = "Eastmoney fund fee page"
    fees["fallback_purchase_mean"] = purchase_mean
    fees["fallback_redemption_mean"] = redemption_mean

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fees.to_parquet(args.output, index=False)
    summary = {
        "status": "ok",
        "funds": len(fees),
        "purchase_observed": int((~fees["purchase_rate_imputed"]).sum()),
        "purchase_imputed": int(fees["purchase_rate_imputed"].sum()),
        "redemption_observed": int((~fees["redemption_rate_imputed"]).sum()),
        "redemption_imputed": int(fees["redemption_rate_imputed"].sum()),
        "purchase_mean": purchase_mean,
        "redemption_mean": redemption_mean,
        "holding_days": args.holding_days,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
