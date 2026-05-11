import requests
import pandas as pd
import json
from pathlib import Path

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
})

def get_json(url, params=None):
    """
    Sending a GET request and returning the JSON response. If the 
    request fails, returns an HTTPError.

    arguments:
        url -- API endpoint URL
        params -- the query parameters for the get request
    """
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_json(x):
    """
    Converts JSON string into a Python object if possible, returning the
    object or the initial input.

    arguments:
        x -- JSON string 
    """
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("[") or x.startswith("{"):
            return json.loads(x)
    return x

def create_output_folder():
    """
    Creates the output folder for the data (if needed) and 
    returns the name of the output folder for future use.
    """
    output_dir = Path.cwd() / "polymarket_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir

# function to look up market via slug
# returns each's condition_id
def lookup_via_slug(slug):
    """
    Queries the Polymarket Gamma API for the market data via the slug,
    extracts the token ID which is used to query the CLOB API and 
    find the associated condition_id, which is returned by the function.

    arguments:
        slug -- unique identifier for individual markets and events that is found
                directly in the Polymarket frontend URL (after /event/) and used
                to fetch the associated data
    """

    market = get_json(f"{GAMMA_BASE}/markets/slug/{slug}")
    token_ids = parse_json(market.get("clobTokenIds"))

    if not isinstance(token_ids, list) or not token_ids:
        raise RuntimeError("Could not parse token IDs")
    
    token_for_lookup = token_ids[0]
    market_by_token = get_json(f"{CLOB_BASE}/markets-by-token/{token_for_lookup}")

    condition_id = market_by_token.get("condition_id") or market_by_token.get("conditionId")

    if not condition_id:
        raise RuntimeError("No condition_id found in markets-by-token response")
    
    return condition_id

def pull_trades(condition_id):
    """
    Retrieves and returns the trade data for the given market condition_id.

    arguments:
        condition_id -- condiion ID representing an individual Polymarke market
    """
    params = {
        "market": condition_id,
        "limit": 10000,
        "offset": 0,
    }

    trades = get_json(f"{DATA_BASE}/trades", params=params)
    return trades

def save_data_csv(trades, output_dir, slug):
    """
    Saves the raw and cleaned trade data to JSON and CSV files.

    arguments:
        trades -- trade information that was previously pulled
        output_dir -- name of output directory where data is being saved
        slug -- market slug that is also used to name/identify the output files
    """
    raw_json_path = output_dir / f"{slug}_trades_raw.json"

    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2)

    if isinstance(trades, list) and len(trades) > 0:
        df = pd.DataFrame(trades)

        if "timestamp" in df.columns:
            df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")

        raw_csv_path = output_dir / f"{slug}_trades_raw.csv"
        df.to_csv(raw_csv_path, index=False)

        keep_cols = [
            "timestamp",
            "datetime_utc",
            "price",
            "size",
            "side",
            "asset",
            "conditionId",
            "slug",
            "eventSlug",
            "outcome",
            "outcomeIndex",
            "transactionHash",
            "proxyWallet",
        ]
        existing_cols = [c for c in keep_cols if c in df.columns]

        clean_df = df[existing_cols].copy()
        clean_csv_path = output_dir / f"{slug}_trades_clean.csv"
        clean_df.to_csv(clean_csv_path, index=False)
    else:
        print("No trade rows available")
        return
    
if __name__ == "__main__":
    slugs = ["nfl-sea-ne-2026-02-08", "world-series-winner", "nba-finals-thunder-vs-pacers"]

    output_dir = create_output_folder()

    for slug in slugs:
        print(f"Extracting data for SLUG {slug}")
        condition_id = lookup_via_slug(slug)
        trades = pull_trades(condition_id)
        save_data_csv(trades, output_dir, slug)
        print(f"Data for SLUG {slug} saved")

        print("All polymarket data extracted and saved.")

        
