import pandas as pd
import requests
import numpy as np
import math
import time
from tqdm import tqdm
import concurrent.futures as cf

# -------------------------
# Configurations
# -------------------------
GRAPH_URL = "https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2"
BATCH = 1000
HTTP_RETRIES = 3
HTTP_BACKOFF = 1.5

WEIGHTS = {
    "ltv": 0.35,
    "liquidation_rate": 0.30,
    "borrow_intensity": 0.15,
    "inactivity": 0.10,
    "tx_volatility": 0.10
}

SCORE_SCALE = 1000


# -------------------------
# Helper Functions
# -------------------------
def retry_post_json(url, json, retries=HTTP_RETRIES, backoff=HTTP_BACKOFF):
    for i in range(retries):
        try:
            r = requests.post(url, json=json, timeout=60)
            if r.status_code == 200:
                return r.json()
            else:
                time.sleep(backoff ** i)
        except Exception:
            time.sleep(backoff ** i)
    raise RuntimeError(f"POST {url} failed after {retries} retries")


def run_query(query: str, variables: dict):
    data = retry_post_json(GRAPH_URL, {"query": query, "variables": variables})
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]


def paginate_events(query_tmpl: str, root_field: str, variables: dict, limit=BATCH):
    out = []
    skip = 0
    while True:
        vars_p = dict(variables)
        vars_p["first"] = limit
        vars_p["skip"] = skip
        data = run_query(query_tmpl, vars_p)
        chunk = data[root_field]
        out.extend(chunk)
        if len(chunk) < limit:
            break
        skip += limit
    return out


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


# -------------------------
# GraphQL Queries
# -------------------------
BORROW_Q = """
query($account: String!, $first: Int!, $skip: Int!) {
  borrows(first:$first, skip:$skip, orderBy:blockTime, orderDirection:asc, where:{borrower:$account}) {
    id borrower amount amountUSD blockTime
  }
}
"""

REPAY_Q = """
query($account: String!, $first: Int!, $skip: Int!) {
  repayBorrows(first:$first, skip:$skip, orderBy:blockTime, orderDirection:asc, where:{borrower:$account}) {
    id borrower payer amount amountUSD blockTime
  }
}
"""

LIQ_Q = """
query($account: String!, $first: Int!, $skip: Int!) {
  liquidateBorrows(first:$first, skip:$skip, orderBy:blockTime, orderDirection:asc, where:{borrower:$account}) {
    id borrower liquidator amount amountUSD blockTime
  }
}
"""

MINT_Q = """
query($account: String!, $first: Int!, $skip: Int!) {
  mints(first:$first, skip:$skip, orderBy:blockTime, orderDirection:asc, where:{minter:$account}) {
    id minter amount amountUSD blockTime
  }
}
"""

REDEEM_Q = """
query($account: String!, $first: Int!, $skip: Int!) {
  redeems(first:$first, skip:$skip, orderBy:blockTime, orderDirection:asc, where:{redeemer:$account}) {
    id redeemer amount amountUSD blockTime
  }
}
"""


# -------------------------
# Data Processing
# -------------------------
def fetch_wallet_events(wallet: str):
    variables = {"account": wallet}
    borrows = paginate_events(BORROW_Q, "borrows", variables)
    repays = paginate_events(REPAY_Q, "repayBorrows", variables)
    liqs = paginate_events(LIQ_Q, "liquidateBorrows", variables)
    mints = paginate_events(MINT_Q, "mints", variables)
    redeems = paginate_events(REDEEM_Q, "redeems", variables)

    return {
        "wallet": wallet,
        "borrows": borrows,
        "repays": repays,
        "liquidations": liqs,
        "mints": mints,
        "redeems": redeems,
    }


def engineer_features(events: dict) -> dict:
    wallet = events["wallet"]

    n_borrows = len(events["borrows"])
    n_liqs = len(events["liquidations"])

    total_borrow_usd = sum(safe_float(e.get("amountUSD", 0)) for e in events["borrows"])
    total_supply_usd = sum(safe_float(e.get("amountUSD", 0)) for e in events["mints"])

    ltv = total_borrow_usd / (total_supply_usd + 1e-9)
    liquidation_rate = n_liqs / (n_borrows + 1e-9)

    # Basic proxy for borrow intensity
    borrow_intensity = total_borrow_usd / (n_borrows + 1)  

    # Placeholder for inactivity and volatility
    inactivity_days = 30  # Example fixed value
    tx_volatility = 0.5   # Example fixed value

    return dict(
        wallet_id=wallet,
        ltv=ltv,
        liquidation_rate=liquidation_rate,
        borrow_intensity=borrow_intensity,
        inactivity_days=inactivity_days,
        tx_volatility=tx_volatility,
    )


def min_max_normalize(series: pd.Series):
    mn, mx = series.min(), series.max()
    if math.isclose(mx, mn):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["ltv", "liquidation_rate", "borrow_intensity", "inactivity_days", "tx_volatility"]:
        df[f"norm_{col}"] = min_max_normalize(df[col])

    df["risk_score"] = SCORE_SCALE * (
        WEIGHTS["ltv"] * df["norm_ltv"] +
        WEIGHTS["liquidation_rate"] * df["norm_liquidation_rate"] +
        WEIGHTS["borrow_intensity"] * df["norm_borrow_intensity"] +
        WEIGHTS["inactivity"] * df["norm_inactivity_days"] +
        WEIGHTS["tx_volatility"] * df["norm_tx_volatility"]
    )

    df["risk_score"] = df["risk_score"].clip(0, SCORE_SCALE)
    return df


def main():
    wallets_df = pd.read_csv("/mnt/data/40cfdb94-02a7-432b-bc7a-4ef3764052d9.csv")
    wallets = wallets_df["wallet_id"].str.lower().tolist()

    results = []
    with cf.ThreadPoolExecutor(max_workers=5) as executor:
        for res in tqdm(executor.map(fetch_wallet_events, wallets), total=len(wallets), desc="Fetching Wallet Data"):
            feats = engineer_features(res)
            results.append(feats)

    df = pd.DataFrame(results)
    df = compute_scores(df)

    df[["wallet_id", "risk_score"]].rename(columns={"risk_score": "score"}).to_csv("wallet_scores.csv", index=False)
    print("Saved wallet_scores.csv")


if __name__ == "__main__":
    main()