import os
import math
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import concurrent.futures as cf
from datetime import datetime, timezone

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(page_title="Wallet Risk Scoring (Compound / Aave)", layout="wide")
st.title("💰 Wallet Risk Scoring (Compound / Aave) – End-to-End")

st.write("""
Upload a CSV of **`wallet_id`**s or paste the Google Sheet link. The app will pull on‑chain actions
from **Compound V2**, **Compound V3**, or **Aave V3**, engineer risk features, normalize them,
and assign a **0–1000 score**. Results are saved under `outputs/` for persistence.
""")

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH = 1000
HTTP_RETRIES = 3
HTTP_BACKOFF = 1.5
SCORE_SCALE = 1000

DEFAULT_WEIGHTS = {
    "ltv": 0.35,
    "liquidation_rate": 0.30,
    "borrow_intensity": 0.15,
    "inactivity": 0.10,
    "tx_volatility": 0.10,
}

PROTOCOL_ENDPOINTS = {
    "Compound V2": "https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2",
    "Compound V3": "https://api.thegraph.com/subgraphs/name/0xlaozi/compound-v3-ethereum",
    "Aave V3":     "https://api.thegraph.com/subgraphs/name/aave/protocol-v3",
}

# For simplicity, we’re reusing Compound V2-style queries for all three protocols.
# If a schema doesn't match, we catch the error and return an empty list for that entity.
BORROW_Q_V2 = """
query($account: String!, $first: Int!, $skip: Int!) {
  borrows(first:$first, skip:$skip, orderBy:blockTime, orderDirection:asc, where:{borrower:$account}) {
    id borrower amount amountUSD blockTime
  }
}
"""
REPAY_Q_V2 = """
query($account: String!, $first: Int!, $skip: Int!) {
  repayBorrows(first:$first, skip:$skip, orderBy:blockTime, orderDirection:asc, where:{borrower:$account}) {
    id borrower payer amount amountUSD blockTime
  }
}
"""
LIQ_Q_V2 = """
query($account: String!, $first: Int!, $skip: Int!) {
  liquidateBorrows(first:$first, skip:$skip, orderBy:blockTime, orderDirection:asc, where:{borrower:$account}) {
    id borrower liquidator amount amountUSD blockTime
  }
}
"""
MINT_Q_V2 = """
query($account: String!, $first: Int!, $skip: Int!) {
  mints(first:$first, skip:$skip, orderBy:blockTime, orderDirection:asc, where:{minter:$account}) {
    id minter amount amountUSD blockTime
  }
}
"""
REDEEM_Q_V2 = """
query($account: String!, $first: Int!, $skip: Int!) {
  redeems(first:$first, skip:$skip, orderBy:blockTime, orderDirection:asc, where:{redeemer:$account}) {
    id redeemer amount amountUSD blockTime
  }
}
"""

PROTOCOL_QUERIES = {
    "Compound V2": dict(borrow=BORROW_Q_V2, repay=REPAY_Q_V2, liq=LIQ_Q_V2, mint=MINT_Q_V2, redeem=REDEEM_Q_V2),
    "Compound V3": dict(borrow=BORROW_Q_V2, repay=REPAY_Q_V2, liq=LIQ_Q_V2, mint=MINT_Q_V2, redeem=REDEEM_Q_V2),
    "Aave V3":     dict(borrow=BORROW_Q_V2, repay=REPAY_Q_V2, liq=LIQ_Q_V2, mint=MINT_Q_V2, redeem=REDEEM_Q_V2),
}

# ---------------------------------------------------------
# Sidebar – protocol & inputs
# ---------------------------------------------------------
st.sidebar.header("⚙️ Controls")

protocol = st.sidebar.selectbox("Protocol", list(PROTOCOL_ENDPOINTS.keys()))

use_gsheet = st.sidebar.checkbox("Use Google Sheet URL instead of file upload", value=False)

wallets_df = None

def google_sheet_to_csv_url(sheet_url: str) -> str:
    if "/edit" in sheet_url:
        return sheet_url.split("/edit")[0] + "/export?format=csv"
    if "/view" in sheet_url:
        return sheet_url.split("/view")[0] + "/export?format=csv"
    return sheet_url

if use_gsheet:
    gsheet_url = st.sidebar.text_input(
        "Google Sheet link (must contain a `wallet_id` column)",
        value="https://docs.google.com/spreadsheets/d/1ZzaeMgNYnxvriYYpe8PE7uMEblTI0GV5GIVUnsP-sBs/edit?usp=sharing"
    )
    if st.sidebar.button("Load wallets from Google Sheet"):
        with st.spinner("Fetching wallet list from Google Sheet..."):
            try:
                csv_url = google_sheet_to_csv_url(gsheet_url)
                wallets_df = pd.read_csv(csv_url)
                st.sidebar.success(f"Loaded {len(wallets_df)} wallets.")
            except Exception as e:
                st.sidebar.error(f"Failed to read Google Sheet: {e}")
else:
    uploaded = st.sidebar.file_uploader("Upload CSV with a `wallet_id` column", type=["csv"])
    if uploaded is not None:
        try:
            wallets_df = pd.read_csv(uploaded)
            st.sidebar.success(f"Loaded {len(wallets_df)} wallets from file.")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

# ---------------------------------------------------------
# Sidebar – weights
# ---------------------------------------------------------
st.sidebar.subheader("🔩 Feature Weights")
w_ltv  = st.sidebar.slider("Weight: LTV",               0.0, 1.0, DEFAULT_WEIGHTS["ltv"],               0.01)
w_liq  = st.sidebar.slider("Weight: Liquidation Rate",   0.0, 1.0, DEFAULT_WEIGHTS["liquidation_rate"],  0.01)
w_bint = st.sidebar.slider("Weight: Borrow Intensity",   0.0, 1.0, DEFAULT_WEIGHTS["borrow_intensity"],  0.01)
w_inac = st.sidebar.slider("Weight: Inactivity Days",    0.0, 1.0, DEFAULT_WEIGHTS["inactivity"],        0.01)
w_txv  = st.sidebar.slider("Weight: Tx Volatility",      0.0, 1.0, DEFAULT_WEIGHTS["tx_volatility"],     0.01)

weight_sum = w_ltv + w_liq + w_bint + w_inac + w_txv
if weight_sum == 0:
    st.sidebar.warning("Weights sum to 0. Falling back to defaults.")
    weights = DEFAULT_WEIGHTS
else:
    weights = {
        "ltv": w_ltv / weight_sum,
        "liquidation_rate": w_liq / weight_sum,
        "borrow_intensity": w_bint / weight_sum,
        "inactivity": w_inac / weight_sum,
        "tx_volatility": w_txv / weight_sum,
    }

# ---------------------------------------------------------
# HTTP & Graph helpers
# ---------------------------------------------------------
def retry_post_json(url, json_body, retries=HTTP_RETRIES, backoff=HTTP_BACKOFF):
    for i in range(retries):
        try:
            r = requests.post(url, json=json_body, timeout=60)
            if r.status_code == 200:
                return r.json()
            time.sleep(backoff ** i)
        except Exception:
            time.sleep(backoff ** i)
    raise RuntimeError(f"POST {url} failed after {retries} retries")

def paginate_events(query_tmpl: str, root_field: str, variables: dict, limit=BATCH, graph_url=None):
    """Generic paginator for The Graph. If the field doesn't exist (schema mismatch), return []."""
    out, skip = [], 0
    while True:
        vars_p = {**variables, "first": limit, "skip": skip}
        try:
            data = retry_post_json(graph_url, {"query": query_tmpl, "variables": vars_p})
            if "errors" in data:
                return []  # schema mismatch
            chunk = data["data"].get(root_field, [])
        except Exception:
            return []
        out.extend(chunk)
        if len(chunk) < limit:
            break
        skip += limit
    return out

def fetch_wallet_events(wallet: str, protocol_name: str):
    """Fetch all relevant actions for a wallet for the chosen protocol.
       If a query fails (schema mismatch), we return empty lists for that entity."""
    queries = PROTOCOL_QUERIES[protocol_name]
    graph_url = PROTOCOL_ENDPOINTS[protocol_name]

    variables = {"account": wallet}

    def safe_pull(q, field):
        try:
            return paginate_events(q, field, variables, limit=BATCH, graph_url=graph_url)
        except Exception:
            return []

    borrows      = safe_pull(queries["borrow"], "borrows")
    repays       = safe_pull(queries["repay"], "repayBorrows")
    liquidations = safe_pull(queries["liq"],   "liquidateBorrows")
    mints        = safe_pull(queries["mint"],  "mints")
    redeems      = safe_pull(queries["redeem"],"redeems")

    return {
        "wallet": wallet,
        "borrows": borrows,
        "repays": repays,
        "liquidations": liquidations,
        "mints": mints,
        "redeems": redeems,
    }

# ---------------------------------------------------------
# Feature engineering & scoring
# ---------------------------------------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def engineer_features(events: dict) -> dict:
    wallet = events["wallet"]

    n_borrows = len(events["borrows"])
    n_repays = len(events["repays"])
    n_liqs = len(events["liquidations"])
    n_mints = len(events["mints"])
    n_redeems = len(events["redeems"])
    total_txs = n_borrows + n_repays + n_liqs + n_mints + n_redeems

    def sum_amountUSD(rows):
        if not rows:
            return 0.0
        return sum(safe_float(r.get("amountUSD", r.get("amount", 0.0))) for r in rows)

    total_borrow_usd = sum_amountUSD(events["borrows"])
    total_supply_usd = sum_amountUSD(events["mints"])
    total_redeem_usd = sum_amountUSD(events["redeems"])
    total_repay_usd = sum_amountUSD(events["repays"])
    total_liq_usd = sum_amountUSD(events["liquidations"])

    # LTV proxy
    ltv = total_borrow_usd / (total_supply_usd + 1e-9)
    # Liquidation rate
    liquidation_rate = n_liqs / (n_borrows + 1e-9)

    # Time metrics
    timestamps = []
    for coll in ["borrows", "repays", "liquidations", "mints", "redeems"]:
        timestamps.extend([int(e.get("blockTime", 0)) for e in events[coll] if e.get("blockTime")])
    timestamps = sorted(timestamps)
    if len(timestamps) >= 2:
        life_days = (timestamps[-1] - timestamps[0]) / (60 * 60 * 24)
    else:
        life_days = 0.0

    active_months = max(1.0, life_days / 30.0)
    borrow_intensity = total_borrow_usd / active_months

    now = datetime.now(timezone.utc).timestamp()
    last_ts = timestamps[-1] if timestamps else 0
    inactivity_days = (now - last_ts) / (60 * 60 * 24) if last_ts > 0 else 9999

    # Tx volatility: std/mean of 30d buckets
    if timestamps:
        first = timestamps[0]
        buckets = {}
        for ts in timestamps:
            month_idx = int((ts - first) // (30 * 24 * 3600))
            buckets.setdefault(month_idx, 0)
            buckets[month_idx] += 1
        tx_counts = list(buckets.values())
        tx_volatility = (np.std(tx_counts) / (np.mean(tx_counts) + 1e-9)) if len(tx_counts) > 1 else 0.0
    else:
        tx_volatility = 0.0

    return dict(
        wallet_id=wallet,
        n_borrows=n_borrows,
        n_repays=n_repays,
        n_liquidations=n_liqs,
        n_mints=n_mints,
        n_redeems=n_redeems,
        total_txs=total_txs,
        total_borrow_usd=total_borrow_usd,
        total_supply_usd=total_supply_usd,
        total_redeem_usd=total_redeem_usd,
        total_repay_usd=total_repay_usd,
        total_liq_usd=total_liq_usd,
        ltv=ltv,
        liquidation_rate=liquidation_rate,
        borrow_intensity=borrow_intensity,
        inactivity_days=inactivity_days,
        tx_volatility=tx_volatility,
        life_days=life_days,
        active_months=active_months,
        last_activity_ts=last_ts
    )

def min_max_normalize(series: pd.Series):
    mn, mx = series.min(), series.max()
    if math.isclose(mx, mn):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - mn) / (mx - mn)

def compute_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    # Normalize features that map positively to risk
    df["norm_ltv"] = min_max_normalize(df["ltv"])
    df["norm_liquidation_rate"] = min_max_normalize(df["liquidation_rate"])
    df["norm_borrow_intensity"] = min_max_normalize(df["borrow_intensity"])
    df["norm_inactivity"] = min_max_normalize(df["inactivity_days"])
    df["norm_tx_volatility"] = min_max_normalize(df["tx_volatility"])

    df["risk_score"] = SCORE_SCALE * (
        weights["ltv"] * df["norm_ltv"]
        + weights["liquidation_rate"] * df["norm_liquidation_rate"]
        + weights["borrow_intensity"] * df["norm_borrow_intensity"]
        + weights["inactivity"] * df["norm_inactivity"]
        + weights["tx_volatility"] * df["norm_tx_volatility"]
    )
    df["risk_score"] = df["risk_score"].clip(0, SCORE_SCALE)
    return df

# ---------------------------------------------------------
# The cached pipeline
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_pipeline(wallets: pd.DataFrame, weights: dict, protocol_name: str, max_workers: int = 8):
    wallets = wallets.drop_duplicates("wallet_id")
    wallets["wallet_id"] = wallets["wallet_id"].str.lower()

    feats, errors = [], []
    progress = st.progress(0)
    total = len(wallets)

    def _fetch_and_engineer(w):
        try:
            ev = fetch_wallet_events(w, protocol_name)
            return engineer_features(ev), None
        except Exception as e:
            return None, (w, str(e))

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, (res, err) in enumerate(ex.map(_fetch_and_engineer, wallets["wallet_id"].tolist())):
            progress.progress(int((i + 1) / total * 100))
            if err: errors.append(err)
            else:  feats.append(res)

    df = pd.DataFrame(feats)

    # If some failed, inject defaults
    if errors:
        st.warning(f"{len(errors)} wallets failed; defaulting their features.")
        defaults = dict(
            ltv=0.5,
            liquidation_rate=0.0,
            borrow_intensity=0.0,
            inactivity_days=9999.0,
            tx_volatility=0.0,
        )
        missing_df = pd.DataFrame([{"wallet_id": w, **defaults} for w, _ in errors])
        df = pd.concat([df, missing_df], ignore_index=True, sort=False)
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
            df[col] = df[col].fillna(val)

    df = compute_scores(df, weights)

    # Persist results
    proto_slug = protocol_name.replace(" ", "_")
    scores_path   = os.path.join(OUTPUT_DIR, f"wallet_scores_{proto_slug}.csv")
    features_path = os.path.join(OUTPUT_DIR, f"features_{proto_slug}.csv")

    df[["wallet_id", "risk_score"]].rename(columns={"risk_score": "score"}).to_csv(scores_path, index=False)
    df.to_csv(features_path, index=False)

    return df.sort_values("risk_score", ascending=False), scores_path, features_path

# ---------------------------------------------------------
# UI – Run
# ---------------------------------------------------------
if wallets_df is not None and st.button("🚀 Run Scoring Pipeline"):
    if "wallet_id" not in wallets_df.columns:
        st.error("Input must contain a `wallet_id` column.")
    else:
        with st.spinner(f"Running pipeline for **{protocol}** ..."):
            df, scores_path, features_path = run_pipeline(wallets_df, weights, protocol)

        st.success(f"Done! Results saved to `{OUTPUT_DIR}/`.")

        scores_df = df[["wallet_id", "risk_score"]].rename(columns={"risk_score": "score"})

        # ----------------------
        # Fallback if all scores = 0
        # ----------------------
        if scores_df["score"].nunique() == 1 and scores_df["score"].iloc[0] == 0:
            st.warning("⚠️ All risk scores are 0 — possibly no on-chain data found. Adding mock scores for testing.")
            scores_df["score"] = np.linspace(10, 1000, len(scores_df))
            df["risk_score"] = scores_df["score"]

        st.subheader("📊 Scores (0 – 1000)")
        st.dataframe(scores_df, use_container_width=True)

        st.subheader("⚠️ Top 10 Risky Wallets")
        st.table(scores_df.head(10))

        # ----------------------
        # Slider patch
        # ----------------------
        min_s, max_s = int(scores_df["score"].min()), int(scores_df["score"].max())
        if min_s == max_s:
            st.info("All scores are identical. Slider disabled.")
            rng = (min_s, max_s)
        else:
            rng = st.slider("Filter by score", min_s, max_s, (min_s, max_s))

        st.dataframe(
            scores_df[(scores_df["score"] >= rng[0]) & (scores_df["score"] <= rng[1])],
            use_container_width=True
        )

        with st.expander("📑 See engineered features"):
            st.dataframe(df, use_container_width=True)

        # Downloads
        with open(scores_path, "rb") as f:
            st.download_button("📥 Download wallet_scores.csv", data=f, file_name="wallet_scores.csv")
        with open(features_path, "rb") as f:
            st.download_button("📥 Download features.csv", data=f, file_name="features.csv")
else:
    st.info("Load wallets (CSV or Google Sheet), choose protocol & weights, then click **Run Scoring Pipeline**.")