
# Wallet Risk Scoring (Compound / Aave)

This project provides an **end-to-end pipeline** for assessing wallet risk by analyzing on-chain transactions from Compound V2, Compound V3, and Aave V3 protocols. It assigns a risk score between **0–1000** for each wallet.

---

## **1. Data Collection Method**
- We collect transaction data for each wallet using **The Graph** APIs for Compound V2, Compound V3, and Aave V3 subgraphs.
- The pipeline fetches events such as:
  - **Borrow** and **Repay** transactions.
  - **Liquidations** and **Collateral deposits/withdrawals**.
  - **Historical balances** and **health factor data**.
- For each wallet, these events are combined to produce a detailed activity log.

---

## **2. Feature Selection Rationale**
From the raw transaction data, we compute **risk-relevant features** such as:
- **Total Borrowed Amount** – High borrow volume increases risk.
- **Liquidation Count** – Frequent liquidations signal high-risk behavior.
- **Collateral Ratio** – Low collateralization can lead to liquidations.
- **Transaction Frequency** – Infrequent activity may signal abandoned or inactive wallets.
- **Protocol Diversity** – Use of multiple protocols may indicate riskier strategies.

These features were selected because they directly influence the probability of a wallet defaulting or getting liquidated.

---

## **3. Scoring Method**
- **Normalization**: All features are scaled using **Min-Max normalization** to ensure comparability.
- **Weighted Scoring**: Each feature is assigned a weight (e.g., liquidation count has higher weight than borrow frequency).
- **Risk Score Calculation**:
  \\[

  \\text{Risk Score} = \\sum_{i} w_i \\cdot f_i
  \\]
  where \( w_i \) are weights and \( f_i \) are normalized features.

The final score is scaled between **0 (low risk)** and **1000 (high risk)**.

---

## **4. Justification of Risk Indicators**
- **Liquidations** are the clearest indicator of risky behavior since they reflect past failure to maintain healthy collateral.
- **Collateral ratio** and **borrowing habits** are core indicators in lending protocols (as per Aave/Compound risk models).
- **Transaction frequency** and **protocol diversity** are secondary indicators that help identify speculative or inactive wallets.

---

## **5. Scalability**
- The pipeline is modular and can easily integrate additional DeFi protocols.
- Results (wallet features and scores) are cached in CSV files (`/outputs/`) to avoid redundant computations.
- The Streamlit front-end allows non-technical users to upload wallet lists and view risk scores instantly.

---

## **6. Output**
The pipeline produces:
- **wallet_scores.csv** – Each wallet with a final score.
- **features.csv** – Engineered features for transparency.

---

## **7. Deployment**
The app is deployable via **Streamlit Community Cloud**. See [deployment instructions](https://streamlit.io/cloud).

---

## **8. Example CSV Output**
| wallet_id                                | score |
|------------------------------------------|-------|
| 0xfaa0768bde629806739c3a4620656c5d26f44ef2 | 732   |
| 0xabcd1234ef567890abcdef0123456789abcdef01 | 480   |

---

## **Author**
Developed as part of the **Wallet Risk Scoring Assignment (Round 2)**.

