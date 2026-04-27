"""
dashboard.py
------------
LIVE PIPELINE MONITOR

PURPOSE:
    Streamlit dashboard that connects to Snowflake and shows:
      - When the pipeline last ran and how many records were added
      - New data (latest run) vs previous run side-by-side
      - Sentiment distribution over time
      - Full searchable article table

USAGE:
    streamlit run dashboard.py

Then open http://localhost:8501 in your browser.
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import snowflake.connector
from dotenv import load_dotenv

# Load .env for local development.
# On Streamlit Cloud, credentials come from st.secrets instead.
load_dotenv()

def _get_secret(key: str) -> str:
    """
    Reads a credential from Streamlit secrets (cloud) or os.environ (local).
    This makes the dashboard work identically in both environments.
    """
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, "")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Intelligence Pipeline",
    page_icon="📈",
    layout="wide",
)

# ── Snowflake connection (cached so it doesn't reconnect on every refresh) ──
@st.cache_resource
def get_connection():
    raw_account = _get_secret("SNOWFLAKE_ACCOUNT").split("#")[0].strip()
    if "/" in raw_account:
        parts = raw_account.split("/")
        raw_account = f"{parts[1].strip()}.{parts[0].strip()}"

    return snowflake.connector.connect(
        user=_get_secret("SNOWFLAKE_USER"),
        password=_get_secret("SNOWFLAKE_PASSWORD"),
        account=raw_account,
        warehouse=_get_secret("SNOWFLAKE_WAREHOUSE"),
        database=_get_secret("SNOWFLAKE_DATABASE"),
        schema=_get_secret("SNOWFLAKE_SCHEMA"),
    )

@st.cache_data(ttl=60)   # Re-query Snowflake every 60 seconds on refresh
def load_data() -> pd.DataFrame:
    conn = get_connection()
    query = """
        SELECT
            ID,
            TITLE,
            SOURCE_NAME,
            PUBLISHED_AT,
            SENTIMENT,
            SENTIMENT_SCORE,
            KEY_ENTITY,
            KEY_ENTITY_TYPE,
            ONE_LINE_SUMMARY,
            URL,
            INGESTED_AT
        FROM MARKET_NEWS
        ORDER BY INGESTED_AT DESC
    """
    df = pd.read_sql(query, conn)
    df.columns = df.columns.str.lower()
    df["ingested_at"] = pd.to_datetime(df["ingested_at"])
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df


# ── Helper: identify pipeline runs by grouping rows ingested within 5 min ──
def tag_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups rows into pipeline 'runs' based on ingestion time.
    Rows ingested within 5 minutes of each other = same run.
    Assigns run_label: 'Latest Run', 'Previous Run', 'Older', etc.
    """
    if df.empty:
        return df

    df = df.sort_values("ingested_at", ascending=False).copy()
    df["ingested_at_floor"] = df["ingested_at"].dt.floor("5min")
    run_groups = df["ingested_at_floor"].unique()

    labels = {}
    for i, ts in enumerate(sorted(run_groups, reverse=True)):
        if i == 0:
            labels[ts] = "🟢 Latest Run"
        elif i == 1:
            labels[ts] = "🔵 Previous Run"
        else:
            labels[ts] = f"⚪ Run {i + 1}"

    df["run_label"] = df["ingested_at_floor"].map(labels)
    return df


# ── Sentiment badge color ────────────────────────────────────────────────────
SENTIMENT_COLOR = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ════════════════════════════════════════════════════════════════════════════

st.title("📈 Automated Market Intelligence")
st.caption("Live view of your ETL pipeline — data refreshes every 60 seconds.")

# Load data
try:
    df = load_data()
    df = tag_runs(df)
except Exception as e:
    st.error(f"Could not connect to Snowflake: {e}")
    st.stop()

if df.empty:
    st.warning("No data found. Run `python main.py` to populate the pipeline.")
    st.stop()

# ── Top KPI row ──────────────────────────────────────────────────────────────
latest_run_df = df[df["run_label"] == "🟢 Latest Run"]
prev_run_df   = df[df["run_label"] == "🔵 Previous Run"]
last_run_time = df["ingested_at"].max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Pipeline Run",    last_run_time.strftime("%Y-%m-%d %H:%M UTC"))
col2.metric("Records This Run",     len(latest_run_df))
col3.metric("Total Records",        len(df))
col4.metric("Total Pipeline Runs",  df["ingested_at_floor"].nunique())

st.divider()

# ── New vs Previous run comparison ──────────────────────────────────────────
st.subheader("🆕 Latest Run vs Previous Run")

if latest_run_df.empty:
    st.info("Only one pipeline run detected so far. Run the pipeline again to see a comparison.")
else:
    left, right = st.columns(2)

    def render_run_table(run_df: pd.DataFrame, label: str, container):
        with container:
            st.markdown(f"**{label}**  \n"
                        f"*Ingested at: {run_df['ingested_at'].max().strftime('%Y-%m-%d %H:%M UTC')}*")
            if run_df.empty:
                st.caption("No data for this run yet.")
                return
            for _, row in run_df.iterrows():
                badge = SENTIMENT_COLOR.get(row["sentiment"], "⚪")
                with st.expander(f"{badge} {row['title'][:80]}"):
                    st.write(f"**Sentiment:** {row['sentiment']} (score: {row['sentiment_score']:.2f})")
                    st.write(f"**Key Entity:** {row['key_entity']} [{row['key_entity_type']}]")
                    st.write(f"**Summary:** {row['one_line_summary']}")
                    st.write(f"**Source:** {row['source_name']}  |  Published: {row['published_at']}")
                    st.markdown(f"[Read article]({row['url']})")

    render_run_table(latest_run_df, "🟢 Latest Run", left)
    render_run_table(prev_run_df,   "🔵 Previous Run", right)

st.divider()

# ── Sentiment breakdown chart ────────────────────────────────────────────────
st.subheader("📊 Sentiment Distribution Over Time")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    color_map = {"Bullish": "#22c55e", "Bearish": "#ef4444", "Neutral": "#eab308"}
    fig_pie = px.pie(
        sentiment_counts,
        values="Count",
        names="Sentiment",
        color="Sentiment",
        color_discrete_map=color_map,
        title="Overall Sentiment Split",
    )
    fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    daily = (
        df.groupby([df["ingested_at"].dt.date, "sentiment"])
        .size()
        .reset_index(name="count")
    )
    daily.columns = ["Date", "Sentiment", "Count"]
    fig_bar = px.bar(
        daily,
        x="Date",
        y="Count",
        color="Sentiment",
        color_discrete_map=color_map,
        title="Articles by Sentiment per Pipeline Run Date",
        barmode="stack",
    )
    fig_bar.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Full data table with filters ─────────────────────────────────────────────
st.subheader("🗂️ All Records")

filter_col1, filter_col2, filter_col3 = st.columns(3)
sentiment_filter = filter_col1.multiselect(
    "Filter by Sentiment",
    options=["Bullish", "Bearish", "Neutral"],
    default=["Bullish", "Bearish", "Neutral"],
)
run_filter = filter_col2.multiselect(
    "Filter by Run",
    options=df["run_label"].unique().tolist(),
    default=df["run_label"].unique().tolist(),
)
search = filter_col3.text_input("Search title / entity", "")

filtered = df[
    df["sentiment"].isin(sentiment_filter) &
    df["run_label"].isin(run_filter)
]
if search:
    mask = (
        filtered["title"].str.contains(search, case=False, na=False) |
        filtered["key_entity"].str.contains(search, case=False, na=False)
    )
    filtered = filtered[mask]

st.dataframe(
    filtered[[
        "run_label", "ingested_at", "title", "sentiment",
        "sentiment_score", "key_entity", "source_name",
    ]].rename(columns={
        "run_label":      "Run",
        "ingested_at":    "Ingested At",
        "title":          "Title",
        "sentiment":      "Sentiment",
        "sentiment_score":"Score",
        "key_entity":     "Key Entity",
        "source_name":    "Source",
    }),
    use_container_width=True,
    hide_index=True,
)

st.caption(f"Showing {len(filtered)} of {len(df)} total records.")
