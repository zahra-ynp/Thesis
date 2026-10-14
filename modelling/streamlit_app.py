"""Streamlit research dashboard backed exclusively by saved project files."""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modelling.dashboard_data import (
    REGIONS, STATE_LABELS, US_WORKBOOK, article_text, filter_news, filter_period,
    load_monthly, load_news, load_states, original_url,
)

st.set_page_config(page_title="GMS · Global Macro Signals", page_icon="◉", layout="wide")
RED, INK = "#C8102E", "#222A35"
COLORS = ["#64748B", RED]


@st.cache_data(show_spinner=False, ttl=300)
def region_data(region):
    """Load report-aligned Europe inflation and US variance-only state estimates."""
    monthly = load_monthly(region)
    states, model = load_states(region)
    news, missing = load_news(region)
    return monthly, states, model, news, missing


def style_chart(figure, y_title, height=340):
    figure.update_layout(
        template="plotly_white", height=height,
        font=dict(family="Arial, sans-serif", color=INK, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=16, t=20, b=20), hovermode="x unified",
        legend=dict(orientation="h", y=1.13, x=0, title=None),
        xaxis=dict(title=None, showgrid=False, zeroline=False),
        yaxis=dict(title=y_title, gridcolor="#ECEEF2", zerolinecolor="#D5D9DF"),
    )
    return figure


def line_chart(frame, field, y_title, color=RED):
    figure = go.Figure(go.Scatter(
        x=frame["Date"], y=frame[field], name=y_title, mode="lines",
        line=dict(color=color, width=2.5), connectgaps=False,
        hovertemplate="%{x|%b %Y}<br>%{y:.2f}<extra></extra>",
    ))
    return style_chart(figure, y_title)


def plot(figure, key):
    st.plotly_chart(figure, width="stretch", key=key,
                    config={"displayModeBar": False, "responsive": True})


def metric(column, frame, field, label, suffix="%"):
    available = frame.dropna(subset=[field])
    if available.empty:
        column.metric(label, "Unavailable")
        return
    last = available.iloc[-1]
    with column:
        st.metric(label, f"{last[field]:.2f}{suffix}")
        st.caption(f"Latest available month · {last['Date']:%b %Y}")


def show_history(monthly, region):
    st.subheader("Inflation through time")
    st.caption(f"{REGIONS[region]['price_index']}-based inflation · observed monthly data")
    measure = st.radio("Inflation measure", ["Year over year", "Month over month"],
                       horizontal=True, key=f"inflation_{region}")
    field = "Inflation_YoY" if measure == "Year over year" else "Inflation_MoM"
    if monthly[field].notna().any():
        plot(line_chart(monthly, field, "Inflation (%)"), "inflation_history")
    else:
        st.info("Inflation observations are unavailable for this period.")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Labour market")
        plot(line_chart(monthly, "Unemployment_Rate", "Unemployment (%)", INK), "unemployment")
    with right:
        st.subheader("Industrial activity")
        plot(line_chart(monthly, "Industrial_Production", "Production index", "#8B5A66"), "production")
    st.subheader("The tone of the news")
    sentiment = line_chart(monthly, "News_Sentiment", "Net sentiment")
    sentiment.add_hline(y=0, line_dash="dot", line_color="#A4ABB6")
    plot(sentiment, "sentiment_history")
    st.caption("Monthly FinBERT net sentiment: P(positive) − P(negative), from −1 to +1.")
    with st.expander("View historical observations"):
        fields = ["Date", "Inflation_MoM", "Inflation_YoY", "Unemployment_Rate",
                  "Industrial_Production", "News_Sentiment", "Article_Count"]
        shown = monthly[fields]
        st.dataframe(shown, width="stretch", hide_index=True,
                     column_config={"Date": st.column_config.DateColumn("Month", format="MMM YYYY")})
        st.download_button("Download observations · CSV", shown.to_csv(index=False).encode("utf-8"),
                           file_name=f"{region.lower()}_historical_observations.csv", mime="text/csv")


def show_states(states, model, selected_start, selected_end, region):
    labels = STATE_LABELS[region]
    state_kind = REGIONS[region]["state_kind"]
    st.subheader(f"Estimated {state_kind} states")
    st.caption(model)
    if region == "US":
        st.caption("State 1: Low volatility · State 2: High volatility. "
                   "The intercept and AR coefficient are common to both states; "
                   "the states differ in the variance of inflation shocks. "
                   "Probabilities are predicted before observing each month's inflation.")
    else:
        st.caption("State 1 (Level 1): Low inflation · State 2 (Level 2): High inflation. "
                   "These are the model's lower and higher inflation regimes. "
                   "Probabilities are predicted before observing each month's inflation.")
    frame = filter_period(states, selected_start, selected_end)
    if frame.empty:
        st.info(f"Saved estimates cover {states['Date'].min():%b %Y}–{states['Date'].max():%b %Y}. "
                "Choose a period within that range to view the states.")
        return
    last = frame.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest estimated state", labels[last["PredictedRegime"]].split(" · ", 1)[-1])
    c1.caption(f"{labels[last['PredictedRegime']]} · {last['Date']:%b %Y}")
    probability = max(last[f"Probability_{label}"] for label in labels)
    c2.metric("Probability of that state", f"{probability:.1%}")
    c2.caption("Model probability for the selected month")
    c3.metric("Monthly inflation estimate", f"{last['Forecast']:.2f}%")
    c3.caption(f"Observed: {last['Actual']:.2f}% · {last['Date']:%b %Y}")
    figure = go.Figure()
    for index, label in enumerate(labels):
        figure.add_trace(go.Scatter(
            x=frame["Date"], y=frame[f"Probability_{label}"], name=labels[label],
            mode="lines", stackgroup="states", line=dict(color=COLORS[index], width=1),
            hovertemplate="%{x|%b %Y}<br>" + labels[label] + ": %{y:.1%}<extra></extra>",
        ))
    style_chart(figure, "State probability")
    figure.update_yaxes(range=[0, 1], tickformat=".0%")
    plot(figure, "state_probabilities")
    st.subheader("Observed and estimated inflation")
    comparison = go.Figure()
    for field, label, color, dash in (("Actual", "Observed inflation", INK, "solid"),
                                       ("Forecast", "Model estimate", "#9CA3AF", "solid")):
        comparison.add_trace(go.Scatter(x=frame["Date"], y=frame[field], name=label,
                                        line=dict(color=color, width=2.5, dash=dash),
                                        hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>"))
    for index, label in enumerate(labels):
        selected = frame.loc[frame["PredictedRegime"].eq(label)]
        comparison.add_trace(go.Scatter(
            x=selected["Date"], y=selected["Forecast"], mode="markers", name=labels[label],
            marker=dict(color=COLORS[index], size=8),
            hovertemplate="%{x|%b %Y}<br>" + labels[label] + "<br>%{y:.2f}%<extra></extra>",
        ))
    plot(style_chart(comparison, "Monthly inflation (%)"), "state_forecasts")
    if pd.Timestamp(selected_end) > states["Date"].max() + pd.offsets.MonthEnd(0):
        st.info(f"{region} state estimates end in {states['Date'].max():%B %Y}. "
                "Later historical observations may be available.")
    with st.expander("View monthly state estimates"):
        displayed = frame.assign(EstimatedState=frame["PredictedRegime"].map(labels))
        st.dataframe(displayed.drop(columns="PredictedRegime"), width="stretch", hide_index=True,
                     column_config={"EstimatedState": "Estimated state",
                                    **{f"Probability_{label}": f"{label} probability" if region == "US"
                                       else ("Low inflation probability" if label == "Level 1" else "High inflation probability")
                                       for label in labels}})
        st.download_button("Download state estimates · CSV", displayed.to_csv(index=False).encode("utf-8"),
                           file_name=f"{region.lower()}_estimated_states.csv", mime="text/csv")


@st.dialog("Article reader", width="large")
def show_article(article):
    st.subheader(article["title"])
    section = article.get("section_name")
    section = str(section) if pd.notna(section) and section else ""
    st.caption(f"{article['source']} · {article['news_date']:%d %B %Y}" +
               (f" · {section}" if section else ""))
    text, full_text = article_text(article)
    if text:
        if not full_text:
            st.caption("Saved introductory text. Read the complete article on the publisher's website.")
        st.text(text)
    else:
        st.info("Article text is not included in the saved dataset.")
    url = original_url(article)
    if url:
        st.link_button("Read original article ↗", url, type="primary")


def show_news(news, region, missing):
    st.subheader(f"News from {REGIONS[region]['publisher']}")
    st.caption("Select a headline to open its saved article text.")
    if missing:
        st.info("Some local news files are unavailable. Showing the saved articles that are present.")
    left, middle, right = st.columns([3, 1.3, 1.2])
    query = left.text_input("Search articles", placeholder="Search headlines, text or topics…", key=f"search_{region}")
    label = middle.selectbox("Sentiment", ["All sentiments", "Positive", "Neutral", "Negative", "Unscored"],
                             key=f"sentiment_{region}")
    order = right.selectbox("Sort", ["Newest first", "Oldest first"], key=f"sort_{region}")
    filtered = filter_news(news, query, label)
    if order == "Oldest first":
        filtered = filtered.sort_values("news_date", ascending=True, kind="stable")
    if filtered.empty:
        st.info("No articles match this period and these filters.")
        return
    size = 10
    pages = max(1, (len(filtered) + size - 1) // size)
    signature = hashlib.sha256(
        f"{region}|{query}|{label}|{order}|{news['news_date'].min()}|{news['news_date'].max()}".encode()
    ).hexdigest()[:12]
    page = st.number_input("News page", min_value=1, max_value=pages, value=1, step=1,
                           key=f"page_{signature}")
    st.caption(f"{len(filtered):,} articles · Page {page} of {pages:,}")
    header = st.columns([1.1, 5.3, 1.1])
    for col, title in zip(header, ["PUBLISHED", "HEADLINE", "SENTIMENT"]):
        col.caption(title)
    for _, article in filtered.iloc[(page - 1) * size:page * size].iterrows():
        date, title, sentiment = st.columns([1.1, 5.3, 1.1], vertical_alignment="center")
        date.caption(f"{article['news_date']:%d %b %Y}")
        identity = hashlib.sha256(str(article["external_article_id"]).encode()).hexdigest()[:16]
        if title.button(article["title"], key=f"article_{region}_{identity}", width="stretch"):
            show_article(article.to_dict())
        sentiment.write(article["sentiment_label"])
    st.caption(f"Source: {REGIONS[region]['publisher']} · Original articles remain on the publisher's website.")


def main():
    st.markdown((Path(__file__).with_name("dashboard.css")).read_text(encoding="utf-8"), unsafe_allow_html=True)
    with st.sidebar:
        st.image(str(Path(__file__).with_name("assets") / "generali_logo.png"), width="stretch")
        st.markdown('<div class="brand-mark">G<span>MS</span></div>', unsafe_allow_html=True)
        st.markdown("### Global Macro Signals")
        st.caption("Inflation · States · News")
        st.divider()
        region = st.selectbox("Region", list(REGIONS), key="region")
    try:
        with st.spinner("Loading saved observations…"):
            monthly, states, model, news, missing = region_data(region)
    except (OSError, ValueError, KeyError, ImportError) as error:
        st.error(f"Could not load the saved {region} data. {error}")
        st.stop()
    if monthly.empty:
        st.info(f"No saved observations are available for {region}.")
        st.stop()
    minimum = monthly["Date"].min().date()
    maximum = (monthly["Date"].max() + pd.offsets.MonthEnd(0)).date()
    with st.sidebar:
        period = st.selectbox("Period", ["Last 5 years", "Full history", "Custom period"], key=f"period_{region}")
        default_start = max(minimum, pd.Timestamp(maximum.year - 4, 1, 1).date())
        if period == "Custom period":
            start = st.date_input("From", value=default_start, min_value=minimum, max_value=maximum, key=f"from_{region}")
            end = st.date_input("To", value=maximum, min_value=minimum, max_value=maximum, key=f"to_{region}")
        else:
            start, end = (minimum if period == "Full history" else default_start), maximum
        st.divider()
        st.caption("DATA COVERAGE")
        st.write(f"{minimum:%b %Y} — {maximum:%b %Y}")
        st.caption(f"Economy: {REGIONS[region]['macro_source']}\n\nNews: {REGIONS[region]['publisher']}")
        if region == "US":
            st.download_button("Download US data · Excel", US_WORKBOOK.read_bytes(), file_name=US_WORKBOOK.name,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               width="stretch")
    if start > end:
        st.error("Choose an end date on or after the start date.")
        st.stop()
    visible = filter_period(monthly, start, end)
    visible_news = filter_period(news, start, end, "news_date")
    st.markdown('<div class="eyebrow">MACROECONOMIC RESEARCH</div>', unsafe_allow_html=True)
    st.title(f"{region} at a glance")
    st.caption(f"{REGIONS[region]['area']} · {start:%b %Y} — {end:%b %Y}")
    st.markdown('<div class="intro-line">Explore the economic history, follow estimated economic states, '
                'and read the news behind the numbers.</div>', unsafe_allow_html=True)
    if visible.empty:
        st.info("No monthly observations fall within this period. Choose a wider date range.")
    else:
        st.caption("Latest available monthly observations within the selected period. "
                   "The four cards are not averages over the selected period. "
                   "News sentiment is the average score of articles published in the month shown.")
        cards = st.columns(4)
        metric(cards[0], visible, "Inflation_YoY", "Annual inflation (YoY)")
        metric(cards[1], visible, "Inflation_MoM", "Monthly inflation")
        metric(cards[2], visible, "Unemployment_Rate", "Unemployment")
        metric(cards[3], visible, "News_Sentiment", "News sentiment", "")
        missing_months = visible.loc[visible["Inflation_MoM"].isna(), "Date"]
        if len(missing_months):
            st.caption("Monthly inflation unavailable: " + ", ".join(missing_months.dt.strftime("%b %Y")) +
                       ". Gaps are preserved in the charts.")
    history, estimates, articles = st.tabs(["Historical situation", "Estimated states", "News"])
    with history:
        if not visible.empty:
            show_history(visible, region)
    with estimates:
        show_states(states, model, start, end, region)
    with articles:
        show_news(visible_news, region, missing)
    st.divider()
    st.caption("GMS · Global Macro Signals · Historical observations and model estimates")


if __name__ == "__main__":
    main()
