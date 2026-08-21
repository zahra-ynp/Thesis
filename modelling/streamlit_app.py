import hashlib
import hmac
import os
import urllib.parse
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text


st.set_page_config(page_title="Macro & News Dashboard", page_icon="📊", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = (PROJECT_ROOT / ".." / "data").resolve()
DEFAULT_NEWS_FILE = DATA_ROOT / "raw" / "news" / "FinSen_US_Categorized_Timestamp.csv"


@st.cache_resource
def get_engine():
    server = os.getenv("DB_SERVER", r"Lara")
    database = os.getenv("DB_DATABASE", "MacroSentimentDB")
    driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    conn = (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
        "Trusted_Connection=yes;TrustServerCertificate=yes;"
    )
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn)}",
        pool_pre_ping=True,
    )


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify PBKDF2 hashes and the project's legacy SHA-256 hashes."""
    if not password or not stored_hash or not salt:
        return False
    if stored_hash.startswith("pbkdf2_sha256$"):
        _, iterations, digest = stored_hash.split("$", 2)
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), int(iterations)
        ).hex()
        return hmac.compare_digest(candidate, digest)

    candidates = (
        hashlib.sha256((salt + password).encode()).hexdigest(),
        hashlib.sha256((password + salt).encode()).hexdigest(),
    )
    return any(hmac.compare_digest(value, stored_hash) for value in candidates)


def authenticate(username: str, password: str):
    sql = text("""
        SELECT TOP (1) user_id, username, password_hash, salt, email
        FROM auth.vw_user_login
        WHERE username = :username AND is_active = 1
    """)
    with get_engine().connect() as connection:
        row = connection.execute(sql, {"username": username}).mappings().first()
        if not row or not verify_password(password, row["password_hash"], row["salt"]):
            return None
        permissions = pd.read_sql(
            text("""
                SELECT role_name, company_id, company_name, region_id, region_name
                FROM auth.vw_user_permissions
                WHERE user_id = :user_id
            """),
            connection,
            params={"user_id": row["user_id"]},
        )
    return {
        "user_id": row["user_id"],
        "username": row["username"],
        "email": row["email"],
        "permissions": permissions.to_dict("records"),
    }


def login_page():
    _, login_column, _ = st.columns([1, 1.15, 1])
    with login_column:
        st.title("Macro & News Dashboard")
        st.caption("Sign in to view the dashboard.")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
    if submitted:
        try:
            user = authenticate(username.strip(), password)
        except Exception:
            st.error("The database is unavailable. Check the connection settings.")
            return
        if user:
            st.session_state.user = user
            st.rerun()
        else:
            st.error("Invalid username or password.")


def allowed_regions(user):
    grants = user["permissions"]
    if any(g["role_name"] == "admin" and g["region_id"] is None for g in grants):
        return None
    scoped = {g["region_id"] for g in grants if g["region_id"] is not None}
    return scoped or None


def load_regions(region_scope):
    query = "SELECT region_id, region_name FROM core.source_regions"
    params = {}
    if region_scope:
        names = []
        for index, region_id in enumerate(sorted(region_scope)):
            key = f"r{index}"
            names.append(f":{key}")
            params[key] = region_id
        query += f" WHERE region_id IN ({','.join(names)})"
    query += " ORDER BY region_name"
    return pd.read_sql(text(query), get_engine(), params=params)


def load_dashboard(start_date, end_date, region_ids):
    params = {"start": start_date, "end": end_date}
    placeholders = []
    for index, region_id in enumerate(region_ids):
        key = f"region{index}"
        placeholders.append(f":{key}")
        params[key] = region_id
    region_filter = f"AND region_id IN ({','.join(placeholders)})" if placeholders else ""
    with get_engine().connect() as connection:
        has_active_view = bool(connection.execute(
            text("SELECT CASE WHEN OBJECT_ID('analytics.Active_Daily_News', 'V') IS NULL THEN 0 ELSE 1 END")
        ).scalar())
        has_article_url = bool(connection.execute(
            text("SELECT CASE WHEN COL_LENGTH('core.daily_news', 'article_url') IS NULL THEN 0 ELSE 1 END")
        ).scalar())
        news_source = "analytics.Active_Daily_News" if has_active_view else "core.daily_news"
        article_url_field = "article_url" if has_article_url else "CAST(NULL AS VARCHAR(1000)) AS article_url"

        monthly = pd.read_sql(
            text(f"""
                SELECT * FROM analytics.Monthly_Region_Wide_View
                WHERE month_date BETWEEN :start AND :end {region_filter}
                ORDER BY month_date, region_name
            """),
            connection,
            params=params,
            parse_dates=["month_date"],
        )
        news = pd.read_sql(
            text(f"""
                SELECT article_id, news_date, title, source_name, region_id,
                       sentiment_score, sentiment_label, lake_pointer_url,
                       {article_url_field}
                FROM {news_source}
                WHERE news_date BETWEEN :start AND :end {region_filter}
                ORDER BY news_date DESC
            """),
            connection,
            params=params,
            parse_dates=["news_date"],
        )
    return monthly, news


@st.cache_data(show_spinner=False)
def read_news_file(file_path: str):
    return pd.read_csv(file_path, encoding="utf-8")


def get_article_content(article):
    """Return the raw article matching the selected database row."""
    stored_path = Path(str(article.get("lake_pointer_url") or ""))
    candidates = [stored_path, DEFAULT_NEWS_FILE]

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except (OSError, RuntimeError):
            continue

        # Never allow a database value to make the app read outside project data.
        is_project_data = DATA_ROOT == resolved or DATA_ROOT in resolved.parents
        if not is_project_data or not resolved.is_file():
            continue

        try:
            raw = read_news_file(str(resolved))
        except (OSError, UnicodeError, pd.errors.ParserError):
            continue

        if {"title", "news_date", "text_for_finbert"}.issubset(raw.columns):
            title_match = raw["title"].fillna("").astype(str).str.strip() == str(article["title"]).strip()
            matches = raw.loc[title_match].copy()
            if matches.empty:
                continue
            raw_dates = pd.to_datetime(matches["news_date"], errors="coerce").dt.date
            article_date = pd.Timestamp(article["news_date"]).date()
            dated = matches.loc[raw_dates == article_date]
            selected = dated.iloc[0] if not dated.empty else matches.iloc[0]
            return {
                "title": selected["title"],
                "tag": selected.get("news_desk", "New York Times API metadata"),
                "date": selected["news_date"],
                "content": selected["text_for_finbert"],
            }

        required = {"Title", "Time", "Content"}
        if not required.issubset(raw.columns):
            continue

        title_match = (
            raw["Title"].fillna("").astype(str).str.strip()
            == str(article["title"]).strip()
        )
        matches = raw.loc[title_match].copy()
        if matches.empty:
            continue

        raw_dates = pd.to_datetime(matches["Time"], dayfirst=True, errors="coerce").dt.date
        article_date = pd.Timestamp(article["news_date"]).date()
        dated = matches.loc[raw_dates == article_date]
        selected = dated.iloc[0] if not dated.empty else matches.iloc[0]
        return {
            "title": selected["Title"],
            "tag": selected.get("Tag", ""),
            "date": selected["Time"],
            "content": selected["Content"],
        }
    return None


def show_article(article_row):
    article = get_article_content(article_row)
    article_url = article_row.get("article_url")
    has_article_url = pd.notna(article_url) and bool(str(article_url).strip())
    if article:
        with st.container(border=True):
            st.subheader(article["title"])
            details = " · ".join(
                str(value) for value in (article["date"], article["tag"]) if value
            )
            st.caption(details)
            st.write(article["content"])
            if has_article_url:
                st.link_button("Open original article", str(article_url))
    else:
        if has_article_url:
            st.link_button("Open original article", str(article_url))
        else:
            st.warning("The article text could not be found in the linked source file.")


def load_accessible_assets(user, roles):
    grants = user["permissions"]
    params = {}
    restrictions = []

    if "admin" in roles:
        has_global_admin = any(
            grant["role_name"] == "admin" and grant["region_id"] is None
            for grant in grants
        )
        if not has_global_admin:
            region_ids = sorted(
                {
                    int(grant["region_id"])
                    for grant in grants
                    if grant["role_name"] == "admin"
                    and grant["region_id"] is not None
                }
            )
            placeholders = []
            for index, region_id in enumerate(region_ids):
                key = f"asset_region_{index}"
                placeholders.append(f":{key}")
                params[key] = region_id
            if not placeholders:
                return pd.DataFrame()
            restrictions.append(f"a.region_id IN ({','.join(placeholders)})")
    elif "company" in roles:
        company_ids = sorted(
            {
                int(grant["company_id"])
                for grant in grants
                if grant["role_name"] == "company"
                and grant["company_id"] is not None
            }
        )
        placeholders = []
        for index, company_id in enumerate(company_ids):
            key = f"asset_company_{index}"
            placeholders.append(f":{key}")
            params[key] = company_id
        if not placeholders:
            return pd.DataFrame()
        restrictions.append(f"a.company_id IN ({','.join(placeholders)})")
    else:
        return pd.DataFrame()

    where_clause = f"WHERE {' AND '.join(restrictions)}" if restrictions else ""
    return pd.read_sql(
        text(f"""
            SELECT a.asset_id, a.asset_name, a.asset_class, r.region_name,
                   c.company_name, a.market_value, a.currency_code,
                   a.as_of_date, a.notes
            FROM core.assets a
            LEFT JOIN core.source_regions r ON a.region_id = r.region_id
            LEFT JOIN auth.companies c ON a.company_id = c.company_id
            {where_clause}
            ORDER BY a.as_of_date DESC, a.asset_name
        """),
        get_engine(),
        params=params,
        parse_dates=["as_of_date"],
    )


def show_dashboard(user):
    roles = {grant["role_name"] for grant in user["permissions"]}
    is_admin = "admin" in roles
    with st.sidebar:
        st.success(f"Signed in as {user['username']}")
        st.caption("Access: " + ", ".join(sorted(roles)))
        companies = sorted(
            {
                grant["company_name"]
                for grant in user["permissions"]
                if grant["company_name"]
            }
        )
        if companies:
            st.info("Company: " + ", ".join(companies))
        if st.button("Log out", use_container_width=True):
            st.query_params.clear()
            st.session_state.clear()
            st.rerun()

    st.title("Macroeconomic Indicators & News Sentiment")
    scope = allowed_regions(user)
    regions = load_regions(scope)
    if regions.empty:
        st.warning("No regions are assigned to this account.")
        return

    today = pd.Timestamp.today().date()
    c1, c2, c3 = st.columns([1, 1, 2])
    start_date = c1.date_input("From", value=pd.Timestamp("2007-01-01").date())
    end_date = c2.date_input("To", value=today)
    individual_region_names = [
        name for name in regions["region_name"].tolist() if name != "Global"
    ]
    region_choice = c3.selectbox(
        "Region",
        options=["Global"] + individual_region_names,
        index=0,
        help="Global shows every region permitted for this account.",
    )
    selected_names = (
        regions["region_name"].tolist()
        if region_choice == "Global"
        else [region_choice]
    )
    region_ids = regions.loc[
        regions["region_name"].isin(selected_names), "region_id"
    ].astype(int).tolist()
    if start_date > end_date:
        st.error("The start date must be before the end date.")
        return

    monthly, news = load_dashboard(start_date, end_date, region_ids)
    if monthly.empty:
        st.info("No monthly observations match these filters.")
        return

    chart_settings = {
        "cpi_value": ("Consumer Price Index", "Index value", "#2563EB"),
        "unemployment_rate": ("Unemployment Rate", "Percent", "#DC2626"),
        "fed_funds_rate": ("Federal Funds Rate", "Percent", "#7C3AED"),
        "industrial_production_value": (
            "Industrial Production Index",
            "Index value",
            "#059669",
        ),
    }
    period_label = (
        f"{pd.Timestamp(start_date):%b %Y} – {pd.Timestamp(end_date):%b %Y}"
    )
    st.subheader("Monthly macroeconomic indicators")
    for field, (title, y_label, color) in chart_settings.items():
        chart_data = monthly[["month_date", "region_name", field]].dropna()
        indicator_figure = px.line(
            chart_data,
            x="month_date",
            y=field,
            color="region_name",
            markers=True,
            color_discrete_sequence=[
                color,
                "#F59E0B",
                "#0891B2",
                "#DB2777",
            ],
            title=f"{title}<br><sup>Monthly observations · {period_label}</sup>",
            labels={
                "month_date": "Observation month",
                field: y_label,
                "region_name": "Region",
            },
        )
        indicator_figure.update_layout(
            legend_title_text="Region",
            hovermode="x unified",
            margin={"l": 20, "r": 20, "t": 75, "b": 20},
        )
        st.plotly_chart(indicator_figure, use_container_width=True)

    st.subheader("Monthly news sentiment")
    sentiment = (
        news.assign(month=lambda frame: frame["news_date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "region_id"], as_index=False)["sentiment_score"]
        .mean()
        .merge(regions, on="region_id", how="left")
    )
    sentiment_figure = px.line(
        sentiment,
        x="month",
        y="sentiment_score",
        color="region_name",
        markers=True,
        color_discrete_sequence=["#F59E0B", "#0891B2", "#DB2777", "#65A30D"],
        title=f"News sentiment by region<br><sup>Monthly observations · {period_label}</sup>",
        labels={
            "month": "Publication month",
            "sentiment_score": "Sentiment score",
            "region_name": "Region",
        },
    )
    sentiment_figure.add_hline(
        y=0,
        line_dash="dot",
        line_color="#6B7280",
        annotation_text="Neutral",
    )
    sentiment_figure.update_layout(
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 75, "b": 20},
    )
    st.plotly_chart(sentiment_figure, use_container_width=True)
    st.caption("Use the Open article button to display the full content on this page.")
    requested_article_id = st.query_params.get("article_id")
    if requested_article_id and str(requested_article_id).isdigit():
        requested_id = int(requested_article_id)
        if requested_id in news["article_id"].values:
            st.session_state.selected_article_id = requested_id

    page_size = 20
    total_pages = max(1, (len(news) + page_size - 1) // page_size)
    news_page = st.number_input(
        "News page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
    )
    page_start = (int(news_page) - 1) * page_size
    page_news = news.iloc[page_start : page_start + page_size]
    st.caption(f"Page {int(news_page)} of {total_pages} · {len(news):,} matching articles")

    header = st.columns([1.2, 4.2, 1.4, 1.2, 1.1])
    for column, label in zip(
        header, ["Date", "News title", "Source", "Sentiment", "Article"]
    ):
        column.markdown(f"**{label}**")
    st.divider()

    for article in page_news.itertuples(index=False):
        row = st.columns([1.2, 4.2, 1.4, 1.2, 1.1], vertical_alignment="center")
        row[0].write(f"{pd.Timestamp(article.news_date):%d %b %Y}")
        row[1].write(article.title)
        row[2].write(article.source_name or "—")
        row[3].write(
            f"{article.sentiment_label or '—'} ({article.sentiment_score:.3f})"
            if pd.notna(article.sentiment_score)
            else article.sentiment_label or "—"
        )
        if row[4].button(
            "Open article",
            key=f"open_article_{article.article_id}",
            use_container_width=True,
        ):
            st.session_state.selected_article_id = int(article.article_id)
            st.query_params["article_id"] = str(article.article_id)

    selected_article_id = st.session_state.get("selected_article_id")
    if selected_article_id is not None:
        selected = news.loc[news["article_id"] == int(selected_article_id)]
        if not selected.empty:
            show_article(selected.iloc[0])
        else:
            st.session_state.pop("selected_article_id", None)
            st.query_params.pop("article_id", None)

    if is_admin or "company" in roles:
        st.subheader("Accessible assets")
        assets = load_accessible_assets(user, roles)
        if assets.empty:
            st.info("No assets are assigned to your company or region.")
        else:
            st.dataframe(assets, use_container_width=True, hide_index=True)


if "user" not in st.session_state:
    if "article_id" in st.query_params:
        st.query_params.pop("article_id", None)
    st.session_state.pop("selected_article_id", None)
    login_page()
else:
    show_dashboard(st.session_state.user)
