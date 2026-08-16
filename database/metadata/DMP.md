# Data Management Plan (DMP)

**Project:** Macroeconomic Sentiment and Inflation Database

**Author:** Zahra Younes Pour Langaroudi

**Institution:** University of Trieste - Data Science and Artificial Intelligence

**Date:** 2026-07-27

**Version:** 3.1

---

## 1. Scope

This project studies the relationship between macroeconomic conditions and financial-news sentiment from 2007 to 2023.

It combines:

- **FRED:** U.S. Consumer Price Index (`CPIAUCSL`), unemployment (`UNRATE`), federal funds rate (`FEDFUNDS`), and industrial production (`INDPRO`).
- **Eurostat:** European macroeconomic data, including the Harmonised Index of Consumer Prices.
- **Financial news:** FinSen data from Kaggle and Financial News Multisource data from Hugging Face.

FinBERT assigns each article a sentiment label and a score from -1 to +1. The scores are aggregated by month and region and joined with the macroeconomic indicators.

The repository also contains demonstration users, companies, access grants, and assets required by the Streamlit dashboard. These support the application but do not change the research scope.

---

## 2. Storage and Architecture

| Layer | Purpose | Location |
|---|---|---|
| **Raw** | Immutable source news and downloaded data | `data/raw/` |
| **Processed** | Reusable derived files, including precomputed FinBERT scores | `data/processed/` |
| **Core** | Cleaned, normalized research and asset data | SQL Server `core` schema |
| **Analytics** | Monthly aggregations and dashboard-ready results | SQL Server `analytics` schema |
| **Access** | Users, roles, companies, and scoped permissions | SQL Server `auth` schema |
| **Application** | Authorized exploration of analytics, news, and assets | `streamlit_app.py` |

Raw files are not modified after ingestion. Derived files and analytical views can be recreated from the raw sources and repository scripts. News rows retain a `lake_pointer_url` that links them to their source file; the application allows these links to resolve only inside the project `data/` directory.

### Core tables

| Table | Purpose |
|---|---|
| `core.source_regions` | Defines supported geographic regions and their type. |
| `core.source_agencies` | Defines macroeconomic providers such as FRED and Eurostat and links them to regions. |
| `core.indicators` | Defines indicator codes, names, and units. |
| `core.monthly_macro_indicators` | Stores one value per month, agency, and indicator. |
| `core.daily_news` | Stores one scored article per row with its date, source, region, sentiment, and raw-file pointer. |
| `core.assets` | Stores demonstration assets linked to a region and/or company. |

This normalized structure keeps reference information separate from observations and prevents duplicate monthly indicator records through a unique month-agency-indicator key.

### Analytics views

| View | Purpose |
|---|---|
| `analytics.Monthly_View` | Long-format monthly view joining regional sentiment to macroeconomic observations and calculating month-over-month changes. |
| `analytics.Monthly_Region_Wide_View` | One row per month and region, with separate columns for the four indicators, their month-over-month changes, and average sentiment. |

Month-over-month values are calculated across each complete agency/indicator series before sentiment is joined. This avoids using the wrong comparison month when news is missing.

### Access objects

The `auth` schema contains:

- `roles`, `users`, and `companies`;
- `user_access`, which assigns roles with optional company or region scope;
- `vw_user_login`, `vw_user_permissions`, and `vw_user_access_summary`, which support login, authorization, and administration.

The application uses these permissions to filter regional analytics, news, and company assets.

---

## 3. Data Processing

1. Source data is downloaded or placed in the raw layer.
2. `scripts/data_prep.ipynb` cleans dates, regions, identifiers, and numerical values.
3. FinBERT scores the news; reusable results are stored in `data/processed/`.
4. Clean records are loaded into `core.daily_news` and `core.monthly_macro_indicators`.
5. SQL views aggregate monthly sentiment, calculate month-over-month changes, and reshape the results.
6. Streamlit presents the permitted data to authenticated users.

The SQL scripts are executed in filename order:

```text
01_create_schemas.sql
02a_create_source_regions.sql
02b_build_core_tables.sql
03_create_analytics_view.sql
03b_create_monthly_region_wide_view.sql
04_create_auth_tables.sql
05_create_access_views.sql
06_create_assets_table.sql
```

---

## 4. Ownership and Licensing

The project is available at <https://github.com/zahra-ynp/Advanced-Data-Management>.

Project code, metadata, and eligible derived outputs are licensed under **CC BY-NC 4.0**. Raw data remains subject to its provider's terms and is not relicensed by this project.

| Data | Provider or owner |
|---|---|
| U.S. macroeconomic data | FRED and the originating U.S. agencies |
| European macroeconomic data | European Commission / Eurostat |
| FinSen financial news | Kaggle contributor `eaglewhl` |
| Financial News Multisource | `Brianferrell787` via Hugging Face |
| Project outputs | Zahra Younes Pour Langaroudi |

Users must review the current source licenses before redistributing raw data.

---

## 5. Documentation and Preservation

Supporting metadata is stored in:

- `metadata/dataset_metadata.jsonld` - dataset description;
- `metadata/data_dictionary.json` - field definitions;
- `metadata/data_lineage.jsonld` - sources and transformations.

The SQL scripts are the authoritative definitions of the implemented database objects. The DMP and metadata should be updated whenever sources, fields, transformations, or access rules change.

Code and documentation are versioned in Git. Raw data, metadata, scripts, and derived outputs will be retained for the academic program and for at least 12 months after submission. Reproducibility requires the repository version, dependencies, SQL Server, source data or API access, and locally supplied credentials.

---

## Appendix: API Parameters

### FRED

- Date range: `2007-01-01` to `2023-12-31`
- Series: `CPIAUCSL`, `UNRATE`, `FEDFUNDS`, `INDPRO`
- Requires a FRED API key

```text
https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&file_type=json&api_key=YOUR_API_KEY
```

### Eurostat

- Dataset: `prc_hicp_midx`
- Format: JSON/SDMX
- Main geographies: Italy and the Euro Area/European aggregate
- No API key required

```text
https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/prc_hicp_midx/1.0?format=JSON
```
