CREATE OR ALTER VIEW analytics.Monthly_Region_Wide_View AS
WITH monthly_sentiment AS (
    SELECT
        DATEFROMPARTS(YEAR(news_date), MONTH(news_date), 1) AS month_date,
        region_id,
        AVG(sentiment_score) AS avg_monthly_sentiment
    FROM core.daily_news
    WHERE region_id IS NOT NULL
    GROUP BY
        DATEFROMPARTS(YEAR(news_date), MONTH(news_date), 1),
        region_id
),
macro_with_mom AS (
    SELECT
        m.month_date,
        m.agency_id,
        m.indicator_id,
        m.indicator_value,
        m.indicator_value - LAG(m.indicator_value, 1)
            OVER (
                PARTITION BY m.agency_id, m.indicator_id
                ORDER BY m.month_date
            ) AS indicator_mom
    FROM core.monthly_macro_indicators m
),
regional_monthly_long AS (
    SELECT
        m.month_date,
        region.region_id,
        region.region_code,
        region.region_name,
        region.region_type,
        indicator.indicator_code,
        m.indicator_value,
        m.indicator_mom,
        sentiment.avg_monthly_sentiment
    FROM macro_with_mom m
        INNER JOIN core.source_agencies agency
            ON m.agency_id = agency.agency_id
        INNER JOIN core.source_regions region
            ON agency.region_id = region.region_id
        INNER JOIN core.indicators indicator
            ON m.indicator_id = indicator.indicator_id
        LEFT JOIN monthly_sentiment sentiment
            ON m.month_date = sentiment.month_date
            AND region.region_id = sentiment.region_id
)
SELECT
    month_date,
    region_id,
    region_code,
    region_name,
    region_type,

    MAX(avg_monthly_sentiment) AS avg_monthly_sentiment,

    -- Consumer Price Index
    MAX(CASE
        WHEN indicator_code = 'CPIAUCSL'
        THEN indicator_value
    END) AS cpi_value,

    MAX(CASE
        WHEN indicator_code = 'CPIAUCSL'
        THEN indicator_mom
    END) AS cpi_mom,

    -- Unemployment Rate
    MAX(CASE
        WHEN indicator_code = 'UNRATE'
        THEN indicator_value
    END) AS unemployment_rate,

    MAX(CASE
        WHEN indicator_code = 'UNRATE'
        THEN indicator_mom
    END) AS unemployment_rate_mom,

    -- Effective Federal Funds Rate
    MAX(CASE
        WHEN indicator_code = 'FEDFUNDS'
        THEN indicator_value
    END) AS fed_funds_rate,

    MAX(CASE
        WHEN indicator_code = 'FEDFUNDS'
        THEN indicator_mom
    END) AS fed_funds_rate_mom,

    -- Industrial Production Index
    MAX(CASE
        WHEN indicator_code = 'INDPRO'
        THEN indicator_value
    END) AS industrial_production_value,

    MAX(CASE
        WHEN indicator_code = 'INDPRO'
        THEN indicator_mom
    END) AS industrial_production_mom

FROM regional_monthly_long
GROUP BY
    month_date,
    region_id,
    region_code,
    region_name,
    region_type;
GO
