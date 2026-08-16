CREATE OR ALTER VIEW analytics.Monthly_View AS
WITH monthly_sentiment AS (
    -- Step A: Aggregate Daily News into Monthly Scores by snapping to the 1st of the month
    SELECT 
        DATEFROMPARTS(YEAR(n.news_date), MONTH(n.news_date), 1) AS analysis_month,
        n.region_id,
        AVG(n.sentiment_score) AS avg_monthly_sentiment
    FROM core.daily_news n
    GROUP BY DATEFROMPARTS(YEAR(n.news_date), MONTH(n.news_date), 1), n.region_id
),
macro_with_mom AS (
    -- Calculate MoM across the complete indicator series before joining to news.
    SELECT
        m.record_id,
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
)
-- Step B: Join sentiment with indicators via shared region_id and compute MoM
SELECT 
    ms.analysis_month AS month_date,
    ind.record_id AS macro_record_id,

    -- Source agency identifiers and descriptive attributes
    agency.agency_id,
    agency.agency_code,
    agency.agency_name,

    -- Geographic identifiers and descriptive attributes
    agency.region_id,
    region.region_code,
    region.region_name,
    region.region_type,

    -- Indicator identifiers and descriptive attributes
    indicator.indicator_id,
    indicator.indicator_code,
    indicator.indicator_name,
    indicator.unit,

    -- Analytical measures
    ms.avg_monthly_sentiment,
    ind.indicator_value,
    ind.indicator_mom

FROM monthly_sentiment ms
    -- Join to macro indicators directly on region_id + month
    INNER JOIN macro_with_mom ind
        ON ms.analysis_month = ind.month_date
    INNER JOIN core.source_agencies agency
        ON ind.agency_id = agency.agency_id
        AND ms.region_id = agency.region_id
    INNER JOIN core.source_regions region
        ON agency.region_id = region.region_id
    INNER JOIN core.indicators indicator
        ON ind.indicator_id = indicator.indicator_id;
GO
