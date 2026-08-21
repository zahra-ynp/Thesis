/*
Run once after 06_create_assets_table.sql.
This preserves FinSen for comparison while allowing one active news source.
*/

IF OBJECT_ID('core.news_sources', 'U') IS NULL
BEGIN
    CREATE TABLE core.news_sources (
        source_id     INT IDENTITY(1,1) PRIMARY KEY,
        source_code   VARCHAR(50)  NOT NULL,
        source_name   VARCHAR(200) NOT NULL,
        provider_url  VARCHAR(1000) NULL,
        is_active     BIT NOT NULL DEFAULT 0,
        CONSTRAINT UQ_News_Source_Code UNIQUE (source_code)
    );
END;
GO

IF NOT EXISTS (SELECT 1 FROM core.news_sources WHERE source_code = 'FINSEN')
    INSERT INTO core.news_sources (source_code, source_name, provider_url, is_active)
    VALUES ('FINSEN', 'FinSen Financial Sentiment Dataset',
            'https://www.kaggle.com/datasets/eaglewhl/finsen-financial-sentiment-dataset', 1);

IF NOT EXISTS (SELECT 1 FROM core.news_sources WHERE source_code = 'NYT')
    INSERT INTO core.news_sources (source_code, source_name, provider_url, is_active)
    VALUES ('NYT', 'The New York Times Archive API',
            'https://developer.nytimes.com/docs/archive-product/1/overview', 0);

IF NOT EXISTS (SELECT 1 FROM core.news_sources WHERE is_active = 1)
    UPDATE core.news_sources SET is_active = 1 WHERE source_code = 'FINSEN';
GO

IF COL_LENGTH('core.daily_news', 'source_id') IS NULL
    ALTER TABLE core.daily_news ADD source_id INT NULL;
IF COL_LENGTH('core.daily_news', 'external_article_id') IS NULL
    ALTER TABLE core.daily_news ADD external_article_id VARCHAR(300) NULL;
IF COL_LENGTH('core.daily_news', 'published_at') IS NULL
    ALTER TABLE core.daily_news ADD published_at DATETIME2 NULL;
IF COL_LENGTH('core.daily_news', 'article_url') IS NULL
    ALTER TABLE core.daily_news ADD article_url VARCHAR(1000) NULL;
GO

UPDATE n
SET source_id = s.source_id
FROM core.daily_news n
JOIN core.news_sources s ON s.source_code = 'FINSEN'
WHERE n.source_id IS NULL
  AND LOWER(ISNULL(n.source_name, '')) IN ('kaggle', 'finsen');
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_Daily_News_Source')
    ALTER TABLE core.daily_news ADD CONSTRAINT FK_Daily_News_Source
        FOREIGN KEY (source_id) REFERENCES core.news_sources(source_id);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UX_Daily_News_Source_External' AND object_id = OBJECT_ID('core.daily_news'))
    CREATE UNIQUE INDEX UX_Daily_News_Source_External
        ON core.daily_news(source_id, external_article_id)
        WHERE source_id IS NOT NULL AND external_article_id IS NOT NULL;
GO

CREATE OR ALTER VIEW analytics.Active_Daily_News AS
SELECT n.*
FROM core.daily_news n
JOIN core.news_sources s ON n.source_id = s.source_id
WHERE s.is_active = 1;
GO

CREATE OR ALTER VIEW analytics.Monthly_View AS
WITH monthly_sentiment AS (
    SELECT DATEFROMPARTS(YEAR(n.news_date), MONTH(n.news_date), 1) AS analysis_month,
           n.region_id,
           AVG(n.sentiment_score) AS avg_monthly_sentiment
    FROM analytics.Active_Daily_News n
    GROUP BY DATEFROMPARTS(YEAR(n.news_date), MONTH(n.news_date), 1), n.region_id
),
macro_with_mom AS (
    SELECT m.*,
           m.indicator_value - LAG(m.indicator_value, 1) OVER (
               PARTITION BY m.agency_id, m.indicator_id ORDER BY m.month_date
           ) AS indicator_mom
    FROM core.monthly_macro_indicators m
)
SELECT ms.analysis_month AS month_date,
       ind.record_id AS macro_record_id,
       agency.agency_id, agency.agency_code, agency.agency_name,
       agency.region_id, region.region_code, region.region_name, region.region_type,
       indicator.indicator_id, indicator.indicator_code, indicator.indicator_name, indicator.unit,
       ms.avg_monthly_sentiment, ind.indicator_value, ind.indicator_mom
FROM monthly_sentiment ms
JOIN macro_with_mom ind ON ms.analysis_month = ind.month_date
JOIN core.source_agencies agency ON ind.agency_id = agency.agency_id AND ms.region_id = agency.region_id
JOIN core.source_regions region ON agency.region_id = region.region_id
JOIN core.indicators indicator ON ind.indicator_id = indicator.indicator_id;
GO

CREATE OR ALTER VIEW analytics.Monthly_Region_Wide_View AS
WITH monthly_sentiment AS (
    SELECT DATEFROMPARTS(YEAR(news_date), MONTH(news_date), 1) AS month_date,
           region_id,
           AVG(sentiment_score) AS avg_monthly_sentiment
    FROM analytics.Active_Daily_News
    WHERE region_id IS NOT NULL
    GROUP BY DATEFROMPARTS(YEAR(news_date), MONTH(news_date), 1), region_id
),
macro_with_mom AS (
    SELECT m.*,
           m.indicator_value - LAG(m.indicator_value, 1) OVER (
               PARTITION BY m.agency_id, m.indicator_id ORDER BY m.month_date
           ) AS indicator_mom
    FROM core.monthly_macro_indicators m
),
regional_monthly_long AS (
    SELECT m.month_date, region.region_id, region.region_code, region.region_name, region.region_type,
           indicator.indicator_code, m.indicator_value, m.indicator_mom,
           sentiment.avg_monthly_sentiment
    FROM macro_with_mom m
    JOIN core.source_agencies agency ON m.agency_id = agency.agency_id
    JOIN core.source_regions region ON agency.region_id = region.region_id
    JOIN core.indicators indicator ON m.indicator_id = indicator.indicator_id
    LEFT JOIN monthly_sentiment sentiment
        ON m.month_date = sentiment.month_date AND region.region_id = sentiment.region_id
)
SELECT month_date, region_id, region_code, region_name, region_type,
       MAX(avg_monthly_sentiment) AS avg_monthly_sentiment,
       MAX(CASE WHEN indicator_code = 'CPIAUCSL' THEN indicator_value END) AS cpi_value,
       MAX(CASE WHEN indicator_code = 'CPIAUCSL' THEN indicator_mom END) AS cpi_mom,
       MAX(CASE WHEN indicator_code = 'UNRATE' THEN indicator_value END) AS unemployment_rate,
       MAX(CASE WHEN indicator_code = 'UNRATE' THEN indicator_mom END) AS unemployment_rate_mom,
       MAX(CASE WHEN indicator_code = 'FEDFUNDS' THEN indicator_value END) AS fed_funds_rate,
       MAX(CASE WHEN indicator_code = 'FEDFUNDS' THEN indicator_mom END) AS fed_funds_rate_mom,
       MAX(CASE WHEN indicator_code = 'INDPRO' THEN indicator_value END) AS industrial_production_value,
       MAX(CASE WHEN indicator_code = 'INDPRO' THEN indicator_mom END) AS industrial_production_mom
FROM regional_monthly_long
GROUP BY month_date, region_id, region_code, region_name, region_type;
GO
