CREATE TABLE core.daily_news (
    article_id INT IDENTITY(1,1) PRIMARY KEY,
    news_date DATE NOT NULL,
    title VARCHAR(500),
    source_name VARCHAR(100),
    region_id INT NULL,
    sentiment_score FLOAT,
    sentiment_label VARCHAR(50),
    lake_pointer_url VARCHAR(500),

    CONSTRAINT FK_News_Region FOREIGN KEY (region_id)
        REFERENCES core.source_regions (region_id)
);
GO

CREATE TABLE core.source_agencies (
    agency_id      INT           IDENTITY(1,1) PRIMARY KEY,
    agency_code    VARCHAR(50)   NOT NULL,
    agency_name    VARCHAR(200)  NOT NULL,
    region_id      INT           NOT NULL,

    CONSTRAINT UQ_Source_Agency_Code UNIQUE (agency_code),

    CONSTRAINT FK_Source_Agency_Region FOREIGN KEY (region_id)
        REFERENCES core.source_regions (region_id)
);
GO

CREATE TABLE core.indicators (
    indicator_id    INT           IDENTITY(1,1) PRIMARY KEY,
    indicator_code  VARCHAR(100)  NOT NULL,
    indicator_name  VARCHAR(200)  NOT NULL,
    unit            VARCHAR(50)   NULL,

    CONSTRAINT UQ_Indicator_Code UNIQUE (indicator_code)
);
GO

INSERT INTO core.source_agencies (agency_code, agency_name, region_id)
SELECT 'FRED', 'Federal Reserve Economic Data', region_id
FROM core.source_regions
WHERE region_code = 'US';

INSERT INTO core.source_agencies (agency_code, agency_name, region_id)
SELECT 'EUROSTAT', 'Eurostat', region_id
FROM core.source_regions
WHERE region_code = 'EU';
GO

INSERT INTO core.indicators (indicator_code, indicator_name, unit) VALUES
    ('CPIAUCSL', 'Consumer Price Index for All Urban Consumers', 'Index'),
    ('UNRATE',   'Unemployment Rate',                            '%'),
    ('FEDFUNDS', 'Effective Federal Funds Rate',                 '%'),
    ('INDPRO',   'Industrial Production Index',                  'Index');
GO

CREATE TABLE core.monthly_macro_indicators (
    record_id        INT           IDENTITY(1,1) PRIMARY KEY,
    month_date       DATE          NOT NULL,
    agency_id        INT           NOT NULL,
    indicator_id     INT           NOT NULL,
    indicator_value  FLOAT         NULL,

    CONSTRAINT UQ_Macro_Indicator UNIQUE (month_date, agency_id, indicator_id),

    CONSTRAINT FK_Macro_Agency FOREIGN KEY (agency_id)
        REFERENCES core.source_agencies (agency_id),

    CONSTRAINT FK_Macro_Indicator FOREIGN KEY (indicator_id)
        REFERENCES core.indicators (indicator_id)
);
GO
