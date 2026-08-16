IF OBJECT_ID('core.assets', 'U') IS NULL
BEGIN
    CREATE TABLE core.assets (
        asset_id       INT IDENTITY(1,1) PRIMARY KEY,
        asset_name     VARCHAR(200) NOT NULL,
        asset_class    VARCHAR(100) NOT NULL,
        region_id      INT NULL,
        company_id     INT NULL,
        market_value   DECIMAL(18,2) NULL,
        currency_code  CHAR(3) NULL,
        as_of_date     DATE NOT NULL,
        notes          VARCHAR(500) NULL,
        created_at     DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),

        CONSTRAINT FK_Assets_Region FOREIGN KEY (region_id)
            REFERENCES core.source_regions (region_id),
        CONSTRAINT FK_Assets_Company FOREIGN KEY (company_id)
            REFERENCES auth.companies (company_id)
    );

    CREATE INDEX IX_Assets_AsOfDate_Region
        ON core.assets (as_of_date, region_id);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM core.assets
    WHERE asset_name = 'US Treasury Bond Portfolio'
      AND as_of_date = '2026-06-30'
)
BEGIN
    INSERT INTO core.assets (
        asset_name,
        asset_class,
        region_id,
        company_id,
        market_value,
        currency_code,
        as_of_date,
        notes
    )
    SELECT
        'US Treasury Bond Portfolio',
        'Government Bonds',
        region.region_id,
        company.company_id,
        2450000.00,
        'USD',
        '2026-06-30',
        'Example fixed-income asset for the admin dashboard'
    FROM core.source_regions region
        CROSS JOIN auth.companies company
    WHERE region.region_code = 'US'
      AND company.company_name = 'Generali Italia';
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM core.assets
    WHERE asset_name = 'European Sustainable Equity Fund'
      AND as_of_date = '2026-06-30'
)
BEGIN
    INSERT INTO core.assets (
        asset_name,
        asset_class,
        region_id,
        company_id,
        market_value,
        currency_code,
        as_of_date,
        notes
    )
    SELECT
        'European Sustainable Equity Fund',
        'Equities',
        region.region_id,
        company.company_id,
        1875000.00,
        'EUR',
        '2026-06-30',
        'Example ESG-focused equity asset for the admin dashboard'
    FROM core.source_regions region
        CROSS JOIN auth.companies company
    WHERE region.region_code = 'EU'
      AND company.company_name = 'Assicurazioni Generali';
END;
GO
