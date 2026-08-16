CREATE TABLE core.source_regions (
    region_id      INT           IDENTITY(1,1) PRIMARY KEY,
    region_code    VARCHAR(10)   NOT NULL,
    region_name    VARCHAR(100)  NOT NULL,
    region_type    VARCHAR(20)   NOT NULL,
    created_at     DATE          NOT NULL DEFAULT GETDATE(),

    CONSTRAINT UQ_Region_Code UNIQUE (region_code),

    CONSTRAINT CK_Region_Type CHECK (region_type IN ('country', 'continent'))
);
GO

INSERT INTO core.source_regions (region_code, region_name, region_type) VALUES
    ('US',    'United States',    'country'),
    ('EU',    'European Union',   'continent'),
    ('IT',    'Italy',            'country'),
    ('GLB',   'Global',           'continent'),
    ('APAC',  'Asia-Pacific',     'continent'),
    ('ME',    'Middle East',      'continent'),
    ('LATAM', 'Latin America',    'continent');
GO

