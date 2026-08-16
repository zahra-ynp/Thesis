CREATE TABLE auth.roles (
    role_id       INT           IDENTITY(1,1) PRIMARY KEY,
    role_name     VARCHAR(50)   NOT NULL,
    description   VARCHAR(255)  NULL,
    created_at    DATE          NOT NULL DEFAULT GETDATE(),

    CONSTRAINT UQ_Role_Name UNIQUE (role_name)
);
GO

INSERT INTO auth.roles (role_name, description) VALUES
    ('admin',      'Administrator — unrestricted access to all data, users, and settings'),
    ('general',    'Public user — can view inflation predictions and news sentiment only'),
    ('company',    'Company user — can view asset classes and inflation impact for their company');
GO


CREATE TABLE auth.users (
    user_id        INT           IDENTITY(1,1) PRIMARY KEY,
    username       VARCHAR(100)  NOT NULL,
    password_hash  VARCHAR(255)  NOT NULL,
    salt           VARCHAR(255)  NOT NULL,
    email          VARCHAR(255)  NOT NULL,
    first_name     VARCHAR(100)  NULL,
    last_name      VARCHAR(100)  NULL,
    is_active      BIT           NOT NULL DEFAULT 1,
    created_at     DATE          NOT NULL DEFAULT GETDATE(),
    updated_at     DATE          NOT NULL DEFAULT GETDATE(),

    CONSTRAINT UQ_User_Username UNIQUE (username),
    CONSTRAINT UQ_User_Email    UNIQUE (email)
);
GO


CREATE TABLE auth.companies (
    company_id    INT           IDENTITY(1,1) PRIMARY KEY,
    company_name  VARCHAR(200)  NOT NULL,
    created_at    DATE          NOT NULL DEFAULT GETDATE(),

    CONSTRAINT UQ_Company_Name UNIQUE (company_name)
);
GO


CREATE TABLE auth.user_access (
    access_id      INT        IDENTITY(1,1) PRIMARY KEY,
    user_id        INT        NOT NULL,
    role_id        INT        NOT NULL,
    company_id     INT        NULL,
    region_id      INT        NULL,
    created_at     DATE       NOT NULL DEFAULT GETDATE(),

    -- Foreign keys
    CONSTRAINT FK_UA_User FOREIGN KEY (user_id)
        REFERENCES auth.users (user_id)
        ON DELETE CASCADE,

    CONSTRAINT FK_UA_Role FOREIGN KEY (role_id)
        REFERENCES auth.roles (role_id),

    CONSTRAINT FK_UA_Company FOREIGN KEY (company_id)
        REFERENCES auth.companies (company_id),

    -- Region FK points to the shared core.source_regions table
    CONSTRAINT FK_UA_Region FOREIGN KEY (region_id)
        REFERENCES core.source_regions (region_id),

    -- Prevent duplicate access grants for the same scope
    CONSTRAINT UQ_User_Access_Grant UNIQUE (user_id, role_id, company_id, region_id)
);
GO


INSERT INTO auth.companies (company_name) VALUES
    ('Generali Italia'),
    ('Assicurazioni Generali');
GO

INSERT INTO auth.users (
    username,
    password_hash,
    salt,
    email,
    first_name,
    last_name
) VALUES
    (
        'Zahra',
        '815135a6281a5590dee125607c5e62fa8f882bd0ed2e138b9b0dd7a7a6c9142a',
        '03ac67',
        'zahra.admin@example.test',
        'Zahra',
        'Younes Pour'
    ),
    (
        'Yingjie',
        '815135a6281a5590dee125607c5e62fa8f882bd0ed2e138b9b0dd7a7a6c9142a',
        '03ac67',
        'yingjie.company@example.test',
        'Yingjie',
        'Li'
    ),
    (
        'ali',
        '815135a6281a5590dee125607c5e62fa8f882bd0ed2e138b9b0dd7a7a6c9142a',
        '03ac67',
        'ali.region@example.test',
        'Ali',
        'Rossi'
    );
GO

INSERT INTO auth.user_access (user_id, role_id, company_id, region_id)
SELECT
    u.user_id,
    r.role_id,
    NULL,
    NULL
FROM auth.users u
    CROSS JOIN auth.roles r
WHERE u.username = 'Zahra'
  AND r.role_name = 'admin';
GO

INSERT INTO auth.user_access (user_id, role_id, company_id, region_id)
SELECT
    u.user_id,
    r.role_id,
    c.company_id,
    NULL
FROM auth.users u
    CROSS JOIN auth.roles r
    CROSS JOIN auth.companies c
WHERE u.username = 'Yingjie'
  AND r.role_name = 'company'
  AND c.company_name = 'Generali Italia';
GO

INSERT INTO auth.user_access (user_id, role_id, region_id)
SELECT
    u.user_id,
    r.role_id,
    region.region_id
FROM auth.users u
    CROSS JOIN auth.roles r
    CROSS JOIN core.source_regions region
WHERE u.username = 'ali'
  AND r.role_name = 'admin'
  AND region.region_code = 'US';
GO


-- Fast lookup of active users by username (login flow)
CREATE NONCLUSTERED INDEX IX_Users_Username_Active
    ON auth.users (username, is_active);
GO

-- Fast lookup of all access grants for a specific user
CREATE NONCLUSTERED INDEX IX_UserAccess_UserId
    ON auth.user_access (user_id)
    INCLUDE (role_id, company_id, region_id);
GO
