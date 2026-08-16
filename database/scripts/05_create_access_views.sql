CREATE OR ALTER VIEW auth.vw_user_permissions AS
SELECT
    u.user_id,
    u.username,
    u.email,
    u.first_name,
    u.last_name,
    u.is_active,

    r.role_id,
    r.role_name,
    r.description AS role_description,

    c.company_id,
    c.company_name,

    sr.region_id,
    sr.region_name,
    sr.region_code,
    sr.region_type,

    ua.access_id,
    ua.created_at AS access_created_at

FROM auth.user_access ua
    INNER JOIN auth.users u
        ON ua.user_id = u.user_id
    INNER JOIN auth.roles r
        ON ua.role_id = r.role_id
    LEFT JOIN auth.companies c
        ON ua.company_id = c.company_id
    LEFT JOIN core.source_regions sr
        ON ua.region_id = sr.region_id;
GO


CREATE OR ALTER VIEW auth.vw_user_access_summary AS
WITH role_summary AS (
    SELECT
        distinct_roles.user_id,
        STRING_AGG(distinct_roles.role_name, ', ')
            WITHIN GROUP (ORDER BY distinct_roles.role_name) AS assigned_roles
    FROM (
        SELECT DISTINCT ua.user_id, r.role_name
        FROM auth.user_access ua
            INNER JOIN auth.roles r
                ON ua.role_id = r.role_id
    ) distinct_roles
    GROUP BY distinct_roles.user_id
),
company_summary AS (
    SELECT
        distinct_companies.user_id,
        STRING_AGG(distinct_companies.company_name, ', ')
            WITHIN GROUP (ORDER BY distinct_companies.company_name)
            AS accessible_companies
    FROM (
        SELECT DISTINCT ua.user_id, c.company_name
        FROM auth.user_access ua
            INNER JOIN auth.companies c
                ON ua.company_id = c.company_id
    ) distinct_companies
    GROUP BY distinct_companies.user_id
),
region_summary AS (
    SELECT
        distinct_regions.user_id,
        STRING_AGG(distinct_regions.region_name, ', ')
            WITHIN GROUP (ORDER BY distinct_regions.region_name)
            AS accessible_regions
    FROM (
        SELECT DISTINCT ua.user_id, sr.region_name
        FROM auth.user_access ua
            INNER JOIN core.source_regions sr
                ON ua.region_id = sr.region_id
    ) distinct_regions
    GROUP BY distinct_regions.user_id
),
grant_summary AS (
    SELECT
        user_id,
        COUNT(*) AS total_grants
    FROM auth.user_access
    GROUP BY user_id
)
SELECT
    u.user_id,
    u.username,
    u.email,
    u.first_name,
    u.last_name,
    u.is_active,
    roles.assigned_roles,
    COALESCE(grants.total_grants, 0) AS total_grants,
    companies.accessible_companies,
    regions.accessible_regions

FROM auth.users u
    LEFT JOIN role_summary roles
        ON u.user_id = roles.user_id
    LEFT JOIN grant_summary grants
        ON u.user_id = grants.user_id
    LEFT JOIN company_summary companies
        ON u.user_id = companies.user_id
    LEFT JOIN region_summary regions
        ON u.user_id = regions.user_id;
GO


CREATE OR ALTER VIEW auth.vw_user_login AS
SELECT
    user_id,
    username,
    password_hash,
    salt,
    email,
    is_active,
    updated_at
FROM auth.users;
GO

