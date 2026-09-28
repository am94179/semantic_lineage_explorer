CREATE OR REPLACE TABLE analytics.customer_revenue_yearly AS
WITH channel_sales AS (
    SELECT ss.ss_customer_sk AS customer_sk, d.d_year AS calendar_year, 'store' AS sales_channel,
           ss.ss_net_paid AS net_paid
    FROM raw.store_sales AS ss
    INNER JOIN raw.date_dim AS d ON ss.ss_sold_date_sk = d.d_date_sk
    WHERE ss.ss_customer_sk IS NOT NULL
    UNION ALL
    SELECT cs.cs_bill_customer_sk AS customer_sk, d.d_year AS calendar_year, 'catalog' AS sales_channel,
           cs.cs_net_paid AS net_paid
    FROM raw.catalog_sales AS cs
    INNER JOIN raw.date_dim AS d ON cs.cs_sold_date_sk = d.d_date_sk
    WHERE cs.cs_bill_customer_sk IS NOT NULL
    UNION ALL
    SELECT ws.ws_bill_customer_sk AS customer_sk, d.d_year AS calendar_year, 'web' AS sales_channel,
           ws.ws_net_paid AS net_paid
    FROM raw.web_sales AS ws
    INNER JOIN raw.date_dim AS d ON ws.ws_sold_date_sk = d.d_date_sk
    WHERE ws.ws_bill_customer_sk IS NOT NULL
)
SELECT
    customer_sk,
    calendar_year,
    COALESCE(SUM(net_paid) FILTER (WHERE sales_channel = 'store'), 0) AS store_net_paid,
    COALESCE(SUM(net_paid) FILTER (WHERE sales_channel = 'catalog'), 0) AS catalog_net_paid,
    COALESCE(SUM(net_paid) FILTER (WHERE sales_channel = 'web'), 0) AS web_net_paid,
    SUM(net_paid) AS total_net_paid
FROM channel_sales
GROUP BY customer_sk, calendar_year;
