Find My Query
-

Query to pull table and column metadata:
```SQL
	SELECT 
	    n.nspname AS schema_name,
	    t.relname AS table_name,
	    pg_catalog.obj_description(t.oid, 'pg_class') AS table_comment,
	    c.column_name,
	    c.data_type,
	    pg_catalog.col_description(t.oid, c.ordinal_position) AS column_comment
	FROM pg_catalog.pg_class t
	JOIN pg_catalog.pg_namespace n ON n.oid = t.relnamespace
	JOIN information_schema.columns c ON (
	    t.relname = c.table_name AND 
	    n.nspname = c.table_schema
	)
	WHERE
		n.nspname IN (
			'public',
			'configs',
			'email_ops',
			'email_ops_stage',
			'experiment',
			'measures',
			'pipeline_stats',
			'predictive_analytics',
			'predictive_analytics_stage',
			'revenue',
			'pg_catalog',
			'information_schema'
		)
		AND t.relkind = 'r'
	ORDER BY schema_name, table_name, c.ordinal_position
	;
```
