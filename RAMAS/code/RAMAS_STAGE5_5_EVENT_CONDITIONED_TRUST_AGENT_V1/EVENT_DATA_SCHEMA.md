# Point-in-time event data contract

The event ledger is a UTF-8 CSV with this exact header:

```text
event_id,published_at_utc,available_at_utc,source,source_url,category,severity,headline,body,content_sha256
```

Rules:

- timestamps must include a UTC offset;
- `available_at_utc` cannot precede `published_at_utc`;
- severity is an integer from 1 to 5;
- category is one of `macro`, `regulation`, `exchange`, `stablecoin`,
  `institutional_flow`, `onchain_stress`, `derivatives_stress`, `security`, or
  `other`;
- headline and raw body are required;
- `content_sha256` hashes the whitespace-normalized `headline + newline + body`;
- event identifiers must be unique;
- the agent sees an event only when `available_at_utc` is no later than the
  decision timestamp.

The separate JSON manifest must contain exactly:

```json
{
  "dataset_id": "unique-versioned-id",
  "event_csv_sha256": "sha256-of-entire-csv",
  "collection_method": "how raw documents were collected",
  "revision_policy": "how edits and later revisions were handled",
  "raw_text_preserved": true,
  "point_in_time_availability_verified": true,
  "coverage_start_utc": "2021-01-01T00:00:00+00:00",
  "coverage_end_utc": "2023-12-31T23:59:59+00:00"
}
```

Setting a Boolean field to `true` without documentary evidence is not a valid
scientific audit. The server script verifies structure and hashes; collection
evidence still requires human/source audit.

