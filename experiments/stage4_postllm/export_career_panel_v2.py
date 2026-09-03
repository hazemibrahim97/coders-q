import os, sys, time
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery

out = sys.argv[1]
t0 = time.time()
client = bigquery.Client(project="sampleproject-392511")
table = client.get_table("sampleproject-392511.codersq.career_panel_v2")
print(f"table: {table.num_rows:,} rows, {table.num_bytes/1e9:.2f} GB logical", flush=True)

writer, n = None, 0
for batch in client.list_rows(table).to_arrow_iterable(bqstorage_client=None or __import__(
        "google.cloud.bigquery_storage", fromlist=["BigQueryReadClient"]).BigQueryReadClient()):
    if writer is None:
        writer = pq.ParquetWriter(out, batch.schema, compression="zstd")
    writer.write_batch(batch)
    n += batch.num_rows
    if n % 10_000_000 < batch.num_rows:
        print(f"  {n:,} rows  {time.time()-t0:.0f}s  rss_ok", flush=True)
writer.close()
print(f"downloaded {n:,} rows in {time.time()-t0:.0f}s", flush=True)
assert n == table.num_rows, f"row count mismatch: {n} vs {table.num_rows}"
print(f"wrote {out}: {os.path.getsize(out)/1e6:.0f} MB", flush=True)
