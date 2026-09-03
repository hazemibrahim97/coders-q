import time, pyarrow.parquet as pq
from google.cloud import bigquery
t0 = time.time()
client = bigquery.Client(project="sampleproject-392511")
table = client.get_table("sampleproject-392511.codersq.career_panel")
print(f"table: {table.num_rows:,} rows, {table.num_bytes/1e9:.2f} GB logical", flush=True)
arrow = client.list_rows(table).to_arrow(create_bqstorage_client=True)
print(f"downloaded {arrow.num_rows:,} rows in {time.time()-t0:.0f}s", flush=True)
assert arrow.num_rows == table.num_rows, "row count mismatch vs table metadata"
out = "/private/tmp/claude-502/-Users-$USER-Desktop-Work-Coders-Q/9cc18677-8bb7-4018-810b-6d6bc14693dc/scratchpad/career_panel.parquet"
pq.write_table(arrow, out, compression="zstd")
import os; print(f"wrote {out}: {os.path.getsize(out)/1e6:.0f} MB", flush=True)
