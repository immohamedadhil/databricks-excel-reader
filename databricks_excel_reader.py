# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Excel Reader
# MAGIC
# MAGIC **Read large Excel files (100MB - 1GB+) from Delta tables on small clusters.**
# MAGIC
# MAGIC Uses foreachPartition + Polars + Calamine to bypass driver OOM and gRPC limits.
# MAGIC
# MAGIC | File Size | Cluster RAM |
# MAGIC |-----------|-------------|
# MAGIC | 100MB | 8GB |
# MAGIC | 500MB | 16GB |
# MAGIC | 1GB | 32GB |

# COMMAND ----------

# MAGIC %pip install fastexcel polars --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# =============================================================================
# CONFIGURATION
# =============================================================================

VOLUME_BASE_PATH = "/Volumes/adhil/test/weather_data"  # Your Volume path
TABLE_NAME = "adhil.test.excel_binary_data"            # Source table
BINARY_COLUMN = "content_inline_content"               # Column with binary Excel
SHEET_NAME = "result"                                  # Excel sheet to read

# COMMAND ----------

import os
import uuid
import time
from typing import List

import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame

# =============================================================================
# CORE FUNCTION
# =============================================================================

def read_binary_excel(
    spark,
    table_name: str,
    binary_column: str,
    sheet_name: str,
    volume_base_path: str,
) -> SparkDataFrame:
    """
    Read binary Excel data from Delta table and return Spark DataFrame.
    
    Uses foreachPartition to write binary on executor (not driver),
    then Polars + Calamine (Rust) to parse Excel efficiently.
    
    Args:
        spark: SparkSession
        table_name: Full table name (catalog.schema.table)
        binary_column: Column containing binary Excel content
        sheet_name: Excel sheet name to read
        volume_base_path: Volume path for temp files
        
    Returns:
        Spark DataFrame with parsed Excel data
    """
    start_time = time.time()
    run_id = uuid.uuid4().hex[:8]
    run_path = f"{volume_base_path}/excel_run_{run_id}"
    
    print(f"Run ID: {run_id}")
    print(f"Temp path: {run_path}")
    print("=" * 60)
    
    try:
        # 1. Create run directory
        dbutils.fs.mkdirs(run_path)
        excel_dir = f"{run_path}/excel_files"
        dbutils.fs.mkdirs(excel_dir)
        
        # Check table size from Delta metadata
        size_df = spark.sql(f"DESCRIBE DETAIL {table_name}")
        total_bytes = size_df.select("sizeInBytes").collect()[0][0]
        total_mb = total_bytes / (1024 * 1024)
        
        print(f"Table size: {total_mb:.1f} MB")
        
        # 2. Write binary to Volume via foreachPartition (no data returned, no OOM)
        print("[1/4] Writing binary to Volume...", end="", flush=True)
        
        _excel_dir = excel_dir
        _binary_column = binary_column
        
        def write_partition(rows):
            import uuid
            for row in rows:
                binary = row[_binary_column]
                path = f"{_excel_dir}/excel_{uuid.uuid4().hex[:8]}.xlsx"
                with open(path, "wb") as f:
                    f.write(binary)
        
        spark.table(table_name).select(binary_column).foreachPartition(write_partition)
        
        # Get file paths from Volume
        file_infos = dbutils.fs.ls(excel_dir)
        file_paths = [f.path.replace("dbfs:", "") for f in file_infos]
        
        print(f" done ({len(file_paths)} file(s))")
        
        # 3. Read Excel files with Polars + Calamine
        print("[2/4] Parsing Excel with Polars + Calamine...")
        all_dfs: List[pl.DataFrame] = []
        
        for idx, file_path in enumerate(file_paths):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"       [{idx + 1}/{len(file_paths)}] {file_size_mb:.1f} MB", end="", flush=True)
            file_start = time.time()
            
            polars_df = pl.read_excel(
                source=file_path,
                sheet_name=sheet_name,
                engine="calamine",
                infer_schema_length=10000,
            )
            
            all_dfs.append(polars_df)
            elapsed = time.time() - file_start
            print(f" -> {polars_df.shape[0]:,} rows, {polars_df.shape[1]} cols ({elapsed:.1f}s)")
        
        # 4. Combine DataFrames
        print("[3/4] Combining DataFrames...", end="", flush=True)
        if len(all_dfs) == 1:
            combined = all_dfs[0]
        else:
            combined = pl.concat(all_dfs, how="diagonal_relaxed")
        print(f" done ({combined.shape[0]:,} rows)")
        
        # 5. Convert to Spark via Parquet
        print("[4/4] Converting to Spark DataFrame...", end="", flush=True)
        output_parquet = f"{run_path}/output.parquet"
        combined.write_parquet(output_parquet)
        spark_df = spark.read.parquet(output_parquet)
        
        # Force evaluation before cleanup (localCheckpoint breaks lineage to deleted parquet)
        spark_df = spark_df.localCheckpoint(eager=True)
        row_count = spark_df.count()
        print(" done")
        
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"Complete: {row_count:,} rows x {len(spark_df.columns)} cols ({total_time:.1f}s)")
        
        return spark_df
        
    finally:
        # Cleanup run directory regardless of success/failure
        try:
            dbutils.fs.rm(run_path, recurse=True)
            print(f"Cleaned up: {run_path}")
        except Exception as e:
            print(f"Cleanup failed: {e}")

# COMMAND ----------

# =============================================================================
# EXECUTE
# =============================================================================

result_df = read_binary_excel(
    spark=spark,
    table_name=TABLE_NAME,
    binary_column=BINARY_COLUMN,
    sheet_name=SHEET_NAME,
    volume_base_path=VOLUME_BASE_PATH,
)

# COMMAND ----------

# Preview schema
result_df.printSchema()

# COMMAND ----------

# Preview data
display(result_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta (Optional)

# COMMAND ----------

# OUTPUT_TABLE = "adhil.test.clean_excel_data"
# result_df.write.format("delta").mode("overwrite").saveAsTable(OUTPUT_TABLE)
# print(f"Written to {OUTPUT_TABLE}")
