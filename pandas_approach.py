# Databricks notebook source
# MAGIC %pip install fastexcel polars pyarrow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import uuid
import time
import tempfile
from typing import List

import polars as pl
import pyarrow.parquet as pq
from pyspark.sql import DataFrame as SparkDataFrame

def read_binary_excel_polars(spark, table_name: str, sheet_name: str = "result") -> SparkDataFrame:
    """
    Read binary Excel data from table and convert to Spark DataFrame.
    
    Uses Polars + fastexcel (Rust calamine) for:
    - 6-10x faster than openpyxl
    - Proper null handling (empty cells → null, not empty string)
    
    Args:
        spark: SparkSession
        table_name: Full table name (catalog.schema.table)
        sheet_name: Excel sheet name
        
    Returns:
        Spark DataFrame with parsed Excel data
    """
    start_time = time.time()
    
    # 1. Write binary data to parquet (avoids gRPC size limit)
    binary_parquet_path = f"/Volumes/adhil/test/weather_data/binary_temp_{uuid.uuid4().hex[:8]}.parquet"
    
    print(f"Writing binary data to temp parquet...", end="", flush=True)
    spark.sql(f"select content_inline_content from table_name").write.mode("overwrite").parquet(binary_parquet_path)
    print(" done")
    
    # 2. Read binary parquet with PyArrow (bypasses gRPC entirely)
    binary_table = pq.read_table(binary_parquet_path)
    binary_list = binary_table["content_inline_content"].to_pylist()
    row_count = len(binary_list)
    
    print(f"Found {row_count} file(s) in table")
    
    if row_count == 0:
        # Cleanup
        dbutils.fs.rm(binary_parquet_path, recurse=True)
        raise ValueError(f"No files found in {table_name}")
    
    # 3. Process each binary Excel file
    all_dfs: List[pl.DataFrame] = []
    
    for idx, xlsx_bytes in enumerate(binary_list):
        size_mb = len(xlsx_bytes) / (1024 * 1024)
        
        print(f"Processing file {idx + 1}/{row_count}: {size_mb:.1f} MB", end="", flush=True)
        file_start = time.time()
        
        # Write to temp file (fastexcel requires file path)
        temp_path = os.path.join(tempfile.gettempdir(), f"xl_{uuid.uuid4().hex[:8]}.xlsx")
        
        try:
            with open(temp_path, "wb") as f:
                f.write(xlsx_bytes)
            
            # Parse with Polars + calamine (Rust-based, fast)
            polars_df = pl.read_excel(
                source=temp_path,
                sheet_name=sheet_name,
                engine="calamine",
                infer_schema_length=10000,
            )
            
            all_dfs.append(polars_df)
            elapsed = time.time() - file_start
            print(f" → {polars_df.shape[0]:,} rows, {polars_df.shape[1]} cols ({elapsed:.1f}s)")
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # 4. Combine all DataFrames
    if len(all_dfs) == 1:
        combined = all_dfs[0]
    else:
        combined = pl.concat(all_dfs, how="diagonal_relaxed")
    
    # 5. Convert to Spark DataFrame via Parquet (bypasses gRPC size limit)
    output_parquet_path = f"/Volumes/adhil/test/weather_data/output_temp_{uuid.uuid4().hex[:8]}.parquet"
    
    print("Writing output parquet...", end="", flush=True)
    combined.write_parquet(output_parquet_path)
    print(" done")
    
    print("Reading into Spark...", end="", flush=True)
    spark_df = spark.read.parquet(output_parquet_path)
    print(" done")
    
    row_count = spark_df.count()
    total_time = time.time() - start_time
    print(f"\n✓ Result: {row_count:,} rows × {len(spark_df.columns)} cols ({total_time:.1f}s)")
    
    # Cleanup temp files
    dbutils.fs.rm(binary_parquet_path, recurse=True)
    # dbutils.fs.rm(output_parquet_path, recurse=True)
    
    return spark_df


# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
TABLE_NAME = "adhil.test.excel_binary_data"

# Read and convert to Spark DataFrame
result_df = read_binary_excel_polars(spark, TABLE_NAME, sheet_name="result")

# COMMAND ----------

# Preview schema
result_df.printSchema()

# COMMAND ----------

# Preview data (limit to avoid gRPC issues)
display(result_df.limit(20))
dbutils.fs.rm(output_parquet_path, recurse=True)

