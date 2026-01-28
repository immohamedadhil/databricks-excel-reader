# Databricks notebook source
# MAGIC %md
# MAGIC # Binary Excel to DataFrame - Performance Benchmark (Serverless Compatible)
# MAGIC ## Pandas vs PySpark (using applyInPandas) Approach
# MAGIC
# MAGIC This version works on serverless compute - no JAR libraries needed.

# COMMAND ----------

import time
import io
import pandas as pd
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, BinaryType
from pyspark.sql.functions import monotonically_increasing_id, col, lit

print("Imports loaded")

# COMMAND ----------

# Configuration
TEST_SCHEMA = {
    'Column1': 'str',
    'Column2': 'str',
    'Column3': 'str',
    'Column4': 'str',
    'Column5': 'str'
}

OUTPUT_SCHEMA = StructType([StructField(c, StringType(), True) for c in TEST_SCHEMA.keys()])

# COMMAND ----------

def generate_sample_excel_binary(num_rows: int = 10000) -> bytes:
    """Generate a sample Excel file as binary content (in-memory)"""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"

    headers = list(TEST_SCHEMA.keys())
    ws.append(headers)

    for i in range(num_rows):
        row = [f"Value_{col}_{i}" for col in headers]
        ws.append(row)

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def create_test_dataframe(spark, num_files: int, rows_per_file: int):
    """Create test DataFrame with binary Excel content"""
    print(f"Generating {num_files} files x {rows_per_file} rows...")

    binary_data = []
    for i in range(num_files):
        excel_bytes = generate_sample_excel_binary(rows_per_file)
        binary_data.append((str(i), excel_bytes))
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_files}...")

    schema = StructType([
        StructField("file_id", StringType(), True),
        StructField("content_inline_content", BinaryType(), True)
    ])

    df = spark.createDataFrame(binary_data, schema=schema)
    total_mb = sum(len(d) for _, d in binary_data) / (1024 * 1024)
    print(f"Total size: {total_mb:.2f} MB")

    return df, total_mb

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 1: Pandas (toPandas - all data to driver)

# COMMAND ----------

def process_pandas(spark, df: DataFrame, schema: dict) -> DataFrame:
    """Pandas approach - collects ALL data to driver memory"""

    pdf = df.toPandas()

    processed = []
    for _, row in pdf.iterrows():
        xlsx_bytes = row["content_inline_content"]
        excel_df = pd.read_excel(
            io.BytesIO(xlsx_bytes),
            sheet_name="Data",
            dtype=str,
            keep_default_na=False,
            na_values=[],
            engine='openpyxl'
        ).astype(schema)
        processed.extend(excel_df.to_dict(orient="records"))

    result = spark.createDataFrame(processed)
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 2: PySpark (applyInPandas - distributed)

# COMMAND ----------

def process_pyspark(spark, df: DataFrame, schema: dict) -> DataFrame:
    """PySpark approach - distributed using applyInPandas"""

    output_schema = StructType([StructField(c, StringType(), True) for c in schema.keys()])

    def parse_excel_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        """Process each partition's Excel files"""
        all_records = []
        for _, row in pdf.iterrows():
            xlsx_bytes = row["content_inline_content"]
            excel_df = pd.read_excel(
                io.BytesIO(xlsx_bytes),
                sheet_name="Data",
                dtype=str,
                keep_default_na=False,
                na_values=[],
                engine='openpyxl'
            ).astype(schema)
            all_records.append(excel_df)

        if all_records:
            return pd.concat(all_records, ignore_index=True)
        return pd.DataFrame(columns=list(schema.keys()))

    # Add partition key and group by it
    df_partitioned = df.withColumn("partition_id", (monotonically_increasing_id() % 10).cast("int"))

    result = df_partitioned.groupBy("partition_id").applyInPandas(
        parse_excel_partition,
        schema=output_schema
    )

    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Benchmark

# COMMAND ----------

def run_benchmark(spark, num_files: int, rows_per_file: int):
    """Run both approaches and compare"""

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {num_files} files x {rows_per_file} rows")
    print(f"{'='*60}")

    test_df, size_mb = create_test_dataframe(spark, num_files, rows_per_file)
    test_df.cache()
    test_df.count()

    results = {"files": num_files, "rows_per_file": rows_per_file, "size_mb": round(size_mb, 2)}

    # Pandas
    print("\n--- Pandas Approach ---")
    try:
        start = time.time()
        r1 = process_pandas(spark, test_df, TEST_SCHEMA)
        c1 = r1.count()
        t1 = time.time() - start
        results["pandas_time"] = round(t1, 2)
        results["pandas_rows"] = c1
        results["pandas_status"] = "SUCCESS"
        print(f"Time: {t1:.2f}s | Rows: {c1}")
    except Exception as e:
        results["pandas_status"] = f"FAILED: {str(e)[:50]}"
        print(f"FAILED: {e}")

    # PySpark
    print("\n--- PySpark Approach ---")
    try:
        start = time.time()
        r2 = process_pyspark(spark, test_df, TEST_SCHEMA)
        c2 = r2.count()
        t2 = time.time() - start
        results["pyspark_time"] = round(t2, 2)
        results["pyspark_rows"] = c2
        results["pyspark_status"] = "SUCCESS"
        print(f"Time: {t2:.2f}s | Rows: {c2}")
    except Exception as e:
        results["pyspark_status"] = f"FAILED: {str(e)[:50]}"
        print(f"FAILED: {e}")

    test_df.unpersist()

    # Summary
    if results.get("pandas_time") and results.get("pyspark_time"):
        speedup = results["pandas_time"] / results["pyspark_time"]
        winner = "PySpark" if speedup > 1 else "Pandas"
        print(f"\nSpeedup: {speedup:.2f}x ({winner} faster)")

    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Tests

# COMMAND ----------

all_results = []

# Small
r1 = run_benchmark(spark, num_files=10, rows_per_file=1000)
all_results.append(r1)

# COMMAND ----------

# Medium
r2 = run_benchmark(spark, num_files=50, rows_per_file=5000)
all_results.append(r2)

# COMMAND ----------

# Large - may OOM on Pandas
r3 = run_benchmark(spark, num_files=100, rows_per_file=50000)
all_results.append(r3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Summary

# COMMAND ----------

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

for r in all_results:
    print(f"\n{r['files']} files x {r['rows_per_file']} rows ({r['size_mb']} MB)")
    print(f"  Pandas:  {r.get('pandas_time', 'N/A')}s - {r.get('pandas_status', 'N/A')}")
    print(f"  PySpark: {r.get('pyspark_time', 'N/A')}s - {r.get('pyspark_status', 'N/A')}")

print("\n" + "="*60)
