# Databricks notebook source
# MAGIC %md
# MAGIC # Binary Excel to DataFrame - Performance Benchmark
# MAGIC ## Pandas vs PySpark Approach
# MAGIC
# MAGIC This notebook compares two approaches for converting binary Excel data to Spark DataFrames:
# MAGIC 1. **Pandas Approach**: Uses `toPandas()` and `pd.read_excel()`
# MAGIC 2. **PySpark Approach**: Uses `foreachPartition` to write temp files + `spark-excel` library

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------

import time
import io
import uuid
import pandas as pd
import traceback
from functools import reduce
from datetime import datetime
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, BinaryType
from pyspark.sql.functions import monotonically_increasing_id, col, lit
from pyspark import StorageLevel

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration - adjust based on your environment
CATALOG = "hive_metastore"  # Change to your catalog
SCHEMA = "default"  # Change to your schema
VOLUME_PATH_BASE = "/tmp/benchmark_excel_test"

# Test schema for Excel files
TEST_SCHEMA = {
    'Column1': 'str',
    'Column2': 'str',
    'Column3': 'str',
    'Column4': 'str',
    'Column5': 'str'
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions - Metrics Collection

# COMMAND ----------

class BenchmarkMetrics:
    def __init__(self):
        self.results = []

    def record(self, approach: str, metric: str, value):
        self.results.append({
            "approach": approach,
            "metric": metric,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })

    def get_results_df(self):
        return spark.createDataFrame(self.results)

    def print_summary(self):
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)
        for r in self.results:
            print(f"{r['approach']:20} | {r['metric']:25} | {r['value']}")
        print("="*60)

metrics = BenchmarkMetrics()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Sample Binary Excel Data
# MAGIC Creates sample Excel files as binary content (simulating Lakeflow Connect ingestion)

# COMMAND ----------

def generate_sample_excel_binary(num_rows: int = 10000) -> bytes:
    """Generate a sample Excel file as binary content"""
    import openpyxl
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"

    # Header
    headers = list(TEST_SCHEMA.keys())
    ws.append(headers)

    # Data rows
    for i in range(num_rows):
        row = [f"Value_{col}_{i}" for col in headers]
        ws.append(row)

    # Save to bytes
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

# COMMAND ----------

def create_test_dataframe(num_files: int, rows_per_file: int) -> DataFrame:
    """Create a test DataFrame with binary Excel content"""
    print(f"Generating {num_files} Excel files with {rows_per_file} rows each...")

    binary_data = []
    for i in range(num_files):
        excel_bytes = generate_sample_excel_binary(rows_per_file)
        binary_data.append((i, excel_bytes))
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_files} files...")

    schema = StructType([
        StructField("file_id", StringType(), True),
        StructField("content_inline_content", BinaryType(), True)
    ])

    # Create DataFrame from binary data
    df = spark.createDataFrame(
        [(str(i), data) for i, data in binary_data],
        schema=schema
    )

    total_size_mb = sum(len(data) for _, data in binary_data) / (1024 * 1024)
    print(f"Total data size: {total_size_mb:.2f} MB")
    metrics.record("Setup", "total_data_size_mb", round(total_size_mb, 2))
    metrics.record("Setup", "num_files", num_files)
    metrics.record("Setup", "rows_per_file", rows_per_file)

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 1: Pandas-based (Current Production Code)

# COMMAND ----------

def process_batch_pandas(spark, df: DataFrame, table_schema: dict) -> DataFrame:
    """
    Pandas approach - collects all data to driver
    WARNING: Memory-intensive for large datasets
    """
    # 1) Convert the Spark DataFrame to Pandas DataFrame
    pdf = df.toPandas()
    t_schema = table_schema

    # 2) Process the Pandas DataFrame
    processed_records = []
    for _, row in pdf.iterrows():
        xlsx_bytes = row["content_inline_content"]
        excel_stream = io.BytesIO(xlsx_bytes)

        pdf_excel = pd.read_excel(
            excel_stream,
            sheet_name="Data",
            dtype=str,
            keep_default_na=False,
            na_values=[],
            engine='openpyxl'
        ).astype(t_schema)

        processed_records.extend(pdf_excel.to_dict(orient="records"))

    # 3) Convert back to Spark DataFrame
    parsed_df = spark.createDataFrame(processed_records)
    parsed_df = parsed_df.na.replace(['#N/A', 'nan'], None)
    return parsed_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Approach 2: PySpark-based (Optimized Code)

# COMMAND ----------

def process_batch_pyspark(spark, df: DataFrame, table_schema: dict) -> DataFrame:
    """
    PySpark approach - distributed processing using foreachPartition
    """
    t_schema = StructType(
        [StructField(c, StringType(), True) for c in table_schema.keys()]
    )

    unique_id = str(uuid.uuid4().hex)
    volume_path = f"{VOLUME_PATH_BASE}/{unique_id}"

    def write_binary_to_file(partition):
        for row in partition:
            file_path = f"{volume_path}/{row.id}.xlsx"
            with open(file_path, "wb") as file:
                file.write(row['content_inline_content'])

    # Add ID and persist
    df = df.withColumn("id", monotonically_increasing_id())
    df = df.persist(StorageLevel.MEMORY_AND_DISK)

    try:
        # Create directory and write files
        dbutils.fs.mkdirs(volume_path)
        df.foreachPartition(write_binary_to_file)

        # Read Excel files using spark-excel
        excel_dfs = [
            spark.read.format("com.crealytics.spark.excel")
            .option("dataAddress", "'Data'!A1")
            .option("header", "true")
            .option("inferSchema", "false")
            .schema(t_schema)
            .load(file.path)
            for file in dbutils.fs.ls(volume_path)
        ]

        parsed_df = reduce(lambda df1, df2: df1.unionAll(df2), excel_dfs)

    finally:
        df.unpersist()
        # Cleanup temp files
        try:
            dbutils.fs.rm(volume_path, recurse=True)
        except:
            pass

    return parsed_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Benchmark

# COMMAND ----------

def run_benchmark(num_files: int, rows_per_file: int, run_pandas: bool = True, run_pyspark: bool = True):
    """Run benchmark comparing both approaches"""

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {num_files} files x {rows_per_file} rows/file")
    print(f"{'='*60}\n")

    # Generate test data
    test_df = create_test_dataframe(num_files, rows_per_file)
    test_df.cache()
    test_df.count()  # Force materialization

    # Approach 1: Pandas
    if run_pandas:
        print("\n--- Running Pandas Approach ---")
        try:
            start_time = time.time()
            result_pandas = process_batch_pandas(spark, test_df, TEST_SCHEMA)
            row_count_pandas = result_pandas.count()
            elapsed_pandas = time.time() - start_time

            metrics.record("Pandas", "execution_time_sec", round(elapsed_pandas, 2))
            metrics.record("Pandas", "output_row_count", row_count_pandas)
            metrics.record("Pandas", "status", "SUCCESS")
            print(f"Pandas: {elapsed_pandas:.2f}s, {row_count_pandas} rows")
        except Exception as e:
            metrics.record("Pandas", "status", "FAILED")
            metrics.record("Pandas", "error", str(e)[:200])
            print(f"Pandas FAILED: {e}")
            traceback.print_exc()

    # Approach 2: PySpark
    if run_pyspark:
        print("\n--- Running PySpark Approach ---")
        try:
            start_time = time.time()
            result_pyspark = process_batch_pyspark(spark, test_df, TEST_SCHEMA)
            row_count_pyspark = result_pyspark.count()
            elapsed_pyspark = time.time() - start_time

            metrics.record("PySpark", "execution_time_sec", round(elapsed_pyspark, 2))
            metrics.record("PySpark", "output_row_count", row_count_pyspark)
            metrics.record("PySpark", "status", "SUCCESS")
            print(f"PySpark: {elapsed_pyspark:.2f}s, {row_count_pyspark} rows")
        except Exception as e:
            metrics.record("PySpark", "status", "FAILED")
            metrics.record("PySpark", "error", str(e)[:200])
            print(f"PySpark FAILED: {e}")
            traceback.print_exc()

    test_df.unpersist()

    return metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Benchmark Tests
# MAGIC
# MAGIC Running multiple test scenarios to compare performance at different scales

# COMMAND ----------

# Test 1: Small scale (warmup)
print("TEST 1: Small scale (10 files x 1000 rows)")
run_benchmark(num_files=10, rows_per_file=1000)

# COMMAND ----------

# Test 2: Medium scale
print("TEST 2: Medium scale (50 files x 5000 rows)")
run_benchmark(num_files=50, rows_per_file=5000)

# COMMAND ----------

# Test 3: Large scale - approximating 1GB
# ~100 files with 50000 rows each should be close to 1GB
print("TEST 3: Large scale (~1GB total)")
run_benchmark(num_files=100, rows_per_file=50000)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

metrics.print_summary()

# COMMAND ----------

# Save results as table for future reference
results_df = metrics.get_results_df()
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Memory Analysis
# MAGIC Check Spark UI for detailed memory metrics after running the benchmark

# COMMAND ----------

print("""
NEXT STEPS:
1. Check Spark UI -> Executors tab for memory usage
2. Check Spark UI -> Stages tab for task distribution
3. Compare 'Peak Memory' between approaches
4. For Pandas failures on large data, note the OOM error

KEY METRICS TO COMPARE:
- Execution time
- Driver memory usage (Pandas uses more)
- Executor utilization (PySpark distributes better)
- Success/Failure rate at different scales
""")
