# =============================================================================
# BENCHMARK RUNNER
# Copy this to a Databricks notebook and run all cells
# =============================================================================

# CELL 1: Imports and Setup
# -----------------------------------------------------------------------------
import time
import io
import uuid
import pandas as pd
from functools import reduce
from datetime import datetime
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, BinaryType
from pyspark.sql.functions import monotonically_increasing_id, col
from pyspark import StorageLevel

print("Imports loaded successfully")

# CELL 2: Configuration
# -----------------------------------------------------------------------------
# Test schema - adjust to match your actual Excel structure
TEST_SCHEMA = {
    'Column1': 'str',
    'Column2': 'str',
    'Column3': 'str',
    'Column4': 'str',
    'Column5': 'str'
}

VOLUME_BASE_PATH = "/tmp/benchmark_excel"

print(f"Schema: {TEST_SCHEMA}")

# CELL 3: Generate Sample Binary Excel Data
# -----------------------------------------------------------------------------
def generate_sample_excel_binary(num_rows: int = 10000) -> bytes:
    """Generate a sample Excel file as binary content (in-memory)"""
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

    # Save to bytes (in-memory, no file on disk)
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def create_test_dataframe(num_files: int, rows_per_file: int) -> DataFrame:
    """Create a test DataFrame with binary Excel content"""
    print(f"Generating {num_files} Excel files with {rows_per_file} rows each...")

    binary_data = []
    for i in range(num_files):
        excel_bytes = generate_sample_excel_binary(rows_per_file)
        binary_data.append((str(i), excel_bytes))
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_files} files...")

    schema = StructType([
        StructField("file_id", StringType(), True),
        StructField("content_inline_content", BinaryType(), True)
    ])

    df = spark.createDataFrame(binary_data, schema=schema)

    total_size_mb = sum(len(data) for _, data in binary_data) / (1024 * 1024)
    print(f"Total data size: {total_size_mb:.2f} MB")

    return df, total_size_mb

print("Test data generator ready")

# CELL 4: Pandas Approach
# -----------------------------------------------------------------------------
def process_batch_pandas(spark, df: DataFrame, table_schema: dict) -> DataFrame:
    """Pandas approach - collects all data to driver"""

    # 1) Convert to Pandas
    pdf = df.toPandas()

    # 2) Process each Excel file
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
        ).astype(table_schema)

        processed_records.extend(pdf_excel.to_dict(orient="records"))

    # 3) Convert back to Spark
    parsed_df = spark.createDataFrame(processed_records)
    parsed_df = parsed_df.na.replace(['#N/A', 'nan'], None)

    return parsed_df

print("Pandas approach ready")

# CELL 5: PySpark Approach
# -----------------------------------------------------------------------------
def process_batch_pyspark(spark, df: DataFrame, table_schema: dict) -> DataFrame:
    """PySpark approach - distributed processing"""

    t_schema = StructType(
        [StructField(c, StringType(), True) for c in table_schema.keys()]
    )

    unique_id = str(uuid.uuid4().hex)
    volume_path = f"{VOLUME_BASE_PATH}/{unique_id}"

    def write_binary_to_file(partition):
        for row in partition:
            file_path = f"{volume_path}/{row.id}.xlsx"
            with open(file_path, "wb") as file:
                file.write(row['content_inline_content'])

    # Add ID and persist
    df_with_id = df.withColumn("id", monotonically_increasing_id())
    df_with_id = df_with_id.persist(StorageLevel.MEMORY_AND_DISK)

    try:
        dbutils.fs.mkdirs(volume_path)
        df_with_id.foreachPartition(write_binary_to_file)

        files = dbutils.fs.ls(volume_path)
        excel_dfs = [
            spark.read.format("com.crealytics.spark.excel")
            .option("dataAddress", "'Data'!A1")
            .option("header", "true")
            .option("inferSchema", "false")
            .schema(t_schema)
            .load(file.path)
            for file in files
        ]

        parsed_df = reduce(lambda df1, df2: df1.unionAll(df2), excel_dfs)

    finally:
        df_with_id.unpersist()
        try:
            dbutils.fs.rm(volume_path, recurse=True)
        except:
            pass

    return parsed_df

print("PySpark approach ready")

# CELL 6: Run Benchmark
# -----------------------------------------------------------------------------
def run_benchmark(num_files: int, rows_per_file: int, run_pandas: bool = True, run_pyspark: bool = True):
    """Run benchmark comparing both approaches"""

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {num_files} files x {rows_per_file} rows/file")
    print(f"{'='*60}")

    # Generate test data
    test_df, total_size_mb = create_test_dataframe(num_files, rows_per_file)
    test_df.cache()
    test_df.count()  # Force materialization

    results = {"data_size_mb": total_size_mb, "num_files": num_files, "rows_per_file": rows_per_file}

    # Pandas approach
    if run_pandas:
        print("\n--- Running Pandas Approach ---")
        try:
            start = time.time()
            result_pandas = process_batch_pandas(spark, test_df, TEST_SCHEMA)
            count_pandas = result_pandas.count()
            elapsed_pandas = time.time() - start

            results["pandas_time_sec"] = round(elapsed_pandas, 2)
            results["pandas_rows"] = count_pandas
            results["pandas_status"] = "SUCCESS"
            print(f"Pandas: {elapsed_pandas:.2f}s, {count_pandas} rows")
        except Exception as e:
            results["pandas_status"] = f"FAILED: {str(e)[:100]}"
            print(f"Pandas FAILED: {e}")

    # PySpark approach
    if run_pyspark:
        print("\n--- Running PySpark Approach ---")
        try:
            start = time.time()
            result_pyspark = process_batch_pyspark(spark, test_df, TEST_SCHEMA)
            count_pyspark = result_pyspark.count()
            elapsed_pyspark = time.time() - start

            results["pyspark_time_sec"] = round(elapsed_pyspark, 2)
            results["pyspark_rows"] = count_pyspark
            results["pyspark_status"] = "SUCCESS"
            print(f"PySpark: {elapsed_pyspark:.2f}s, {count_pyspark} rows")
        except Exception as e:
            results["pyspark_status"] = f"FAILED: {str(e)[:100]}"
            print(f"PySpark FAILED: {e}")

    test_df.unpersist()

    return results

# CELL 7: Execute Tests
# -----------------------------------------------------------------------------
all_results = []

# Test 1: Small (warmup)
print("\n" + "="*60)
print("TEST 1: SMALL (10 files x 1,000 rows)")
print("="*60)
r1 = run_benchmark(num_files=10, rows_per_file=1000)
all_results.append(r1)

# Test 2: Medium
print("\n" + "="*60)
print("TEST 2: MEDIUM (50 files x 5,000 rows)")
print("="*60)
r2 = run_benchmark(num_files=50, rows_per_file=5000)
all_results.append(r2)

# Test 3: Large (~1GB) - This may cause OOM for Pandas
print("\n" + "="*60)
print("TEST 3: LARGE (100 files x 50,000 rows) - ~1GB")
print("="*60)
r3 = run_benchmark(num_files=100, rows_per_file=50000)
all_results.append(r3)

# CELL 8: Summary
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

for i, r in enumerate(all_results, 1):
    print(f"\nTest {i}: {r.get('num_files')} files x {r.get('rows_per_file')} rows ({r.get('data_size_mb', 0):.1f} MB)")
    print(f"  Pandas:  {r.get('pandas_time_sec', 'N/A')}s - {r.get('pandas_status', 'N/A')}")
    print(f"  PySpark: {r.get('pyspark_time_sec', 'N/A')}s - {r.get('pyspark_status', 'N/A')}")

    if r.get('pandas_time_sec') and r.get('pyspark_time_sec'):
        speedup = r['pandas_time_sec'] / r['pyspark_time_sec']
        print(f"  Speedup: {speedup:.2f}x {'(PySpark faster)' if speedup > 1 else '(Pandas faster)'}")

print("\n" + "="*60)
print("Copy these results and share with Claude for analysis")
print("="*60)
