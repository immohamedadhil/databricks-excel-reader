# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Test Data
# MAGIC
# MAGIC Creates a Delta table with binary Excel content for testing `databricks-excel-reader`.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Unity Catalog Volume access
# MAGIC - Catalog/schema where you can create tables

# COMMAND ----------

# MAGIC %pip install openpyxl --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================

CATALOG = "your_catalog"             # Your catalog
SCHEMA = "your_schema"               # Your schema
TABLE_NAME = "excel_binary_data"     # Table to create
VOLUME_PATH = "/Volumes/your_catalog/your_schema/your_volume"  # Your volume

# Test file size: "small" (10MB), "medium" (100MB), "large" (500MB)
TEST_SIZE = "medium"

# COMMAND ----------

import pandas as pd
import numpy as np
import os
import uuid

# =============================================================================
# GENERATE TEST EXCEL FILE
# =============================================================================


SIZE_CONFIG = {
    "small":  {"rows": 20_000,   "cols": 135, "expected_mb": 10},
    "medium": {"rows": 200_000,  "cols": 135, "expected_mb": 100},
    "large":  {"rows": 1_000_000, "cols": 135, "expected_mb": 500},
}
config = SIZE_CONFIG[TEST_SIZE]
rows, cols = config["rows"], config["cols"]

print(f"Generating {TEST_SIZE} test file: {rows:,} rows x {cols} cols (~{config['expected_mb']}MB)")
print("This may take a few minutes for large files...")

# Generate realistic mixed-type data
np.random.seed(42)

data = {}
for i in range(cols):
    col_type = i % 5
    if col_type == 0:  # String
        data[f"text_col_{i}"] = np.random.choice(
            ["alpha", "beta", "gamma", "delta", None, ""], size=rows
        )
    elif col_type == 1:  # Integer
        data[f"int_col_{i}"] = np.random.choice(
            [1, 2, 3, 100, 999, None], size=rows
        )
    elif col_type == 2:  # Float
        data[f"float_col_{i}"] = np.random.choice(
            [1.23, 45.67, 89.01, None, 0.0], size=rows
        )
    elif col_type == 3:  # Date-like string
        data[f"date_col_{i}"] = np.random.choice(
            ["2024-01-01", "2024-06-15", "2024-12-31", None, ""], size=rows
        )
    else:  # Long text
        data[f"desc_col_{i}"] = np.random.choice(
            ["Lorem ipsum dolor sit amet", "Consectetur adipiscing elit", 
             "Sed do eiusmod tempor", None, ""], size=rows
        )

df = pd.DataFrame(data)
print(f"DataFrame created: {df.shape}")

# COMMAND ----------

# =============================================================================
# SAVE TO EXCEL
# =============================================================================

# Write to local temp first (pandas can't write directly to Volume)
local_path = f"/tmp/test_data_{TEST_SIZE}.xlsx"
volume_path = f"{VOLUME_PATH}/test_data_{TEST_SIZE}.xlsx"

print(f"Writing Excel file to {local_path}...")

# Use context manager to properly close file
with pd.ExcelWriter(local_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="result", index=False)

file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
print(f"Excel file created: {file_size_mb:.1f} MB")

# Copy to Volume using shutil (works on shared clusters)
import shutil
print(f"Copying to {volume_path}...")
shutil.copy(local_path, volume_path)
print("Done")

excel_path = volume_path  # Use volume path for next steps

# COMMAND ----------

# =============================================================================
# CREATE DELTA TABLE WITH BINARY CONTENT
# =============================================================================

full_table_name = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
print(f"Creating Delta table: {full_table_name}")

# Read Excel as binary
with open(excel_path, "rb") as f:
    binary_content = f.read()

print(f"Binary content size: {len(binary_content) / (1024*1024):.1f} MB")

# Create DataFrame with binary content
from pyspark.sql.types import StructType, StructField, BinaryType, StringType

schema = StructType([
    StructField("file_id", StringType(), False),
    StructField("file_name", StringType(), False),
    StructField("content_inline_content", BinaryType(), False),
])

data = [(
    str(uuid.uuid4()),
    f"test_data_{TEST_SIZE}.xlsx",
    binary_content,
)]

binary_df = spark.createDataFrame(data, schema)

# Write to Delta table
binary_df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)

print(f"Delta table created: {full_table_name}")

# COMMAND ----------

# =============================================================================
# VERIFY
# =============================================================================

# Check table
result = spark.sql(f"DESCRIBE DETAIL {full_table_name}")
size_bytes = result.select("sizeInBytes").collect()[0][0]
print(f"Table size: {size_bytes / (1024*1024):.1f} MB")

# Show schema
spark.table(full_table_name).printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC
# MAGIC Now you can test the reader:
# MAGIC
# MAGIC ```python
# MAGIC result_df = read_binary_excel(
# MAGIC     spark=spark,
# MAGIC     table_name="your_catalog.your_schema.excel_binary_data",
# MAGIC     binary_column="content_inline_content",
# MAGIC     sheet_name="result",
# MAGIC     volume_base_path="/Volumes/your_catalog/your_schema/your_volume",
# MAGIC )
# MAGIC ```

# COMMAND ----------

# Cleanup: Remove the temp Excel file (optional)
# dbutils.fs.rm(excel_path)