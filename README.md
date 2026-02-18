# databricks-excel-reader

Read large binary Excel files from Delta tables efficiently on resource-constrained Databricks clusters.

Use Case: SharePoint ingestion pipeline using Databricks Lakeflow Connect

## Problem

Reading binary Excel data stored in Delta tables using PySpark/Pandas fails on shared Databricks clusters due to:

- **Driver OOM** – Spark serializes large binary blobs through driver memory
- **gRPC limits** – Spark Connect enforces 128MB message size limit
- **JVM heap pressure** – Apache POI loads entire workbook into memory

Traditional approaches require 64GB+ clusters to process 500MB Excel files, yet remain slower, leading to increased cloud costs.

## Solution

Bypass Spark's serialization layer entirely:

```
Executor writes binary to UC Volume → Polars reads directly → Parquet → Spark
```

## Benchmarks

Tested on **16GB single-node cluster** (Standard_DC4as_v5, DBR 17.3):

| File Size | Rows | Cols | pandas + openpyxl | This library | Speedup |
|-----------|------|------|-------------------|--------------|---------|
| 100MB | 79,500 | 135 | 2m 2s | 38s | **3.2x** |
| 500MB | 390,000 | 135 | OOM | 130s | ∞ |
| 1GB | 780,000 | 135 | OOM | 449s | ∞ |

## Cluster Sizing

| Excel File Size | Minimum RAM | Recommended RAM |
|-----------------|-------------|-----------------|
| 100MB | 4GB | 8GB |
| 500MB | 8GB | 16GB |
| 1GB | 16GB | 32GB |

## Installation

Add to cluster libraries (Compute → Libraries → Install new → PyPI):

```
fastexcel
polars
```

## Testing

To test with sample data:

1. Run `setup_test_data.py` to generate test Excel and load as binary into Delta table
2. Update configuration in `databricks_excel_reader.py` with your catalog/schema/volume
3. Run `databricks_excel_reader.py`

The setup script generates synthetic data with mixed types (strings, numbers, dates, nulls) similar to real-world Excel files.

## Usage

```python
from databricks_excel_reader import read_binary_excel

df = read_binary_excel(
    spark=spark,
    table_name="catalog.schema.excel_table",
    binary_column="content",
    sheet_name="Sheet1",
    volume_base_path="/Volumes/catalog/schema/volume",
)

df.write.format("delta").saveAsTable("catalog.schema.output")
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `table_name` | Delta table containing binary Excel data | required |
| `binary_column` | Column name with binary content | required |
| `sheet_name` | Excel sheet to read | required |
| `volume_base_path` | Unity Catalog Volume for temp files | required |

## How It Works

`foreachPartition` writes binary to Volume with no return value, avoiding driver memory.

Then:
1. Polars + Calamine (Rust) parses Excel
2. Writes Parquet to Volume
3. Spark reads Parquet natively
4. `localCheckpoint()` breaks lineage
5. Cleanup removes temp files

## Requirements

- Databricks Runtime 13.0+
- Unity Catalog Volume access
- Python packages: `fastexcel`, `polars`

## Testing

Run `setup_test_data.py` in your Databricks workspace to generate test data:

1. Import the notebook
2. Update the configuration (catalog, schema, volume path)
3. Choose test size: `small` (10MB), `medium` (100MB), or `large` (500MB)
4. Run all cells

This creates a Delta table with binary Excel content matching the expected schema.

## Limitations

- Requires Unity Catalog Volume for temp storage
- Single-threaded Excel parsing (Polars limitation)
- Schema inferred from first 10,000 rows

## License

MIT

## Repository Structure

```
databricks-excel-reader/
├── README.md
├── LICENSE
├── databricks_excel_reader.py      # Main solution
├── setup_test_data.py              # Test data generator
├── init_script.sh                  # Cluster setup
└── examples/                       # Comparison approaches
    ├── pandas_approach.py
    └── pyspark_approach.py
    └── new_pyspark_approach.py
```
