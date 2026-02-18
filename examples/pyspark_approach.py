# =============================================================================
# PYSPARK APPROACH - Optimized Code (with foreachPartition fix)
# Copy this to a Databricks notebook cell
# =============================================================================

import time
import uuid
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import monotonically_increasing_id
from pyspark import StorageLevel

def process_batch_pyspark(spark, df: DataFrame, table_schema: dict, volume_base_path: str = "/tmp/benchmark_excel") -> DataFrame:
    """
    PySpark approach - distributed processing using foreachPartition

    Args:
        spark: SparkSession
        df: DataFrame with 'content_inline_content' column (binary Excel data)
        table_schema: dict of column names to types
        volume_base_path: base path for temporary Excel files
    """
    start_time = time.time()

    t_schema = StructType(
        [StructField(c, StringType(), True) for c in table_schema.keys()]
    )

    unique_id = str(uuid.uuid4().hex)
    volume_path = f"{volume_base_path}/{unique_id}"

    def write_binary_to_file(partition):
        """Write each row's binary content to a temp Excel file"""
        for row in partition:
            file_path = f"{volume_path}/{row.id}.xlsx"
            with open(file_path, "wb") as file:
                file.write(row['content_inline_content'])

    # Add ID and persist
    df = df.withColumn("id", monotonically_increasing_id())
    df = df.persist(StorageLevel.MEMORY_AND_DISK)

    try:
        # Create directory
        dbutils.fs.mkdirs(volume_path)

        # Write files using foreachPartition (NOT foreach)
        df.foreachPartition(write_binary_to_file)

        # Read Excel files using spark-excel
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

        # Force evaluation to measure time accurately
        row_count = parsed_df.count()

        elapsed = time.time() - start_time
        print(f"[PYSPARK] Execution time: {elapsed:.2f} seconds")
        print(f"[PYSPARK] Output rows: {row_count}")
        print(f"[PYSPARK] Files processed: {len(files)}")

    finally:
        df.unpersist()
        # Cleanup temp files
        try:
            dbutils.fs.rm(volume_path, recurse=True)
            print(f"[PYSPARK] Cleaned up temp files at {volume_path}")
        except Exception as e:
            print(f"[PYSPARK] Cleanup warning: {e}")

    return parsed_df
