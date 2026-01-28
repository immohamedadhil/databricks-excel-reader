# =============================================================================
# PANDAS APPROACH - Current Production Code
# Copy this to a Databricks notebook cell
# =============================================================================

import time
import io
import pandas as pd
from pyspark.sql import DataFrame

def process_batch_pandas(spark, df: DataFrame, table_schema: dict) -> DataFrame:
    """
    Pandas approach - collects all data to driver
    WARNING: Memory-intensive for large datasets
    """
    start_time = time.time()

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

    elapsed = time.time() - start_time
    print(f"[PANDAS] Execution time: {elapsed:.2f} seconds")
    print(f"[PANDAS] Output rows: {parsed_df.count()}")

    return parsed_df
