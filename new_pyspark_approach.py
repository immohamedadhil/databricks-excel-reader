  # =============================================================================
  # OPTIMIZED PYSPARK APPROACH - Binary Excel to DataFrame
  # =============================================================================
  import uuid
  import time
  from pyspark.sql import DataFrame
  from pyspark.sql.functions import monotonically_increasing_id, spark_partition_id
  from pyspark import StorageLevel

  def read_binary_excel_pyspark_optimized(
      spark,
      table_name: str,
      sheet_name: str = "result",
      header: bool = True,
      volume_base_path: str = "/Volumes/adhil/test/weather_data",
      num_partitions: int = None
  ) -> DataFrame:
      """
      Optimized PySpark approach for reading binary Excel from Delta table.

      Optimizations:
      - Wildcard read (single spark.read instead of loop)
      - Optimal partitioning for parallel writes
      - Single schema inference
      - Proper DBFS path handling
      - Performance timing
      """
      start_time = time.time()

      # 1. Read binary data from table
      df = spark.table(table_name)

      # 2. Setup unique temp path
      unique_id = uuid.uuid4().hex[:8]
      volume_path = f"{volume_base_path}/temp_{unique_id}"

      # Convert to DBFS path for dbutils operations
      dbfs_path = volume_path.replace("/Volumes/", "/Volumes/")

      # 3. Add ID and optimize partitions
      file_count = df.count()

      # Optimal partitions: 1 partition per file for parallel writes
      optimal_partitions = num_partitions or min(file_count, 8)

      df_with_id = (
          df.withColumn("id", monotonically_increasing_id())
          .repartition(optimal_partitions)
          .persist(StorageLevel.MEMORY_AND_DISK)
      )

      # Force materialization
      df_with_id.count()
      print(f"[1/4] Loaded {file_count} file(s), partitions: {optimal_partitions}")

      try:
          # 4. Create temp directory
          dbutils.fs.mkdirs(dbfs_path)

          # 5. Write binary files in parallel using foreachPartition
          def write_partition(partition):
              for row in partition:
                  file_path = f"{volume_path}/{row.id}.xlsx"
                  with open(file_path, "wb") as f:
                      f.write(row["content_inline_content"])

          df_with_id.foreachPartition(write_partition)

          write_time = time.time()
          print(f"[2/4] Written temp files ({write_time - start_time:.1f}s)")

          # 6. Read ALL Excel files with single wildcard read (KEY OPTIMIZATION)
          result_df = (
              spark.read
              .format("excel")
              .option("header", str(header).lower())
              .option("sheetName", sheet_name)
              .option("inferSchema", "true")
              .load(f"{dbfs_path}/*.xlsx")  # Wildcard - reads all files at once
          )

          read_time = time.time()
          print(f"[3/4] Read Excel files ({read_time - write_time:.1f}s)")

          # 7. Force evaluation and get stats
          row_count = result_df.count()
          col_count = len(result_df.columns)

          end_time = time.time()
          print(f"[4/4] Result: {row_count:,} rows x {col_count} columns ({end_time - read_time:.1f}s)")
          print(f"[TOTAL] {end_time - start_time:.1f}s")

      finally:
          # 8. Cleanup
          df_with_id.unpersist()
          try:
              dbutils.fs.rm(dbfs_path, recurse=True)
          except:
              pass

      return result_df


  # =============================================================================
  # ALTERNATIVE: Even faster for single large file
  # =============================================================================
  def read_binary_excel_direct(
      spark,
      table_name: str,
      sheet_name: str = "result",
      header: bool = True,
      volume_base_path: str = "/Volumes/adhil/test/weather_data"
  ) -> DataFrame:
      """
      Fastest approach for SINGLE file - writes directly without partitioning overhead.
      """
      start_time = time.time()

      # 1. Get binary content directly (single file)
      row = spark.table(table_name).first()
      xlsx_bytes = row["content_inline_content"]

      print(f"[1/3] Loaded binary: {len(xlsx_bytes):,} bytes")

      # 2. Write to temp file
      unique_id = uuid.uuid4().hex[:8]
      temp_path = f"{volume_base_path}/temp_{unique_id}.xlsx"

      with open(temp_path, "wb") as f:
          f.write(xlsx_bytes)

      print(f"[2/3] Written temp file")

      # 3. Read with native Excel reader
      result_df = (
          spark.read
          .format("excel")
          .option("header", str(header).lower())
          .option("sheetName", sheet_name)
          .option("inferSchema", "true")
          .load(temp_path)
      )

    #   row_count = result_df.count()
      print(f"""[3/3] Result: row_count:, rows ({time.time() - start_time:.1f}s)""")

      # Cleanup
    #   try:
    #       dbutils.fs.rm(temp_path)
    #   except:
    #       import os
    #       os.remove(temp_path)

      return result_df


  # =============================================================================
  # USAGE
  # =============================================================================
  TABLE_NAME = "adhil.test.excel_binary_data_100mb"

  # Option 1: Multiple files (distributed)
  # pyspark_df = read_binary_excel_pyspark_optimized(
  #     spark=spark,
  #     table_name=TABLE_NAME,
  #     sheet_name="result",
  #     header=True
  # )

  # Option 2: Single file (fastest) - USE THIS FOR YOUR CASE
  pyspark_df = read_binary_excel_direct(
      spark=spark,
      table_name=TABLE_NAME,
      sheet_name="result",
      header=True
  )

#   display(pyspark_df.limit(20))
#   pyspark_df.printSchema()