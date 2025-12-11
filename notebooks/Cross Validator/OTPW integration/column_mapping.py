"""
column_mapping.py - Map OTPW column names to CUSTOM column names

This module provides a function to transform OTPW DataFrame columns to match
the CUSTOM join schema. This ensures seamless integration with the existing
pipeline.

Usage:
    from column_mapping import map_otpw_columns_to_custom
    
    # After loading OTPW data
    df_otpw = spark.read.parquet("path/to/otpw/data.parquet")
    df_mapped = map_otpw_columns_to_custom(df_otpw)
    
    # Now df_mapped has columns matching CUSTOM schema
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# ============================================================================
# COLUMN MAPPING DICTIONARY
# ============================================================================
# This dictionary maps OTPW column names to CUSTOM column names.
# TODO: Fill this in after running print_columns.py and comparing outputs
# Format: "otpw_column_name": "custom_column_name"

COLUMN_MAPPING = {
    # ============================================================================
    # Core Flight Columns
    # ============================================================================
    # NOTE: Labels (DEP_DELAY, DEP_DEL15, SEVERE_DEL60) are UPPERCASE in CUSTOM
    # Other columns are lowercase. This matches split.py and cv.py expectations.
    "FL_DATE": "fl_date",  # Date column is lowercase
    "DEP_DELAY": "DEP_DELAY",  # Label is UPPERCASE in CUSTOM (matches cv.py FlightDelayEvaluator)
    "DEP_DEL15": "DEP_DEL15",  # Label is UPPERCASE in CUSTOM
    "SEVERE_DEL60": "SEVERE_DEL60",  # Label is UPPERCASE in CUSTOM
    "ORIGIN": "origin",
    "DEST": "dest",
    "OP_CARRIER": "op_carrier",
    "OP_CARRIER_AIRLINE_ID": "op_carrier_airline_id",
    "OP_CARRIER_FL_NUM": "op_carrier_fl_num",
    "OP_UNIQUE_CARRIER": "op_unique_carrier",
    "TAIL_NUM": "tail_num",
    
    # ============================================================================
    # Time Columns
    # ============================================================================
    "DEP_TIME": "dep_time",
    "DEP_TIME_BLK": "dep_time_blk",
    "ARR_TIME": "arr_time",
    "ARR_TIME_BLK": "arr_time_blk",
    "CRS_DEP_TIME": "crs_dep_time",
    "CRS_ARR_TIME": "crs_arr_time",
    "CRS_ELAPSED_TIME": "crs_elapsed_time",
    "ACTUAL_ELAPSED_TIME": "actual_elapsed_time",
    "AIR_TIME": "air_time",
    "FIRST_DEP_TIME": "first_dep_time",
    "WHEELS_OFF": "wheels_off",
    "WHEELS_ON": "wheels_on",
    
    # ============================================================================
    # Delay Columns
    # ============================================================================
    "ARR_DELAY": "arr_delay",
    "ARR_DEL15": "arr_del15",
    "ARR_DELAY_GROUP": "arr_delay_group",
    "ARR_DELAY_NEW": "arr_delay_new",
    "DEP_DELAY_GROUP": "dep_delay_group",
    "DEP_DELAY_NEW": "dep_delay_new",
    "CARRIER_DELAY": "carrier_delay",
    "WEATHER_DELAY": "weather_delay",
    "NAS_DELAY": "nas_delay",
    "SECURITY_DELAY": "security_delay",
    "LATE_AIRCRAFT_DELAY": "late_aircraft_delay",
    
    # ============================================================================
    # Taxi Times
    # ============================================================================
    "TAXI_IN": "taxi_in",
    "TAXI_OUT": "taxi_out",
    
    # ============================================================================
    # Distance and Location
    # ============================================================================
    "DISTANCE": "distance",
    "DISTANCE_GROUP": "distance_group",
    "ELEVATION": "elevation",
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
    
    # ============================================================================
    # Airport Information
    # ============================================================================
    "ORIGIN_AIRPORT_ID": "origin_airport_id",
    "ORIGIN_AIRPORT_SEQ_ID": "origin_airport_seq_id",
    "ORIGIN_CITY_MARKET_ID": "origin_city_market_id",
    "ORIGIN_CITY_NAME": "origin_city_name",
    "ORIGIN_STATE_ABR": "origin_state_abr",
    "ORIGIN_STATE_FIPS": "origin_state_fips",
    "ORIGIN_STATE_NM": "origin_state_nm",
    "ORIGIN_WAC": "origin_wac",
    
    "DEST_AIRPORT_ID": "dest_airport_id",
    "DEST_AIRPORT_SEQ_ID": "dest_airport_seq_id",
    "DEST_CITY_MARKET_ID": "dest_city_market_id",
    "DEST_CITY_NAME": "dest_city_name",
    "DEST_STATE_ABR": "dest_state_abr",
    "DEST_STATE_FIPS": "dest_state_fips",
    "DEST_STATE_NM": "dest_state_nm",
    "DEST_WAC": "dest_wac",
    
    # ============================================================================
    # Date/Time Components
    # ============================================================================
    "YEAR": "year",
    "MONTH": "month",
    "DAY_OF_MONTH": "day_of_month",
    "DAY_OF_WEEK": "day_of_week",
    "QUARTER": "quarter",
    "DATE": "date",
    
    # ============================================================================
    # Flight Status
    # ============================================================================
    "CANCELLED": "cancelled",
    "CANCELLATION_CODE": "cancellation_code",
    "DIVERTED": "diverted",
    "FLIGHTS": "flights",
    
    # ============================================================================
    # Additional Time Fields
    # ============================================================================
    "TOTAL_ADD_GTIME": "total_add_gtime",
    "LONGEST_ADD_GTIME": "longest_add_gtime",
    
    # ============================================================================
    # Weather Station Information
    # ============================================================================
    "STATION": "station",
    "NAME": "name",
    "SOURCE": "source",
    "REPORT_TYPE": "report_type",
    "REM": "rem",
    
    # ============================================================================
    # Hourly Weather Variables (Critical for modeling)
    # ============================================================================
    "HourlyPrecipitation": "hourlyprecipitation",
    "HourlySeaLevelPressure": "hourlysealevelpressure",
    "HourlyAltimeterSetting": "hourlyaltimetersetting",
    "HourlyWetBulbTemperature": "hourlywetbulbtemperature",
    "HourlyStationPressure": "hourlystationpressure",
    "HourlyWindDirection": "hourlywinddirection",
    "HourlyRelativeHumidity": "hourlyrelativehumidity",
    "HourlyWindSpeed": "hourlywindspeed",
    "HourlyDewPointTemperature": "hourlydewpointtemperature",
    "HourlyDryBulbTemperature": "hourlydrybulbtemperature",
    "HourlyVisibility": "hourlyvisibility",
    "HourlyPresentWeatherType": "hourlypresentweathertype",
    "HourlyPressureChange": "hourlypressurechange",
    "HourlyPressureTendency": "hourlypressuretendency",
    "HourlySkyConditions": "hourlyskyconditions",
    "HourlyWindGustSpeed": "hourlywindgustspeed",
    
    # ============================================================================
    # Daily Weather Variables
    # ============================================================================
    "DailyAverageDewPointTemperature": "dailyaveragedewpointtemperature",
    "DailyAverageDryBulbTemperature": "dailyaveragedrybulbtemperature",
    "DailyAverageRelativeHumidity": "dailyaveragerelativehumidity",
    "DailyAverageSeaLevelPressure": "dailyaveragesealevelpressure",
    "DailyAverageStationPressure": "dailyaveragestationpressure",
    "DailyAverageWetBulbTemperature": "dailyaveragewetbulbtemperature",
    "DailyAverageWindSpeed": "dailyaveragewindspeed",
    "DailyCoolingDegreeDays": "dailycoolingdegreedays",
    "DailyDepartureFromNormalAverageTemperature": "dailydeparturefromnormalaveragetemperature",
    "DailyHeatingDegreeDays": "dailyheatingdegreedays",
    "DailyMaximumDryBulbTemperature": "dailymaximumdrybulbtemperature",
    "DailyMinimumDryBulbTemperature": "dailyminimumdrybulbtemperature",
    "DailyPeakWindDirection": "dailypeakwinddirection",
    "DailyPeakWindSpeed": "dailypeakwindspeed",
    "DailyPrecipitation": "dailyprecipitation",
    "DailySnowDepth": "dailysnowdepth",
    "DailySnowfall": "dailysnowfall",
    "DailySustainedWindDirection": "dailysustainedwinddirection",
    "DailySustainedWindSpeed": "dailysustainedwindspeed",
    "DailyWeather": "dailyweather",
    
    # ============================================================================
    # Monthly Weather Variables
    # ============================================================================
    "MonthlyAverageRH": "monthlyaveragerh",
    "MonthlyDaysWithGT001Precip": "monthlydayswithgt001precip",
    "MonthlyDaysWithGT010Precip": "monthlydayswithgt010precip",
    "MonthlyDaysWithGT32Temp": "monthlydayswithgt32temp",
    "MonthlyDaysWithGT90Temp": "monthlydayswithgt90temp",
    "MonthlyDaysWithLT0Temp": "monthlydayswithlt0temp",
    "MonthlyDaysWithLT32Temp": "monthlydayswithlt32temp",
    "MonthlyDepartureFromNormalAverageTemperature": "monthlydeparturefromnormalaveragetemperature",
    "MonthlyDepartureFromNormalCoolingDegreeDays": "monthlydeparturefromnormalcoolingdegreedays",
    "MonthlyDepartureFromNormalHeatingDegreeDays": "monthlydeparturefromnormalheatingdegreedays",
    "MonthlyDepartureFromNormalMaximumTemperature": "monthlydeparturefromnormalmaximumtemperature",
    "MonthlyDepartureFromNormalMinimumTemperature": "monthlydeparturefromnormalminimumtemperature",
    "MonthlyDepartureFromNormalPrecipitation": "monthlydeparturefromnormalprecipitation",
    "MonthlyDewpointTemperature": "monthlydewpointtemperature",
    "MonthlyGreatestPrecip": "monthlygreatestprecip",
    "MonthlyGreatestPrecipDate": "monthlygreatestprecipdate",
    "MonthlyGreatestSnowDepth": "monthlygreatestsnowdepth",
    "MonthlyGreatestSnowDepthDate": "monthlygreatestsnowdepthdate",
    "MonthlyGreatestSnowfall": "monthlygreatestsnowfall",
    "MonthlyGreatestSnowfallDate": "monthlygreatestsnowfalldate",
    "MonthlyMaxSeaLevelPressureValue": "monthlymaxsealevelpressurevalue",
    "MonthlyMaxSeaLevelPressureValueDate": "monthlymaxsealevelpressurevaluedate",
    "MonthlyMaxSeaLevelPressureValueTime": "monthlymaxsealevelpressurevaluetime",
    "MonthlyMaximumTemperature": "monthlymaximumtemperature",
    "MonthlyMeanTemperature": "monthlymeantemperature",
    "MonthlyMinSeaLevelPressureValue": "monthlyminsealevelpressurevalue",
    "MonthlyMinSeaLevelPressureValueDate": "monthlyminsealevelpressurevaluedate",
    "MonthlyMinSeaLevelPressureValueTime": "monthlyminsealevelpressurevaluetime",
    "MonthlyMinimumTemperature": "monthlyminimumtemperature",
    "MonthlySeaLevelPressure": "monthlysealevelpressure",
    "MonthlyStationPressure": "monthlystationpressure",
    "MonthlyTotalLiquidPrecipitation": "monthlytotalliquidprecipitation",
    "MonthlyTotalSnowfall": "monthlytotalsnowfall",
    "MonthlyWetBulb": "monthlywetbulb",
    
    # ============================================================================
    # Short Duration Precipitation
    # ============================================================================
    "ShortDurationEndDate005": "shortdurationenddate005",
    "ShortDurationEndDate010": "shortdurationenddate010",
    "ShortDurationEndDate015": "shortdurationenddate015",
    "ShortDurationEndDate020": "shortdurationenddate020",
    "ShortDurationEndDate030": "shortdurationenddate030",
    "ShortDurationEndDate045": "shortdurationenddate045",
    "ShortDurationEndDate060": "shortdurationenddate060",
    "ShortDurationEndDate080": "shortdurationenddate080",
    "ShortDurationEndDate100": "shortdurationenddate100",
    "ShortDurationEndDate120": "shortdurationenddate120",
    "ShortDurationEndDate150": "shortdurationenddate150",
    "ShortDurationEndDate180": "shortdurationenddate180",
    "ShortDurationPrecipitationValue005": "shortdurationprecipitationvalue005",
    "ShortDurationPrecipitationValue010": "shortdurationprecipitationvalue010",
    "ShortDurationPrecipitationValue015": "shortdurationprecipitationvalue015",
    "ShortDurationPrecipitationValue020": "shortdurationprecipitationvalue020",
    "ShortDurationPrecipitationValue030": "shortdurationprecipitationvalue030",
    "ShortDurationPrecipitationValue045": "shortdurationprecipitationvalue045",
    "ShortDurationPrecipitationValue060": "shortdurationprecipitationvalue060",
    "ShortDurationPrecipitationValue080": "shortdurationprecipitationvalue080",
    "ShortDurationPrecipitationValue100": "shortdurationprecipitationvalue100",
    "ShortDurationPrecipitationValue120": "shortdurationprecipitationvalue120",
    "ShortDurationPrecipitationValue150": "shortdurationprecipitationvalue150",
    "ShortDurationPrecipitationValue180": "shortdurationprecipitationvalue180",
    
    # ============================================================================
    # Normals and Other Weather
    # ============================================================================
    "NormalsCoolingDegreeDay": "normalscoolingdegreeday",
    "NormalsHeatingDegreeDay": "normalsheatingdegreeday",
    "AWND": "awnd",
    "CDSD": "cdsd",
    "CLDD": "cldd",
    "DSNW": "dsnw",
    "HDSD": "hdsd",
    "HTDD": "htdd",
    "Sunrise": "sunrise",
    "Sunset": "sunset",
    "WindEquipmentChangeDate": "windequipmentchangedate",
    
    # ============================================================================
    # Backup/Additional Station Info
    # ============================================================================
    "BackupDirection": "backupdirection",
    "BackupDistance": "backupdistance",
    "BackupDistanceUnit": "backupdistanceunit",
    "BackupElements": "backupelements",
    "BackupElevation": "backupelevation",
    "BackupEquipment": "backupequipment",
    "BackupLatitude": "backuplatitude",
    "BackupLongitude": "backuplongitude",
    "BackupName": "backupname",
    
    # ============================================================================
    # OTPW-specific columns (from Indri's processing)
    # These map to lowercase versions if they exist in CUSTOM, otherwise will be kept
    # ============================================================================
    "dest_airport_lat": "dest_airport_lat",
    "dest_airport_lon": "dest_airport_lon",
    "dest_airport_name": "dest_airport_name",
    "dest_iata_code": "dest_iata_code",
    "dest_icao": "dest_icao",
    "dest_region": "dest_region",
    "dest_station_dis": "dest_station_dis",
    "dest_station_id": "dest_station_id",
    "dest_station_lat": "dest_station_lat",
    "dest_station_lon": "dest_station_lon",
    "dest_station_name": "dest_station_name",
    "dest_type": "dest_type",
    "four_hours_prior_depart_UTC": "four_hours_prior_depart_utc",
    "origin_airport_lat": "origin_airport_lat",
    "origin_airport_lon": "origin_airport_lon",
    "origin_airport_name": "origin_airport_name",
    "origin_iata_code": "origin_iata_code",
    "origin_icao": "origin_icao",
    "origin_region": "origin_region",
    "origin_station_dis": "origin_station_dis",
    "origin_station_id": "origin_station_id",
    "origin_station_lat": "origin_station_lat",
    "origin_station_lon": "origin_station_lon",
    "origin_station_name": "origin_station_name",
    "origin_type": "origin_type",
    "sched_depart_date_time": "sched_depart_date_time",
    "sched_depart_date_time_UTC": "sched_depart_date_time_utc",
    "two_hours_prior_depart_UTC": "two_hours_prior_depart_utc",
    "_row_desc": "_row_desc",
}

# Columns that should be dropped (OTPW-specific columns not in CUSTOM)
# Note: Most OTPW columns are mapped above. Only drop columns that are truly
# OTPW-specific and not needed. The flight lineage features will be added
# by the feature engineering pipeline, so we don't need to drop columns
# that don't exist in base OTPW.
COLUMNS_TO_DROP = [
    # Add any truly OTPW-specific columns that should be removed here
    # Most columns are kept even if they don't exist in CUSTOM base folds,
    # as they may be useful or will be handled by feature engineering
]

# Columns that should be added (CUSTOM columns not in OTPW, with default values)
# Format: {"column_name": default_value}
# Note: Flight lineage features (lineage_rank, prev_flight_*, etc.) will be
# added by the feature engineering pipeline (add_engineered_features_to_folds),
# so we don't need to add them here. Only add columns that are expected
# in the base dataset before feature engineering.
COLUMNS_TO_ADD = {
    # Most missing columns are flight lineage features that will be added
    # by the feature engineering pipeline. Only add base columns here if needed.
}


# ============================================================================
# MAPPING FUNCTION
# ============================================================================

def map_otpw_columns_to_custom(df: DataFrame, verbose: bool = True) -> DataFrame:
    """
    Transform OTPW DataFrame columns to match CUSTOM schema.
    
    This function:
    1. Renames columns according to COLUMN_MAPPING
    2. Drops columns in COLUMNS_TO_DROP
    3. Adds columns in COLUMNS_TO_ADD with default values
    4. Ensures column order matches CUSTOM (optional, for consistency)
    
    Args:
        df: PySpark DataFrame with OTPW column names
        verbose: Whether to print progress messages (default: True)
        
    Returns:
        PySpark DataFrame with CUSTOM column names
        
    Example:
        >>> df_otpw = spark.read.parquet("otpw_data.parquet")
        >>> df_custom = map_otpw_columns_to_custom(df_otpw)
        >>> # Now df_custom has columns matching CUSTOM schema
    """
    # Store original columns for reference
    original_columns = set(df.columns)
    result_df = df
    
    # Step 1: Rename columns according to mapping
    # Track all columns that are explicitly mapped (both source and target names)
    # This prevents the fallback from lowercasing columns that should stay UPPERCASE
    explicitly_mapped_source_cols = set()  # Source columns (OTPW names) that are in the mapping
    explicitly_mapped_target_cols = set()  # Target columns (CUSTOM names) that should be protected
    renamed_cols = set()  # Columns that were actually renamed
    
    for otpw_col, custom_col in COLUMN_MAPPING.items():
        if otpw_col in result_df.columns:
            explicitly_mapped_source_cols.add(otpw_col)  # Track source column
            explicitly_mapped_target_cols.add(custom_col)  # Track target column (protect it from fallback)
            # Also protect the source column name itself (in case it's the same as target, like DEP_DELAY -> DEP_DELAY)
            if otpw_col == custom_col:
                # If source and target are the same, protect the column from being renamed
                explicitly_mapped_target_cols.add(otpw_col)
            if otpw_col != custom_col:  # Only rename if different
                result_df = result_df.withColumnRenamed(otpw_col, custom_col)
                renamed_cols.add(otpw_col)
        # Note: Don't warn if column doesn't exist - it might be optional
    
    # Step 1b: Convert any remaining unmapped columns to lowercase as fallback
    # This handles any columns not explicitly mapped (but don't touch explicitly mapped ones)
    # NOTE: We'll handle DEP_DELAY case duplication AFTER all case conversions are done
    if verbose:
        print(f"  Debug: Protected target columns: {sorted(explicitly_mapped_target_cols)}")
        print(f"  Debug: Protected source columns: {sorted(explicitly_mapped_source_cols)}")
    
    for col in list(result_df.columns):  # Use list() to avoid modification during iteration
        # Protect columns that are explicitly mapped targets or sources
        # CRITICAL: Check both the current column name and its uppercase version
        # This ensures that if a column is explicitly mapped to stay uppercase (like DEP_DELAY),
        # it won't be converted to lowercase even if the protection check fails
        is_protected = (
            col in explicitly_mapped_target_cols or  # Target of explicit mapping
            col in explicitly_mapped_source_cols or  # Source of explicit mapping
            col.upper() in explicitly_mapped_target_cols or  # Uppercase version is protected
            col.upper() in explicitly_mapped_source_cols  # Uppercase version is source
        )
        
        # Additional check: if the column is a label column (DEP_DELAY, DEP_DEL15, SEVERE_DEL60),
        # always protect it from case conversion
        is_label_column = col in ["DEP_DELAY", "DEP_DEL15", "SEVERE_DEL60"]
        
        if is_protected or is_label_column:
            if verbose and col != col.lower():
                print(f"  Debug: Protecting '{col}' from case conversion (protected={is_protected}, label={is_label_column})")
        
        if not is_protected and not is_label_column and col != col.lower():
            # Only rename if it's not already lowercase and wasn't explicitly mapped
            if col.lower() not in result_df.columns:  # Avoid duplicate column names
                result_df = result_df.withColumnRenamed(col, col.lower())
                if verbose:
                    print(f"  Debug: Converted '{col}' to lowercase (fallback)")
    
    # Step 2: Drop OTPW-specific columns
    columns_to_drop_actual = [col for col in COLUMNS_TO_DROP if col in result_df.columns]
    if columns_to_drop_actual:
        result_df = result_df.drop(*columns_to_drop_actual)
        print(f"✓ Dropped {len(columns_to_drop_actual)} OTPW-specific columns: {columns_to_drop_actual}")
    
    # Step 3: Add CUSTOM-specific columns with default values
    for custom_col, default_value in COLUMNS_TO_ADD.items():
        if custom_col not in result_df.columns:
            if default_value is None:
                result_df = result_df.withColumn(custom_col, F.lit(None).cast("string"))
            else:
                result_df = result_df.withColumn(custom_col, F.lit(default_value))
            print(f"✓ Added missing CUSTOM column '{custom_col}' with default value: {default_value}")
    
    # Step 3a: Derive missing critical columns if possible
    # Handle fl_date: Try multiple methods to derive the flight date
    if "fl_date" not in result_df.columns:
        if verbose:
            print(f"  ⚠ 'fl_date' missing - attempting to derive...")
        
        found_date_col = None
        
        # Method 1: Check for alternative date column names
        date_col_candidates = ["DATE", "date", "FL_DATE", "flight_date", "FlightDate"]
        for candidate in date_col_candidates:
            if candidate in result_df.columns:
                found_date_col = candidate
                if verbose:
                    print(f"    Found date column: '{candidate}' - converting to 'fl_date'")
                # Convert to date type
                result_df = result_df.withColumn("fl_date", F.to_date(F.col(candidate)))
                break
        
        # Method 2: Extract date from scheduled departure datetime columns
        if found_date_col is None:
            datetime_col_candidates = [
                "sched_depart_date_time",
                "sched_depart_date_time_utc",
                "sched_depart_date_time_UTC",
                "SCHED_DEPART_DATE_TIME",
                "scheduled_departure_datetime",
                "departure_datetime"
            ]
            for candidate in datetime_col_candidates:
                if candidate in result_df.columns:
                    found_date_col = candidate
                    if verbose:
                        print(f"    Found scheduled departure datetime: '{candidate}' - extracting date to 'fl_date'")
                    # Extract date from datetime (handles both timestamp and string formats)
                    result_df = result_df.withColumn("fl_date", F.to_date(F.col(candidate)))
                    if verbose:
                        print(f"    ✓ Extracted 'fl_date' from scheduled departure datetime")
                    break
        
        # Method 3: Construct from year/month/day_of_month components
        if found_date_col is None:
            # Check for year column (check both lowercase and uppercase, and also check original columns)
            has_year = "year" in result_df.columns or "YEAR" in result_df.columns
            # Also check original columns in case year exists but wasn't mapped yet
            if not has_year:
                has_year = "year" in original_columns or "YEAR" in original_columns or any("year" in col.lower() for col in original_columns)
                if has_year:
                    # Find the year column name
                    year_col_candidates = [col for col in original_columns if "year" in col.lower()]
                    if year_col_candidates:
                        year_col_orig = year_col_candidates[0]
                        if verbose:
                            print(f"    Found year column in original data: '{year_col_orig}'")
                        # Map it if needed - check if it was already mapped to lowercase
                        if "year" not in result_df.columns and year_col_orig not in result_df.columns:
                            # Need to check if it exists in result_df with different case
                            result_df_cols_lower = {c.lower(): c for c in result_df.columns}
                            if "year" in result_df_cols_lower:
                                # Already exists with different case
                                has_year = True
                            else:
                                # Add it
                                result_df = result_df.withColumn("year", F.col(year_col_orig))
                                has_year = True
                        else:
                            has_year = True
            
            has_month = "month" in result_df.columns or "MONTH" in result_df.columns
            has_day = "day_of_month" in result_df.columns or "DAY_OF_MONTH" in result_df.columns
            
            if has_month and has_day:
                if has_year:
                    if verbose:
                        print(f"    Constructing 'fl_date' from year/month/day_of_month...")
                    year_col = "year" if "year" in result_df.columns else "YEAR"
                    month_col = "month" if "month" in result_df.columns else "MONTH"
                    day_col = "day_of_month" if "day_of_month" in result_df.columns else "DAY_OF_MONTH"
                    
                    # Construct date string and convert to date
                    result_df = result_df.withColumn(
                        "fl_date",
                        F.to_date(
                            F.concat(
                                F.col(year_col).cast("string"),
                                F.lit("-"),
                                F.lpad(F.col(month_col).cast("string"), 2, "0"),
                                F.lit("-"),
                                F.lpad(F.col(day_col).cast("string"), 2, "0")
                            ),
                            "yyyy-MM-dd"
                        )
                    )
                    if verbose:
                        print(f"    ✓ Constructed 'fl_date' from {year_col}/{month_col}/{day_col}")
                else:
                    # Year is missing but we have month/day - use default year 2015 for 60M data
                    # This is a fallback since 60M should be 2015-2019
                    if verbose:
                        print(f"    ⚠ Year column missing - using default year 2015 (60M data should be 2015-2019)")
                        print(f"    ⚠ WARNING: This assumes all flights are in 2015. Verify data if dates look incorrect.")
                    month_col = "month" if "month" in result_df.columns else "MONTH"
                    day_col = "day_of_month" if "day_of_month" in result_df.columns else "DAY_OF_MONTH"
                    
                    # Construct date with default year 2015
                    result_df = result_df.withColumn(
                        "fl_date",
                        F.to_date(
                            F.concat(
                                F.lit("2015-"),
                                F.lpad(F.col(month_col).cast("string"), 2, "0"),
                                F.lit("-"),
                                F.lpad(F.col(day_col).cast("string"), 2, "0")
                            ),
                            "yyyy-MM-dd"
                        )
                    )
                    if verbose:
                        print(f"    ✓ Constructed 'fl_date' from {month_col}/{day_col} with default year 2015")
            else:
                if verbose:
                    print(f"    ❌ Cannot derive 'fl_date': missing date components")
                    print(f"      Has year: {has_year}, Has month: {has_month}, Has day: {has_day}")
                    print(f"      Available columns: {sorted(result_df.columns)[:30]}...")
    
    # Handle op_carrier: Derive from op_unique_carrier if available
    if "op_carrier" not in result_df.columns:
        if "op_unique_carrier" in result_df.columns:
            if verbose:
                print(f"  ⚠ 'op_carrier' missing - deriving from 'op_unique_carrier'...")
            result_df = result_df.withColumn("op_carrier", F.col("op_unique_carrier"))
            if verbose:
                print(f"    ✓ Created 'op_carrier' from 'op_unique_carrier'")
        elif "OP_UNIQUE_CARRIER" in result_df.columns:
            if verbose:
                print(f"  ⚠ 'op_carrier' missing - deriving from 'OP_UNIQUE_CARRIER'...")
            result_df = result_df.withColumn("op_carrier", F.col("OP_UNIQUE_CARRIER"))
            if verbose:
                print(f"    ✓ Created 'op_carrier' from 'OP_UNIQUE_CARRIER'")
        else:
            if verbose:
                print(f"  ⚠ 'op_carrier' missing and cannot derive from op_unique_carrier")
    
    # Step 3b: Handle DEP_DELAY - ensure UPPERCASE version exists
    # NOTE: We only create DEP_DELAY here. The lowercase 'dep_delay' will be added
    # after saving/loading in process_otpw_data.py to avoid Spark ambiguity issues.
    # This breaks the logical plan chain and ensures clean column creation.
    
    # Check what we have after all mappings and case conversions
    current_cols = set(result_df.columns)
    has_upper = "DEP_DELAY" in current_cols
    has_lower = "dep_delay" in current_cols
    
    if verbose:
        print(f"  Debug: After case conversion - has_upper={has_upper}, has_lower={has_lower}")
        if has_lower and not has_upper:
            print(f"  Note: 'dep_delay' exists but 'DEP_DELAY' is missing - will create DEP_DELAY")
    
    # Simple approach: ensure DEP_DELAY (uppercase) exists
    # The lowercase version will be added later in process_otpw_data.py after save/load
    if not has_upper and not has_lower:
        # Neither exists - check for alternatives
        possible_dep_delay_cols = [
            "DEP_DELAY_NEW",
            "dep_delay_new",
            "DEPARTURE_DELAY",
            "departure_delay",
        ]
        
        found_alt = None
        for alt_col in possible_dep_delay_cols:
            if alt_col in result_df.columns:
                found_alt = alt_col
                if verbose:
                    print(f"⚠ Found alternative departure delay column: '{alt_col}'")
                # Create only DEP_DELAY (uppercase) - dep_delay will be added later
                result_df = result_df.withColumn("DEP_DELAY", F.col(alt_col) * 1.0)
                if verbose:
                    print(f"  Created 'DEP_DELAY' from '{alt_col}' (dep_delay will be added after save/load)")
                break
        
        if found_alt is None:
            # List all columns that might be related to delays
            delay_related = [c for c in result_df.columns if "delay" in c.lower() or "DELAY" in c]
            if verbose:
                print(f"⚠ DEP_DELAY/dep_delay not found. Available delay-related columns: {delay_related}")
            raise ValueError(
                f"Critical column 'DEP_DELAY'/'dep_delay' missing after mapping. "
                f"Available delay-related columns: {delay_related}. "
                f"Please check OTPW data or update COLUMN_MAPPING in column_mapping.py"
            )
    else:
        # At least one exists - ensure DEP_DELAY (uppercase) exists
        # The lowercase version will be added later in process_otpw_data.py after save/load
        if has_lower and not has_upper:
            # We have lowercase, create uppercase version
            result_df = result_df.withColumn("DEP_DELAY", F.col("dep_delay") * 1.0)
            if verbose:
                print("✓ Created 'DEP_DELAY' (uppercase) from 'dep_delay' (lowercase) - required by cv.py")
                print("  Note: 'dep_delay' will be re-added after save/load to avoid ambiguity")
        elif has_upper and not has_lower:
            # We have uppercase - perfect! dep_delay will be added later
            # CRITICAL: Ensure DEP_DELAY stays uppercase - drop any lowercase version that might exist
            if "dep_delay" in result_df.columns:
                result_df = result_df.drop("dep_delay")
                if verbose:
                    print("  Dropped 'dep_delay' to avoid ambiguity (will be re-added after save/load)")
            if verbose:
                print("✓ 'DEP_DELAY' (uppercase) exists - 'dep_delay' will be added after save/load")
        else:
            # Both exist - drop dep_delay and keep only DEP_DELAY to avoid ambiguity
            if verbose:
                print("✓ Both 'DEP_DELAY' and 'dep_delay' exist - dropping 'dep_delay' (will be re-added after save/load)")
            result_df = result_df.drop("dep_delay")
    
    # Force schema evaluation to ensure DEP_DELAY is visible
    _ = result_df.schema
    
    # Verify DEP_DELAY exists (we don't need dep_delay here - it will be added later)
    cols_after_creation = set(result_df.columns)
    has_upper_after = "DEP_DELAY" in cols_after_creation
    
    if verbose:
        print(f"  Debug: After mapping - DEP_DELAY: {has_upper_after}")
    
    if not has_upper_after:
        # Last resort: try to create from dep_delay if it exists
        if "dep_delay" in cols_after_creation:
            result_df = result_df.withColumn("DEP_DELAY", F.col("dep_delay") * 1.0)
            if verbose:
                print("  ⚠ Created DEP_DELAY from dep_delay as fallback")
            _ = result_df.schema
    
    # Step 4: Validate that critical columns exist
    # Force schema evaluation to ensure all columns are materialized (including newly created ones)
    # This is critical - we need to force evaluation after creating columns
    schema = result_df.schema
    _ = schema  # Force schema evaluation
    
    # Get final column list after all transformations - force evaluation
    final_cols_list = list(result_df.columns)
    final_cols = set(final_cols_list)
    
    # Double-check DEP_DELAY exists after schema evaluation
    # Note: We don't check for dep_delay here - it will be added after save/load in process_otpw_data.py
    # Sometimes Spark's lazy evaluation means columns aren't visible until schema is evaluated
    if "DEP_DELAY" not in final_cols:
        # Re-check after forcing schema evaluation
        final_cols_list = list(result_df.columns)
        final_cols = set(final_cols_list)
        
        # If still missing, try to create from dep_delay if it exists
        if "dep_delay" in final_cols:
            result_df = result_df.withColumn("DEP_DELAY", F.col("dep_delay") * 1.0)
            if verbose:
                print("⚠ Fixed: Created 'DEP_DELAY' from 'dep_delay' during validation")
            # Force final schema evaluation to ensure column is visible
            _ = result_df.schema
            final_cols = set(result_df.columns)
    
    critical_columns = [
        "fl_date",  # Date column (lowercase)
        "DEP_DELAY",  # Main label (UPPERCASE - matches cv.py FlightDelayEvaluator)
        # Note: dep_delay (lowercase) will be added after save/load in process_otpw_data.py
        "origin",  # Required for joins and features (lowercase)
        "dest",  # Required for joins and features (lowercase)
        "op_carrier",  # Required for joins and features (lowercase)
    ]
    
    # Force one more schema evaluation and column check before validation
    # CRITICAL: Check schema fields directly to avoid lazy evaluation issues
    schema_fields = {f.name for f in result_df.schema.fields}
    final_cols = set(result_df.columns)
    
    # Ensure DEP_DELAY exists - check both columns list and schema fields
    if "DEP_DELAY" not in final_cols and "DEP_DELAY" not in schema_fields:
        # Try to create from dep_delay if it exists
        if "dep_delay" in final_cols or "dep_delay" in schema_fields:
            if verbose:
                print(f"⚠ DEP_DELAY missing in final check, creating from dep_delay...")
            result_df = result_df.withColumn("DEP_DELAY", F.col("dep_delay") * 1.0)
            _ = result_df.schema
            final_cols = set(result_df.columns)
        else:
            # DEP_DELAY is truly missing - this is an error
            if verbose:
                print(f"❌ ERROR: DEP_DELAY is missing and dep_delay is not available to create it from")
    
    missing_critical = [col for col in critical_columns if col not in final_cols]
    
    if missing_critical:
        # Provide helpful debugging info
        if verbose:
            print(f"\n⚠ Available columns after mapping: {sorted(final_cols)[:20]}...")
            print(f"⚠ Missing critical columns: {missing_critical}")
            # Check if DEP_DELAY exists in any case variation
            if "DEP_DELAY" in missing_critical:
                dep_delay_variants = [c for c in final_cols if "dep" in c.lower() and "delay" in c.lower()]
                if dep_delay_variants:
                    print(f"⚠ Found DEP_DELAY variants: {dep_delay_variants}")
        raise ValueError(
            f"Critical columns missing after mapping: {missing_critical}. "
            f"Please update COLUMN_MAPPING in column_mapping.py. "
            f"Available columns: {sorted(final_cols)[:30]}"
        )
    
    # Final verification: Ensure DEP_DELAY exists in the schema before returning
    # Note: dep_delay will be added after save/load in process_otpw_data.py
    # CRITICAL: Force schema evaluation and check both columns list and schema fields
    _ = result_df.schema
    final_cols_check = set(result_df.columns)
    final_schema_cols = {f.name for f in result_df.schema.fields}
    
    # Check if DEP_DELAY exists in either place
    has_dep_delay_in_cols = "DEP_DELAY" in final_cols_check
    has_dep_delay_in_schema = "DEP_DELAY" in final_schema_cols
    
    if not has_dep_delay_in_cols and not has_dep_delay_in_schema:
        # DEP_DELAY is truly missing - try to create from dep_delay as last resort
        has_dep_delay_lower = "dep_delay" in final_cols_check or "dep_delay" in final_schema_cols
        if has_dep_delay_lower:
            if verbose:
                print("⚠ CRITICAL: DEP_DELAY missing before return, creating from dep_delay...")
            result_df = result_df.withColumn("DEP_DELAY", F.col("dep_delay") * 1.0)
            _ = result_df.schema
            # Verify it was created
            final_check = set(result_df.columns)
            if "DEP_DELAY" not in final_check:
                raise ValueError("Failed to create DEP_DELAY from dep_delay. Cannot proceed.")
            if verbose:
                print("  ✓ DEP_DELAY created successfully")
        else:
            raise ValueError(
                "Critical column 'DEP_DELAY' missing after mapping and 'dep_delay' not available. "
                "Cannot proceed. Available delay columns: " + 
                str([c for c in final_cols_check if "delay" in c.lower()])
            )
    elif verbose:
        print(f"  ✓ Final check: DEP_DELAY exists (in cols: {has_dep_delay_in_cols}, in schema: {has_dep_delay_in_schema})")
    
    return result_df


# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate_mapping(df_otpw: DataFrame, df_custom: DataFrame) -> dict:
    """
    Validate that the mapping produces a DataFrame with matching schema.
    
    Args:
        df_otpw: Original OTPW DataFrame
        df_custom: Reference CUSTOM DataFrame (for comparison)
        
    Returns:
        dict: Validation results with:
            - "mapped_columns": list of columns in mapped DataFrame
            - "custom_columns": list of columns in CUSTOM DataFrame
            - "missing_in_mapped": columns in CUSTOM but not in mapped
            - "extra_in_mapped": columns in mapped but not in CUSTOM
            - "is_valid": bool indicating if schemas match
    """
    df_mapped = map_otpw_columns_to_custom(df_otpw)
    
    mapped_cols = set(df_mapped.columns)
    custom_cols = set(df_custom.columns)
    
    missing = custom_cols - mapped_cols
    extra = mapped_cols - custom_cols
    
    return {
        "mapped_columns": sorted(mapped_cols),
        "custom_columns": sorted(custom_cols),
        "missing_in_mapped": sorted(missing),
        "extra_in_mapped": sorted(extra),
        "is_valid": len(missing) == 0 and len(extra) == 0,
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage (run in Databricks notebook or PySpark):
    
    from pyspark.sql import SparkSession
    from column_mapping import map_otpw_columns_to_custom, validate_mapping
    
    spark = SparkSession.builder.getOrCreate()
    
    # Load OTPW data
    df_otpw = spark.read.parquet("dbfs:/path/to/otpw/data.parquet")
    
    # Map columns
    df_mapped = map_otpw_columns_to_custom(df_otpw)
    
    # Validate (optional - compare with CUSTOM reference)
    df_custom_ref = spark.read.parquet("dbfs:/path/to/custom/data.parquet")
    validation = validate_mapping(df_otpw, df_custom_ref)
    
    print(f"Mapping valid: {validation['is_valid']}")
    if not validation['is_valid']:
        print(f"Missing columns: {validation['missing_in_mapped']}")
        print(f"Extra columns: {validation['extra_in_mapped']}")
    """
    print("This module provides column mapping functions.")
    print("Import and use map_otpw_columns_to_custom() to transform OTPW DataFrames.")
    print("\nSee docstring for usage examples.")

