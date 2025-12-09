"""
time_series_features_spark_native.py - Pure Spark DataFrame implementation for time-series features

NO Prophet, NO Pandas UDFs - just Spark SQL/DataFrame operations!

Features computed:
- Trend: Moving average or linear trend over time
- Weekly seasonality: Average by day of week
- Yearly seasonality: Average by day of year or month
- Forecast: Trend + seasonality components
- is_holiday: Binary indicator for US federal holidays (global, carrier, and airport level)
  - New Year's Day, MLK Day, Presidents' Day, Memorial Day, Independence Day,
    Labor Day, Columbus Day, Veterans Day, Thanksgiving, Christmas

This approach is MUCH faster than Prophet because:
1. All operations are native Spark (no Python serialization overhead)
2. Leverages Spark's distributed computing fully
3. No model fitting required - just aggregations and window functions
"""

from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.functions import col, to_timestamp, broadcast, dayofweek, dayofyear, month, year, when, lit
from pyspark.ml.base import Estimator, Model
from datetime import datetime


class TimeSeriesFeaturesModel(Model):
    """Model that stores pre-computed feature DataFrames and seasonality patterns"""
    
    def __init__(self, 
                 global_features_df=None,
                 carrier_features_df=None,
                 airport_features_df=None,
                 global_seasonality_patterns=None,
                 carrier_seasonality_patterns=None,
                 airport_seasonality_patterns=None,
                 date_col="FL_DATE",
                 carrier_col="op_carrier",
                 origin_col="origin",
                 trend_window_days=30):
        super(TimeSeriesFeaturesModel, self).__init__()
        self.global_features_df = global_features_df
        self.carrier_features_df = carrier_features_df
        self.airport_features_df = airport_features_df
        self.global_seasonality_patterns = global_seasonality_patterns  # For extrapolation
        self.carrier_seasonality_patterns = carrier_seasonality_patterns
        self.airport_seasonality_patterns = airport_seasonality_patterns
        self.date_col = date_col
        self.carrier_col = carrier_col
        self.origin_col = origin_col
        self.trend_window_days = trend_window_days
        self._spark = SparkSession.builder.getOrCreate()
    
    def _transform(self, df):
        """Join pre-computed features"""
        # Prepare date column for joining
        df_with_features = df.withColumn(
            "date_str_join",
            F.when(
                col(self.date_col).rlike("^\\d{4}-\\d{2}-\\d{2}$"),
                col(self.date_col)
            ).otherwise(
                F.date_format(to_timestamp(col(self.date_col), "yyyy-MM-dd"), "yyyy-MM-dd")
            )
        )
        
        # Add direct is_holiday feature (not grouped)
        df_with_features = df_with_features.withColumn(
            "date_for_holiday",
            F.when(
                col(self.date_col).rlike("^\\d{4}-\\d{2}-\\d{2}$"),
                F.to_date(col(self.date_col))
            ).otherwise(
                to_timestamp(col(self.date_col), "yyyy-MM-dd").cast("date")
            )
        )
        # Compute is_holiday directly using Spark SQL functions
        month_val = month("date_for_holiday")
        day_of_month = F.dayofmonth("date_for_holiday")
        day_of_week = dayofweek("date_for_holiday")
        week_of_month = ((day_of_month - 1) / 7).cast("int") + 1
        
        is_holiday = (
            ((month_val == 1) & (day_of_month == 1)) |  # New Year's
            ((month_val == 1) & (day_of_week == 2) & (week_of_month == 3)) |  # MLK Day
            ((month_val == 2) & (day_of_week == 2) & (week_of_month == 3)) |  # Presidents' Day
            ((month_val == 5) & (day_of_week == 2) & (day_of_month >= 25)) |  # Memorial Day
            ((month_val == 7) & (day_of_month == 4)) |  # Independence Day
            ((month_val == 9) & (day_of_week == 2) & (week_of_month == 1)) |  # Labor Day
            ((month_val == 10) & (day_of_week == 2) & (week_of_month == 2)) |  # Columbus Day
            ((month_val == 11) & (day_of_month == 11)) |  # Veterans Day
            ((month_val == 11) & (day_of_week == 5) & (week_of_month == 4)) |  # Thanksgiving
            ((month_val == 12) & (day_of_month == 25))  # Christmas
        )
        
        df_with_features = df_with_features.withColumn(
            "is_holiday",
            when(is_holiday, 1).otherwise(0).cast("int")
        ).drop("date_for_holiday")
        
        # Prepare date column for feature generation
        # Use temporary column names to avoid overwriting existing day_of_week/month columns
        df_with_features = df_with_features.withColumn(
            "date_for_features",
            F.when(
                col(self.date_col).rlike("^\\d{4}-\\d{2}-\\d{2}$"),
                F.to_date(col(self.date_col))
            ).otherwise(
                to_timestamp(col(self.date_col), "yyyy-MM-dd").cast("date")
            )
        )
        df_with_features = df_with_features.withColumn("_temp_day_of_week", dayofweek("date_for_features"))
        df_with_features = df_with_features.withColumn("_temp_month", month("date_for_features"))
        
        # Join global features
        if self.global_features_df is not None:
            # Extract seasonality patterns from pre-computed features for extrapolation
            global_seasonality_stats = self.global_features_df.agg(
                F.avg("ts_weekly_seasonality_global").alias("avg_weekly_seasonality"),
                F.avg("ts_yearly_seasonality_global").alias("avg_yearly_seasonality"),
                F.max("ts_trend_global").alias("last_trend")  # Use last trend for extrapolation
            ).first()
            
            # Select only feature columns (exclude date_str join key) to avoid ambiguity
            global_features_for_join = self.global_features_df.select(
                col("date_str").alias("global_date_str"),
                col("ts_trend_global"),
                col("ts_weekly_seasonality_global"),
                col("ts_yearly_seasonality_global"),
                col("ts_forecast_dep_delay_global"),
                col("is_holiday_global")
            )
            
            df_with_features = df_with_features.join(
                broadcast(global_features_for_join),
                col("date_str_join") == col("global_date_str"),
                "left"
            ).drop("global_date_str")
            
            # For missing dates, use seasonality patterns and last known trend
            last_trend = global_seasonality_stats["last_trend"] if global_seasonality_stats and global_seasonality_stats["last_trend"] is not None else 0.0
            avg_weekly = global_seasonality_stats["avg_weekly_seasonality"] if global_seasonality_stats and global_seasonality_stats["avg_weekly_seasonality"] is not None else 0.0
            avg_yearly = global_seasonality_stats["avg_yearly_seasonality"] if global_seasonality_stats and global_seasonality_stats["avg_yearly_seasonality"] is not None else 0.0
            
            df_with_features = df_with_features.withColumn(
                "ts_trend_global",
                F.coalesce(col("ts_trend_global"), F.lit(last_trend))
            )
            df_with_features = df_with_features.withColumn(
                "ts_weekly_seasonality_global",
                F.coalesce(col("ts_weekly_seasonality_global"), F.lit(avg_weekly))
            )
            df_with_features = df_with_features.withColumn(
                "ts_yearly_seasonality_global",
                F.coalesce(col("ts_yearly_seasonality_global"), F.lit(avg_yearly))
            )
            df_with_features = df_with_features.withColumn(
                "ts_forecast_dep_delay_global",
                F.coalesce(
                    col("ts_forecast_dep_delay_global"),
                    col("ts_trend_global") + col("ts_weekly_seasonality_global") + F.coalesce(col("ts_yearly_seasonality_global"), F.lit(0.0))
                )
            )
            df_with_features = df_with_features.withColumn(
                "is_holiday_global",
                F.coalesce(col("is_holiday_global"), col("is_holiday"))
            )
        
        # Join carrier features (similar logic)
        if self.carrier_features_df is not None:
            # Select only feature columns (exclude join keys) to avoid ambiguity
            carrier_features_for_join = self.carrier_features_df.select(
                col("carrier").alias("carrier_join_key"),
                col("date_str").alias("carrier_date_str"),
                col("ts_trend_carrier"),
                col("ts_weekly_seasonality_carrier"),
                col("ts_yearly_seasonality_carrier"),
                col("ts_forecast_dep_delay_carrier"),
                col("is_holiday_carrier")
            )
            
            df_with_features = df_with_features.join(
                broadcast(carrier_features_for_join),
                (col(self.carrier_col) == col("carrier_join_key")) & 
                (col("date_str_join") == col("carrier_date_str")),
                "left"
            ).drop("carrier_join_key", "carrier_date_str")
            # Fill missing carrier features with defaults (carrier-specific patterns would be better)
            df_with_features = df_with_features.withColumn(
                "ts_trend_carrier",
                F.coalesce(col("ts_trend_carrier"), col("ts_trend_global"))
            )
            df_with_features = df_with_features.withColumn(
                "ts_weekly_seasonality_carrier",
                F.coalesce(col("ts_weekly_seasonality_carrier"), col("ts_weekly_seasonality_global"))
            )
            df_with_features = df_with_features.withColumn(
                "ts_yearly_seasonality_carrier",
                F.coalesce(col("ts_yearly_seasonality_carrier"), col("ts_yearly_seasonality_global"))
            )
            df_with_features = df_with_features.withColumn(
                "ts_forecast_dep_delay_carrier",
                F.coalesce(
                    col("ts_forecast_dep_delay_carrier"),
                    col("ts_trend_carrier") + col("ts_weekly_seasonality_carrier") + F.coalesce(col("ts_yearly_seasonality_carrier"), F.lit(0.0))
                )
            )
            df_with_features = df_with_features.withColumn(
                "is_holiday_carrier",
                F.coalesce(col("is_holiday_carrier"), col("is_holiday"))
            )
        
        # Join airport features (similar logic)
        if self.airport_features_df is not None:
            # Select only feature columns (exclude join keys) to avoid ambiguity
            airport_features_for_join = self.airport_features_df.select(
                col("origin").alias("airport_join_key"),
                col("date_str").alias("airport_date_str"),
                col("ts_trend_airport"),
                col("ts_weekly_seasonality_airport"),
                col("ts_yearly_seasonality_airport"),
                col("ts_forecast_dep_delay_airport"),
                col("is_holiday_airport")
            )
            
            df_with_features = df_with_features.join(
                broadcast(airport_features_for_join),
                (col(self.origin_col) == col("airport_join_key")) & 
                (col("date_str_join") == col("airport_date_str")),
                "left"
            ).drop("airport_join_key", "airport_date_str")
            # Fill missing airport features with defaults
            df_with_features = df_with_features.withColumn(
                "ts_trend_airport",
                F.coalesce(col("ts_trend_airport"), col("ts_trend_global"))
            )
            df_with_features = df_with_features.withColumn(
                "ts_weekly_seasonality_airport",
                F.coalesce(col("ts_weekly_seasonality_airport"), col("ts_weekly_seasonality_global"))
            )
            df_with_features = df_with_features.withColumn(
                "ts_yearly_seasonality_airport",
                F.coalesce(col("ts_yearly_seasonality_airport"), col("ts_yearly_seasonality_global"))
            )
            df_with_features = df_with_features.withColumn(
                "ts_forecast_dep_delay_airport",
                F.coalesce(
                    col("ts_forecast_dep_delay_airport"),
                    col("ts_trend_airport") + col("ts_weekly_seasonality_airport") + F.coalesce(col("ts_yearly_seasonality_airport"), F.lit(0.0))
                )
            )
            df_with_features = df_with_features.withColumn(
                "is_holiday_airport",
                F.coalesce(col("is_holiday_airport"), col("is_holiday"))
            )
        
        # Ensure all expected columns exist
        expected_ts_cols = [
            'ts_forecast_dep_delay_global', 'ts_trend_global', 
            'ts_weekly_seasonality_global', 'ts_yearly_seasonality_global',
            'ts_forecast_dep_delay_carrier', 'ts_trend_carrier',
            'ts_weekly_seasonality_carrier', 'ts_yearly_seasonality_carrier',
            'ts_forecast_dep_delay_airport', 'ts_trend_airport',
            'ts_weekly_seasonality_airport', 'ts_yearly_seasonality_airport',
            'is_holiday', 'is_holiday_global', 'is_holiday_carrier', 'is_holiday_airport'
        ]
        
        for col_name in expected_ts_cols:
            if col_name not in df_with_features.columns:
                if 'is_holiday' in col_name:
                    df_with_features = df_with_features.withColumn(col_name, F.lit(0).cast('int'))
                else:
                    df_with_features = df_with_features.withColumn(col_name, F.lit(0.0).cast('double'))
        
        # Fill any remaining NULL values with 0
        ts_cols = [c for c in df_with_features.columns if c.startswith('ts_')]
        holiday_cols = [c for c in df_with_features.columns if 'is_holiday' in c.lower()]
        fill_dict = {}
        if ts_cols:
            fill_dict.update({col_name: 0.0 for col_name in ts_cols})
        if holiday_cols:
            fill_dict.update({col_name: 0 for col_name in holiday_cols})
        if fill_dict:
            df_with_features = df_with_features.fillna(fill_dict)
        
        # Drop temporary columns (use temp names to avoid dropping original day_of_week/month if they exist)
        df_with_features = df_with_features.drop("date_for_features", "_temp_day_of_week", "_temp_month", "date_str_join")
        
        return df_with_features


class TimeSeriesFeaturesEstimator(Estimator):
    """
    Pure Spark DataFrame implementation - NO Prophet, NO Pandas!
    
    Computes time-series features using Spark SQL window functions:
    - Trend: Moving average over time window (default: 30 days)
    - Weekly seasonality: Average by day of week, adjusted by overall mean
    - Yearly seasonality: Average by month, adjusted by overall mean
    - Forecast: Trend + weekly_seasonality + yearly_seasonality
    - is_holiday: US federal holiday indicator (computed using Spark SQL date functions)
    
    Features are generated at three levels:
    - Global: ts_trend_global, ts_weekly_seasonality_global, ts_yearly_seasonality_global, 
              ts_forecast_dep_delay_global, is_holiday_global
    - Carrier: ts_trend_carrier, ts_weekly_seasonality_carrier, ts_yearly_seasonality_carrier,
               ts_forecast_dep_delay_carrier, is_holiday_carrier
    - Airport: ts_trend_airport, ts_weekly_seasonality_airport, ts_yearly_seasonality_airport,
               ts_forecast_dep_delay_airport, is_holiday_airport
    - Direct: is_holiday (binary indicator for US federal holidays)
    
    Note: This is NOT Prophet - it uses simple moving averages and window functions.
    Much faster (10-20x) but less sophisticated than Prophet's Bayesian approach.
    """
    
    def __init__(self,
                 date_col="FL_DATE",
                 carrier_col="op_carrier",
                 origin_col="origin",
                 delay_col="DEP_DELAY",
                 trend_window_days=30,  # Moving average window for trend
                 min_days_required=14):
        super(TimeSeriesFeaturesEstimator, self).__init__()
        self.date_col = date_col
        self.carrier_col = carrier_col
        self.origin_col = origin_col
        self.delay_col = delay_col
        self.trend_window_days = trend_window_days
        self.min_days_required = min_days_required
        self._spark = SparkSession.builder.getOrCreate()
    
    def _is_us_holiday(self, date_col):
        """
        Determine if a date is a US federal holiday using Spark SQL functions.
        
        US Federal Holidays:
        - New Year's Day: January 1
        - Martin Luther King Jr. Day: Third Monday in January
        - Presidents' Day: Third Monday in February
        - Memorial Day: Last Monday in May
        - Independence Day: July 4
        - Labor Day: First Monday in September
        - Columbus Day: Second Monday in October
        - Veterans Day: November 11
        - Thanksgiving: Fourth Thursday in November
        - Christmas: December 25
        """
        month_val = month(date_col)
        day_of_month = F.dayofmonth(date_col)
        day_of_week = dayofweek(date_col)  # Sunday=1, Monday=2, ..., Saturday=7
        
        # Helper: Calculate which occurrence of weekday in month (1st, 2nd, 3rd, 4th, or last)
        # For a given day_of_month and day_of_week, calculate week number
        # Formula: floor((day_of_month - 1) / 7) + 1 gives week number (1-5)
        week_of_month = ((day_of_month - 1) / 7).cast("int") + 1
        
        # New Year's Day: January 1
        new_years = (month_val == 1) & (day_of_month == 1)
        
        # Martin Luther King Jr. Day: Third Monday in January (Monday = 2)
        mlk_day = (month_val == 1) & (day_of_week == 2) & (week_of_month == 3)
        
        # Presidents' Day: Third Monday in February
        presidents_day = (month_val == 2) & (day_of_week == 2) & (week_of_month == 3)
        
        # Memorial Day: Last Monday in May
        # Last Monday = Monday in last week of May (day_of_month >= 25)
        memorial_day = (month_val == 5) & (day_of_week == 2) & (day_of_month >= 25)
        
        # Independence Day: July 4
        independence_day = (month_val == 7) & (day_of_month == 4)
        
        # Labor Day: First Monday in September
        labor_day = (month_val == 9) & (day_of_week == 2) & (week_of_month == 1)
        
        # Columbus Day: Second Monday in October
        columbus_day = (month_val == 10) & (day_of_week == 2) & (week_of_month == 2)
        
        # Veterans Day: November 11
        veterans_day = (month_val == 11) & (day_of_month == 11)
        
        # Thanksgiving: Fourth Thursday in November (Thursday = 5)
        thanksgiving = (month_val == 11) & (day_of_week == 5) & (week_of_month == 4)
        
        # Christmas: December 25
        christmas = (month_val == 12) & (day_of_month == 25)
        
        # Combine all holidays
        is_holiday = (
            new_years | mlk_day | presidents_day | memorial_day | 
            independence_day | labor_day | columbus_day | veterans_day | 
            thanksgiving | christmas
        )
        
        return when(is_holiday, 1).otherwise(0).cast("int")
    
    def _compute_trend_seasonality_features(self, df_agg, group_cols=None):
        """
        Compute trend and seasonality features using pure Spark operations.
        
        Args:
            df_agg: Spark DataFrame with columns: date, avg_dep_delay, and optionally group_cols
            group_cols: List of columns to group by (e.g., ['carrier'] or ['origin'])
        
        Returns:
            DataFrame with features: date_str, trend, weekly_seasonality, yearly_seasonality, forecast, is_holiday
        """
        # Prepare date column
        df = df_agg.withColumn("date", F.to_date(col("date")))
        
        # Add time-based features
        df = df.withColumn("day_of_week", dayofweek("date"))
        df = df.withColumn("day_of_year", dayofyear("date"))
        df = df.withColumn("month", month("date"))
        df = df.withColumn("year", year("date"))
        df = df.withColumn("date_str", F.date_format("date", "yyyy-MM-dd"))
        
        # Add holiday indicator
        df = df.withColumn("is_holiday", self._is_us_holiday(col("date")))
        
        # Define window for trend (moving average)
        if group_cols:
            trend_window = Window.partitionBy(*group_cols).orderBy("date").rowsBetween(
                -self.trend_window_days, 0
            )
        else:
            trend_window = Window.orderBy("date").rowsBetween(
                -self.trend_window_days, 0
            )
        
        # Compute trend as moving average
        df = df.withColumn(
            "trend",
            F.avg("avg_dep_delay").over(trend_window)
        )
        
        # Compute overall mean for seasonality adjustment
        if group_cols:
            overall_mean_window = Window.partitionBy(*group_cols).rowsBetween(
                Window.unboundedPreceding, Window.unboundedFollowing
            )
        else:
            overall_mean_window = Window.rowsBetween(
                Window.unboundedPreceding, Window.unboundedFollowing
            )
        
        df = df.withColumn("overall_mean", F.avg("avg_dep_delay").over(overall_mean_window))
        
        # Compute weekly seasonality: average by day of week, adjusted by overall mean
        if group_cols:
            weekly_window = Window.partitionBy(*group_cols, "day_of_week")
        else:
            weekly_window = Window.partitionBy("day_of_week")
        
        df = df.withColumn(
            "weekly_seasonality",
            F.avg("avg_dep_delay").over(weekly_window) - col("overall_mean")
        )
        
        # Compute yearly seasonality: average by month, adjusted by overall mean
        if group_cols:
            yearly_window = Window.partitionBy(*group_cols, "month")
        else:
            yearly_window = Window.partitionBy("month")
        
        df = df.withColumn(
            "yearly_seasonality",
            F.avg("avg_dep_delay").over(yearly_window) - col("overall_mean")
        )
        
        # Drop temporary column
        df = df.drop("overall_mean")
        
        # Forecast = trend + seasonality components
        df = df.withColumn(
            "forecast",
            col("trend") + col("weekly_seasonality") + F.coalesce(col("yearly_seasonality"), F.lit(0.0))
        )
        
        # Select and return feature columns
        feature_cols = ["date_str", "trend", "weekly_seasonality", "yearly_seasonality", "forecast", "is_holiday"]
        if group_cols:
            feature_cols = group_cols + feature_cols
        
        return df.select(*feature_cols)
    
    def _fit(self, df):
        """Generate time-series features using pure Spark operations"""
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Generating time-series features (pure Spark, no Prophet)...")
        
        # Prepare date column
        df_prep = df.withColumn(
            "date",
            to_timestamp(col(self.date_col), "yyyy-MM-dd").cast("date")
        ).filter(
            col("date").isNotNull() & 
            col(self.delay_col).isNotNull()
        )
        
        global_features_df = None
        carrier_features_df = None
        airport_features_df = None
        
        # 1. Global time-series aggregation
        print("  Generating global time-series...")
        global_ts = (
            df_prep
            .groupBy("date")
            .agg(
                F.avg(self.delay_col).alias("avg_dep_delay"),
                F.count("*").alias("flight_count")
            )
            .orderBy("date")
        )
        
        # Check if we have enough data
        global_count = global_ts.count()
        if global_count >= self.min_days_required:
            global_features = self._compute_trend_seasonality_features(global_ts, group_cols=None)
            
            # Rename columns with clear naming (no "prophet" prefix)
            global_features_df = global_features.select(
                col("date_str"),
                col("trend").alias("ts_trend_global"),
                col("weekly_seasonality").alias("ts_weekly_seasonality_global"),
                col("yearly_seasonality").alias("ts_yearly_seasonality_global"),
                col("forecast").alias("ts_forecast_dep_delay_global"),
                col("is_holiday").alias("is_holiday_global")
            )
            print(f"    ✓ Generated global features for {global_count} dates")
        
        # 2. Carrier time-series aggregation
        if self.carrier_col in df_prep.columns:
            print("  Generating carrier time-series...")
            carrier_ts = (
                df_prep
                .filter(col(self.carrier_col).isNotNull())
                .groupBy(self.carrier_col, "date")
                .agg(
                    F.avg(self.delay_col).alias("avg_dep_delay"),
                    F.count("*").alias("flight_count")
                )
                .orderBy(self.carrier_col, "date")
            )
            
            # Filter carriers with sufficient data
            carrier_day_counts = (
                carrier_ts
                .groupBy(self.carrier_col)
                .agg(F.count("date").alias("day_count"))
                .filter(col("day_count") >= self.min_days_required)
            )
            
            carrier_ts_filtered = carrier_ts.join(
                broadcast(carrier_day_counts.select(self.carrier_col)),
                self.carrier_col,
                "inner"
            )
            
            carrier_count = carrier_ts_filtered.select(self.carrier_col).distinct().count()
            if carrier_count > 0:
                carrier_features = self._compute_trend_seasonality_features(
                    carrier_ts_filtered, 
                    group_cols=[self.carrier_col]
                )
                
                # Rename columns with clear naming
                carrier_features_df = carrier_features.select(
                    col(self.carrier_col).alias("carrier"),
                    col("date_str"),
                    col("trend").alias("ts_trend_carrier"),
                    col("weekly_seasonality").alias("ts_weekly_seasonality_carrier"),
                    col("yearly_seasonality").alias("ts_yearly_seasonality_carrier"),
                    col("forecast").alias("ts_forecast_dep_delay_carrier"),
                    col("is_holiday").alias("is_holiday_carrier")
                )
                print(f"    ✓ Generated carrier features for {carrier_count} carriers")
        
        # 3. Airport time-series aggregation
        if self.origin_col in df_prep.columns:
            print("  Generating airport time-series...")
            airport_ts = (
                df_prep
                .filter(col(self.origin_col).isNotNull())
                .groupBy(self.origin_col, "date")
                .agg(
                    F.avg(self.delay_col).alias("avg_dep_delay"),
                    F.count("*").alias("flight_count")
                )
                .orderBy(self.origin_col, "date")
            )
            
            # Filter airports with sufficient data and limit to top N
            airport_day_counts = (
                airport_ts
                .groupBy(self.origin_col)
                .agg(F.count("date").alias("day_count"))
                .filter(col("day_count") >= self.min_days_required)
                .orderBy(col("day_count").desc())
                .limit(100)  # Top 100 airports
            )
            
            airport_ts_filtered = airport_ts.join(
                broadcast(airport_day_counts.select(self.origin_col)),
                self.origin_col,
                "inner"
            )
            
            airport_count = airport_ts_filtered.select(self.origin_col).distinct().count()
            if airport_count > 0:
                airport_features = self._compute_trend_seasonality_features(
                    airport_ts_filtered,
                    group_cols=[self.origin_col]
                )
                
                # Rename columns with clear naming
                airport_features_df = airport_features.select(
                    col(self.origin_col).alias("origin"),
                    col("date_str"),
                    col("trend").alias("ts_trend_airport"),
                    col("weekly_seasonality").alias("ts_weekly_seasonality_airport"),
                    col("yearly_seasonality").alias("ts_yearly_seasonality_airport"),
                    col("forecast").alias("ts_forecast_dep_delay_airport"),
                    col("is_holiday").alias("is_holiday_airport")
                )
                print(f"    ✓ Generated airport features for {airport_count} airports")
        
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ✓ Time-series feature generation complete! (took {duration})")
        
        return TimeSeriesFeaturesModel(
            global_features_df=global_features_df,
            carrier_features_df=carrier_features_df,
            airport_features_df=airport_features_df,
            global_seasonality_patterns=None,  # Could extract patterns here for better extrapolation
            carrier_seasonality_patterns=None,
            airport_seasonality_patterns=None,
            date_col=self.date_col,
            carrier_col=self.carrier_col,
            origin_col=self.origin_col,
            trend_window_days=self.trend_window_days
        )

