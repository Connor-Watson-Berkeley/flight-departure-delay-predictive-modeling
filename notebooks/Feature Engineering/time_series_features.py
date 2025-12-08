"""
time_series_features.py - Time-series based feature engineering for flight delay prediction

Provides TimeSeriesFeaturesEstimator for use in Spark ML pipelines.
Generates Prophet-based time-series features (global, carrier, airport level) that capture
temporal trends and seasonality.
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, to_timestamp, when, broadcast
from pyspark.ml.base import Estimator, Model
import pandas as pd
from datetime import datetime

# Suppress verbose cmdstanpy and prophet output BEFORE importing Prophet
import logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

from prophet import Prophet


class TimeSeriesFeaturesModel(Model):
    """Model returned by TimeSeriesFeaturesEstimator after fitting"""
    
    def __init__(self, 
                 global_model=None,
                 carrier_models=None,
                 airport_models=None,
                 global_training_dates=None,
                 carrier_training_dates=None,
                 airport_training_dates=None,
                 date_col="FL_DATE",
                 carrier_col="op_carrier",
                 origin_col="origin",
                 min_days_required=14):
        super(TimeSeriesFeaturesModel, self).__init__()
        self.global_model = global_model  # Fitted Prophet model
        self.carrier_models = carrier_models  # Dict: {carrier: fitted Prophet model}
        self.airport_models = airport_models  # Dict: {airport: fitted Prophet model}
        self.global_training_dates = global_training_dates  # pd.Series of training dates
        self.carrier_training_dates = carrier_training_dates  # Dict: {carrier: pd.Series}
        self.airport_training_dates = airport_training_dates  # Dict: {airport: pd.Series}
        self.date_col = date_col
        self.carrier_col = carrier_col
        self.origin_col = origin_col
        self.min_days_required = min_days_required
        self._spark = SparkSession.builder.getOrCreate()
    
    def _generate_forecasts_for_dates(self, model, training_dates, target_dates):
        """Generate Prophet forecasts for target dates using a fitted model"""
        if model is None:
            return None
        
        # Combine training and target dates, get unique sorted dates
        if training_dates is not None and len(training_dates) > 0:
            all_dates = pd.concat([training_dates, target_dates]).drop_duplicates().sort_values()
        else:
            all_dates = target_dates.drop_duplicates().sort_values()
        
        # Generate forecast for all dates
        future_df = pd.DataFrame({'ds': all_dates})
        forecast = model.predict(future_df)
        
        # Extract features
        feature_cols = ['ds', 'trend', 'weekly', 'yhat', 'yhat_lower', 'yhat_upper']
        if 'yearly' in forecast.columns:
            feature_cols.insert(2, 'yearly')
        
        features = forecast[feature_cols].copy()
        features['date_str'] = features['ds'].dt.strftime('%Y-%m-%d')
        
        return features
    
    def _transform(self, df):
        """Generate Prophet forecasts for dates in input DataFrame and join features"""
        if self.global_model is None and (self.carrier_models is None or len(self.carrier_models) == 0) and (self.airport_models is None or len(self.airport_models) == 0):
            raise ValueError("Model must be fitted before transform()")
        
        # Prepare date column and get unique dates from input DataFrame
        # Ensure date column is in date format for extraction, but keep original for joining
        df_prep = df.withColumn("date_temp", to_timestamp(col(self.date_col), "yyyy-MM-dd").cast("date"))
        unique_dates = df_prep.select("date_temp").distinct().orderBy("date_temp").toPandas()['date_temp']
        unique_dates = pd.to_datetime(unique_dates)
        
        # Ensure date column is formatted as string for joining (yyyy-MM-dd format)
        df_with_features = df.withColumn(
            "date_str_join",
            F.when(
                col(self.date_col).rlike("^\\d{4}-\\d{2}-\\d{2}$"),
                col(self.date_col)
            ).otherwise(
                F.date_format(to_timestamp(col(self.date_col), "yyyy-MM-dd"), "yyyy-MM-dd")
            )
        )
        
        # Generate and join global features
        if self.global_model is not None:
            global_features = self._generate_forecasts_for_dates(
                self.global_model, 
                self.global_training_dates, 
                unique_dates
            )
            
            if global_features is not None:
                # Rename columns
                global_features = global_features.rename(columns={
                    'trend': 'prophet_trend_global',
                    'weekly': 'prophet_weekly_seasonality_global',
                    'yhat': 'prophet_forecast_dep_delay_global',
                    'yhat_lower': 'prophet_forecast_lower_global',
                    'yhat_upper': 'prophet_forecast_upper_global'
                })
                if 'yearly' in global_features.columns:
                    global_features = global_features.rename(columns={'yearly': 'prophet_yearly_seasonality_global'})
                
                # Convert to Spark DataFrame and join
                global_spark = self._spark.createDataFrame(global_features)
                global_feature_cols = [c for c in global_spark.columns if c not in ['ds', 'date_str']]
                global_spark_features = global_spark.select(
                    [col("date_str").alias("global_date_str")] + 
                    [col(c) for c in global_feature_cols]
                )
                
                df_with_features = df_with_features.join(
                    broadcast(global_spark_features),
                    col("date_str_join") == col("global_date_str"),
                    "left"
                ).drop("global_date_str")
        
        # Generate and join carrier features
        if self.carrier_models is not None and len(self.carrier_models) > 0:
            # Get unique carriers from input DataFrame
            unique_carriers = df_with_features.select(self.carrier_col).distinct().filter(col(self.carrier_col).isNotNull()).toPandas()[self.carrier_col].unique()
            
            carrier_features_list = []
            for carrier in unique_carriers:
                if carrier in self.carrier_models:
                    training_dates = None
                    if self.carrier_training_dates is not None:
                        training_dates = self.carrier_training_dates.get(carrier, pd.Series(dtype='datetime64[ns]'))
                    carrier_features = self._generate_forecasts_for_dates(
                        self.carrier_models[carrier],
                        training_dates,
                        unique_dates
                    )
                    
                    if carrier_features is not None:
                        carrier_features = carrier_features.rename(columns={
                            'trend': 'prophet_trend_carrier',
                            'weekly': 'prophet_weekly_seasonality_carrier',
                            'yhat': 'prophet_forecast_dep_delay_carrier',
                            'yhat_lower': 'prophet_forecast_lower_carrier',
                            'yhat_upper': 'prophet_forecast_upper_carrier'
                        })
                        if 'yearly' in carrier_features.columns:
                            carrier_features = carrier_features.rename(columns={'yearly': 'prophet_yearly_seasonality_carrier'})
                        carrier_features['carrier'] = carrier
                        carrier_features_list.append(carrier_features)
            
            if carrier_features_list:
                carrier_features_df = pd.concat(carrier_features_list, ignore_index=True)
                carrier_spark = self._spark.createDataFrame(carrier_features_df)
                carrier_feature_cols = [c for c in carrier_spark.columns if c not in ['ds', 'date_str', 'carrier']]
                carrier_spark_features = carrier_spark.select(
                    [col("carrier").alias("carrier_join_key"),
                     col("date_str").alias("carrier_date_str")] + 
                    [col(c) for c in carrier_feature_cols]
                )
                
                df_with_features = df_with_features.join(
                    broadcast(carrier_spark_features),
                    (col(self.carrier_col) == col("carrier_join_key")) & 
                    (col("date_str_join") == col("carrier_date_str")),
                    "left"
                ).drop("carrier_join_key", "carrier_date_str")
        
        # Generate and join airport features
        if self.airport_models is not None and len(self.airport_models) > 0:
            # Get unique airports from input DataFrame
            unique_airports = df_with_features.select(self.origin_col).distinct().filter(col(self.origin_col).isNotNull()).toPandas()[self.origin_col].unique()
            
            airport_features_list = []
            for airport in unique_airports:
                if airport in self.airport_models:
                    training_dates = None
                    if self.airport_training_dates is not None:
                        training_dates = self.airport_training_dates.get(airport, pd.Series(dtype='datetime64[ns]'))
                    airport_features = self._generate_forecasts_for_dates(
                        self.airport_models[airport],
                        training_dates,
                        unique_dates
                    )
                    
                    if airport_features is not None:
                        airport_features = airport_features.rename(columns={
                            'trend': 'prophet_trend_airport',
                            'weekly': 'prophet_weekly_seasonality_airport',
                            'yhat': 'prophet_forecast_dep_delay_airport',
                            'yhat_lower': 'prophet_forecast_lower_airport',
                            'yhat_upper': 'prophet_forecast_upper_airport'
                        })
                        if 'yearly' in airport_features.columns:
                            airport_features = airport_features.rename(columns={'yearly': 'prophet_yearly_seasonality_airport'})
                        airport_features['origin'] = airport
                        airport_features_list.append(airport_features)
            
            if airport_features_list:
                airport_features_df = pd.concat(airport_features_list, ignore_index=True)
                airport_spark = self._spark.createDataFrame(airport_features_df)
                airport_feature_cols = [c for c in airport_spark.columns if c not in ['ds', 'date_str', 'origin']]
                airport_spark_features = airport_spark.select(
                    [col("origin").alias("airport_join_key"),
                     col("date_str").alias("airport_date_str")] + 
                    [col(c) for c in airport_feature_cols]
                )
                
                df_with_features = df_with_features.join(
                    broadcast(airport_spark_features),
                    (col(self.origin_col) == col("airport_join_key")) & 
                    (col("date_str_join") == col("airport_date_str")),
                    "left"
                ).drop("airport_join_key", "airport_date_str")
        
        # Ensure all expected Prophet columns exist (create with NULL if missing)
        expected_prophet_cols = [
            'prophet_forecast_dep_delay_global', 'prophet_trend_global', 
            'prophet_weekly_seasonality_global', 'prophet_yearly_seasonality_global',
            'prophet_forecast_dep_delay_carrier', 'prophet_trend_carrier',
            'prophet_weekly_seasonality_carrier', 'prophet_yearly_seasonality_carrier',
            'prophet_forecast_dep_delay_airport', 'prophet_trend_airport',
            'prophet_weekly_seasonality_airport', 'prophet_yearly_seasonality_airport'
        ]
        
        for col_name in expected_prophet_cols:
            if col_name not in df_with_features.columns:
                df_with_features = df_with_features.withColumn(col_name, F.lit(None).cast('double'))
        
        # Fill NULL values with 0 for Prophet forecast columns
        prophet_cols = [c for c in df_with_features.columns if 'prophet' in c.lower()]
        if prophet_cols:
            fill_dict = {col_name: 0.0 for col_name in prophet_cols}
            df_with_features = df_with_features.fillna(fill_dict)
        
        # Drop temporary date_str_join column
        df_with_features = df_with_features.drop("date_str_join")
        
        return df_with_features


class TimeSeriesFeaturesEstimator(Estimator):
    """
    Spark ML Estimator that adds time-series based features (Prophet forecasts) to flight data.
    
    In fit(): Generates time-series aggregations and fits Prophet models
    Returns a TimeSeriesFeaturesModel that can transform DataFrames
    
    Features added:
    - Global level: prophet_forecast_dep_delay_global, prophet_trend_global, etc.
    - Carrier level: prophet_forecast_dep_delay_carrier, prophet_trend_carrier, etc.
    - Airport level: prophet_forecast_dep_delay_airport, prophet_trend_airport, etc.
    """
    
    def __init__(self,
                 date_col="FL_DATE",
                 carrier_col="op_carrier",
                 origin_col="origin",
                 delay_col="DEP_DELAY",
                 min_days_required=14,
                 changepoint_prior_scale=0.05):
        super(TimeSeriesFeaturesEstimator, self).__init__()
        self.date_col = date_col
        self.carrier_col = carrier_col
        self.origin_col = origin_col
        self.delay_col = delay_col
        self.min_days_required = min_days_required
        self.changepoint_prior_scale = changepoint_prior_scale
        self._spark = SparkSession.builder.getOrCreate()
    
    def _fit_prophet_model(self, data, y_col='y'):
        """
        Fit a Prophet model on time series data.
        
        Parameters:
        - data: pandas DataFrame with 'ds' (date) and y_col columns
        - y_col: name of the target column
        
        Returns:
        - Tuple of (fitted Prophet model, training dates as pd.Series) or (None, None) if insufficient data
        """
        if len(data) < self.min_days_required:
            return None, None
        
        # Prepare data for Prophet
        prophet_data = data[['ds', y_col]].copy()
        prophet_data = prophet_data.rename(columns={y_col: 'y'})
        prophet_data = prophet_data.dropna()
        
        if len(prophet_data) < self.min_days_required:
            return None, None
        
        # Determine seasonality based on data availability
        has_enough_for_yearly = len(prophet_data) >= 365
        has_enough_for_weekly = len(prophet_data) >= 14
        
        # Fit Prophet model
        prophet_model = Prophet(
            yearly_seasonality=has_enough_for_yearly,
            weekly_seasonality=has_enough_for_weekly,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            interval_width=0.95,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        prophet_model.fit(prophet_data)
        
        # Return the fitted model and training dates
        training_dates = prophet_data['ds'].copy()
        return prophet_model, training_dates
    
    def _fit(self, df):
        """Generate time-series aggregations and fit Prophet models"""
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Generating time-series features...")
        
        # Prepare date column
        df_prep = df.withColumn(
            "date",
            to_timestamp(col(self.date_col), "yyyy-MM-dd").cast("date")
        ).filter(
            col("date").isNotNull() & 
            col(self.delay_col).isNotNull()
        )
        
        global_model = None
        global_training_dates = None
        carrier_models = {}
        carrier_training_dates = {}
        airport_models = {}
        airport_training_dates = {}
        
        # 1. Global time-series aggregation
        print("  Generating global time-series...")
        global_ts_spark = (
            df_prep
            .groupBy("date")
            .agg(
                F.avg(self.delay_col).alias("avg_dep_delay"),
                F.count("*").alias("flight_count")
            )
            .orderBy("date")
        )
        
        global_ts = global_ts_spark.toPandas()
        if len(global_ts) > 0:
            global_ts['ds'] = pd.to_datetime(global_ts['date'])
            
            # Fit Prophet model
            global_model, global_training_dates = self._fit_prophet_model(global_ts, 'avg_dep_delay')
            
            if global_model is not None:
                print(f"    ✓ Fitted global Prophet model on {len(global_training_dates)} training dates")
        
        # 2. Carrier time-series aggregation
        if self.carrier_col in df_prep.columns:
            print("  Generating carrier time-series...")
            carrier_ts_spark = (
                df_prep
                .filter(col(self.carrier_col).isNotNull())
                .groupBy(self.carrier_col, "date")
                .agg(
                    F.avg(self.delay_col).alias("avg_dep_delay"),
                    F.count("*").alias("flight_count")
                )
                .orderBy(self.carrier_col, "date")
            )
            
            carrier_ts = carrier_ts_spark.toPandas()
            if len(carrier_ts) > 0:
                carrier_ts['ds'] = pd.to_datetime(carrier_ts['date'])
                
                # Check data availability per carrier
                carrier_day_counts = carrier_ts.groupby(self.carrier_col)['ds'].count()
                carriers_with_sufficient_data = carrier_day_counts[
                    carrier_day_counts >= self.min_days_required
                ].index.tolist()
                
                print(f"    Fitting Prophet models for {len(carriers_with_sufficient_data)} carriers...")
                
                for carrier in carriers_with_sufficient_data:
                    carrier_data = carrier_ts[carrier_ts[self.carrier_col] == carrier].sort_values('ds')
                    carrier_model, carrier_training_dates_series = self._fit_prophet_model(carrier_data, 'avg_dep_delay')
                    
                    if carrier_model is not None:
                        carrier_models[carrier] = carrier_model
                        carrier_training_dates[carrier] = carrier_training_dates_series
                
                if carrier_models:
                    print(f"    ✓ Fitted Prophet models for {len(carrier_models)} carriers")
        
        # 3. Airport time-series aggregation
        if self.origin_col in df_prep.columns:
            print("  Generating airport time-series...")
            airport_ts_spark = (
                df_prep
                .filter(col(self.origin_col).isNotNull())
                .groupBy(self.origin_col, "date")
                .agg(
                    F.avg(self.delay_col).alias("avg_dep_delay"),
                    F.count("*").alias("flight_count")
                )
                .orderBy(self.origin_col, "date")
            )
            
            airport_ts = airport_ts_spark.toPandas()
            if len(airport_ts) > 0:
                airport_ts['ds'] = pd.to_datetime(airport_ts['date'])
                
                # Check data availability per airport
                airport_day_counts = airport_ts.groupby(self.origin_col)['ds'].count()
                airports_with_sufficient_data = airport_day_counts[
                    airport_day_counts >= self.min_days_required
                ].index.tolist()
                
                # Limit to top N airports for performance (can be adjusted)
                max_airports = 100
                if len(airports_with_sufficient_data) > max_airports:
                    top_airports = airport_day_counts.head(max_airports).index.tolist()
                else:
                    top_airports = airports_with_sufficient_data
                
                print(f"    Fitting Prophet models for {len(top_airports)} airports...")
                
                for airport in top_airports:
                    airport_data = airport_ts[airport_ts[self.origin_col] == airport].sort_values('ds')
                    airport_model, airport_training_dates_series = self._fit_prophet_model(airport_data, 'avg_dep_delay')
                    
                    if airport_model is not None:
                        airport_models[airport] = airport_model
                        airport_training_dates[airport] = airport_training_dates_series
                
                if airport_models:
                    print(f"    ✓ Fitted Prophet models for {len(airport_models)} airports")
        
        end_time = datetime.now()
        duration = end_time - start_time
        timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ✓ Time-series feature generation complete! (took {duration})")
        
        # Return a Model instance with fitted Prophet models
        return TimeSeriesFeaturesModel(
            global_model=global_model,
            carrier_models=carrier_models if carrier_models else None,
            airport_models=airport_models if airport_models else None,
            global_training_dates=global_training_dates,
            carrier_training_dates=carrier_training_dates if carrier_training_dates else None,
            airport_training_dates=airport_training_dates if airport_training_dates else None,
            date_col=self.date_col,
            carrier_col=self.carrier_col,
            origin_col=self.origin_col,
            min_days_required=self.min_days_required
        )