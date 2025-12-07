"""
time_series_features.py - Time-series based feature engineering for flight delay prediction

Provides TimeSeriesFeaturesEstimator for use in Spark ML pipelines.
Generates Prophet-based time-series features (global, carrier, airport level) that capture
temporal trends and seasonality.
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, to_timestamp, when
from pyspark.ml.base import Estimator, Model
import pandas as pd
from prophet import Prophet


class TimeSeriesFeaturesModel(Model):
    """Model returned by TimeSeriesFeaturesEstimator after fitting"""
    
    def __init__(self, 
                 global_features=None,
                 carrier_features=None,
                 airport_features=None,
                 date_col="FL_DATE",
                 carrier_col="op_carrier",
                 origin_col="origin"):
        super(TimeSeriesFeaturesModel, self).__init__()
        self.global_features = global_features
        self.carrier_features = carrier_features
        self.airport_features = airport_features
        self.date_col = date_col
        self.carrier_col = carrier_col
        self.origin_col = origin_col
        self._spark = SparkSession.builder.getOrCreate()
    
    def _transform(self, df):
        """Join Prophet time-series features to input DataFrame"""
        if self.global_features is None and self.carrier_features is None and self.airport_features is None:
            raise ValueError("Model must be fitted before transform()")
        
        df_with_features = df
        
        # Join global features (by date)
        if self.global_features is not None:
            global_spark = self._spark.createDataFrame(self.global_features)
            # Select only feature columns (exclude join keys to avoid ambiguity)
            global_feature_cols = [c for c in global_spark.columns if c not in ['ds', 'date_str']]
            # Rename join key to avoid ambiguity, keep feature column names as-is
            global_spark_features = global_spark.select(
                [col("date_str").alias("global_date_str")] + 
                [col(c) for c in global_feature_cols]
            )
            
            df_with_features = df_with_features.join(
                global_spark_features,
                col(self.date_col) == col("global_date_str"),
                "left"
            ).drop("global_date_str")  # Drop the join key after joining
        
        # Join carrier features (by carrier and date)
        if self.carrier_features is not None:
            carrier_spark = self._spark.createDataFrame(self.carrier_features)
            # Select only feature columns (exclude join keys to avoid ambiguity)
            carrier_feature_cols = [c for c in carrier_spark.columns if c not in ['ds', 'date_str', 'carrier']]
            # Rename join keys to avoid ambiguity, keep feature column names as-is
            carrier_spark_features = carrier_spark.select(
                [col("carrier").alias("carrier_join_key"),
                 col("date_str").alias("carrier_date_str")] + 
                [col(c) for c in carrier_feature_cols]
            )
            
            df_with_features = df_with_features.join(
                carrier_spark_features,
                (col(self.carrier_col) == col("carrier_join_key")) & 
                (col(self.date_col) == col("carrier_date_str")),
                "left"
            ).drop("carrier_join_key", "carrier_date_str")  # Drop join keys after joining
        
        # Join airport features (by origin airport and date)
        if self.airport_features is not None:
            airport_spark = self._spark.createDataFrame(self.airport_features)
            # Select only feature columns (exclude join keys to avoid ambiguity)
            airport_feature_cols = [c for c in airport_spark.columns if c not in ['ds', 'date_str', 'origin']]
            # Rename join keys to avoid ambiguity, keep feature column names as-is
            airport_spark_features = airport_spark.select(
                [col("origin").alias("airport_join_key"),
                 col("date_str").alias("airport_date_str")] + 
                [col(c) for c in airport_feature_cols]
            )
            
            df_with_features = df_with_features.join(
                airport_spark_features,
                (col(self.origin_col) == col("airport_join_key")) & 
                (col(self.date_col) == col("airport_date_str")),
                "left"
            ).drop("airport_join_key", "airport_date_str")  # Drop join keys after joining
        
        # Ensure all expected Prophet columns exist (create with NULL if missing)
        # This prevents imputer from failing if some features weren't generated
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
        for col_name in prophet_cols:
            df_with_features = df_with_features.fillna({col_name: 0.0})
        
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
        - Forecast DataFrame (or None if insufficient data)
        """
        if len(data) < self.min_days_required:
            return None
        
        # Prepare data for Prophet
        prophet_data = data[['ds', y_col]].copy()
        prophet_data = prophet_data.rename(columns={y_col: 'y'})
        prophet_data = prophet_data.dropna()
        
        if len(prophet_data) < self.min_days_required:
            return None
        
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
        
        # Generate forecast
        forecast = prophet_model.predict(prophet_data[['ds']])
        
        return forecast
    
    def _fit(self, df):
        """Generate time-series aggregations and fit Prophet models"""
        print("Generating time-series features...")
        
        # Prepare date column
        df_prep = df.withColumn(
            "date",
            to_timestamp(col(self.date_col), "yyyy-MM-dd").cast("date")
        ).filter(
            col("date").isNotNull() & 
            col(self.delay_col).isNotNull()
        )
        
        global_features = None
        carrier_features = None
        airport_features = None
        
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
            forecast_global = self._fit_prophet_model(global_ts, 'avg_dep_delay')
            
            if forecast_global is not None:
                # Extract features
                feature_cols = ['ds', 'trend', 'weekly', 'yhat', 'yhat_lower', 'yhat_upper']
                if 'yearly' in forecast_global.columns:
                    feature_cols.insert(2, 'yearly')
                
                global_features = forecast_global[feature_cols].copy()
                global_features = global_features.rename(columns={
                    'trend': 'prophet_trend_global',
                    'weekly': 'prophet_weekly_seasonality_global',
                    'yhat': 'prophet_forecast_dep_delay_global',
                    'yhat_lower': 'prophet_forecast_lower_global',
                    'yhat_upper': 'prophet_forecast_upper_global'
                })
                if 'yearly' in global_features.columns:
                    global_features = global_features.rename(columns={
                        'yearly': 'prophet_yearly_seasonality_global'
                    })
                global_features['date_str'] = global_features['ds'].dt.strftime('%Y-%m-%d')
                print(f"    ✓ Generated global Prophet features for {len(global_features)} dates")
        
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
                
                carrier_features_list = []
                for carrier in carriers_with_sufficient_data:
                    carrier_data = carrier_ts[carrier_ts[self.carrier_col] == carrier].sort_values('ds')
                    forecast = self._fit_prophet_model(carrier_data, 'avg_dep_delay')
                    
                    if forecast is not None:
                        feature_cols = ['ds', 'trend', 'weekly', 'yhat', 'yhat_lower', 'yhat_upper']
                        if 'yearly' in forecast.columns:
                            feature_cols.insert(2, 'yearly')
                        
                        features = forecast[feature_cols].copy()
                        features['carrier'] = carrier
                        features = features.rename(columns={
                            'trend': 'prophet_trend_carrier',
                            'weekly': 'prophet_weekly_seasonality_carrier',
                            'yhat': 'prophet_forecast_dep_delay_carrier',
                            'yhat_lower': 'prophet_forecast_lower_carrier',
                            'yhat_upper': 'prophet_forecast_upper_carrier'
                        })
                        if 'yearly' in features.columns:
                            features = features.rename(columns={
                                'yearly': 'prophet_yearly_seasonality_carrier'
                            })
                        features['date_str'] = features['ds'].dt.strftime('%Y-%m-%d')
                        carrier_features_list.append(features)
                
                if carrier_features_list:
                    carrier_features = pd.concat(carrier_features_list, ignore_index=True)
                    print(f"    ✓ Generated carrier Prophet features for {carrier_features['carrier'].nunique()} carriers")
        
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
                
                airport_features_list = []
                for airport in top_airports:
                    airport_data = airport_ts[airport_ts[self.origin_col] == airport].sort_values('ds')
                    forecast = self._fit_prophet_model(airport_data, 'avg_dep_delay')
                    
                    if forecast is not None:
                        feature_cols = ['ds', 'trend', 'weekly', 'yhat', 'yhat_lower', 'yhat_upper']
                        if 'yearly' in forecast.columns:
                            feature_cols.insert(2, 'yearly')
                        
                        features = forecast[feature_cols].copy()
                        features['origin'] = airport
                        features = features.rename(columns={
                            'trend': 'prophet_trend_airport',
                            'weekly': 'prophet_weekly_seasonality_airport',
                            'yhat': 'prophet_forecast_dep_delay_airport',
                            'yhat_lower': 'prophet_forecast_lower_airport',
                            'yhat_upper': 'prophet_forecast_upper_airport'
                        })
                        if 'yearly' in features.columns:
                            features = features.rename(columns={
                                'yearly': 'prophet_yearly_seasonality_airport'
                            })
                        features['date_str'] = features['ds'].dt.strftime('%Y-%m-%d')
                        airport_features_list.append(features)
                
                if airport_features_list:
                    airport_features = pd.concat(airport_features_list, ignore_index=True)
                    print(f"    ✓ Generated airport Prophet features for {airport_features['origin'].nunique()} airports")
        
        print("✓ Time-series feature generation complete!")
        
        # Return a Model instance
        return TimeSeriesFeaturesModel(
            global_features=global_features,
            carrier_features=carrier_features,
            airport_features=airport_features,
            date_col=self.date_col,
            carrier_col=self.carrier_col,
            origin_col=self.origin_col
        )

