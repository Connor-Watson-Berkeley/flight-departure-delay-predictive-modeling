# OPSNET Data Integration Plan

## Data Source
**System**: Operations Network (OPSNET)
**URL**: https://www.aspm.faa.gov/opsnet/sys/main.asp
**Official Source**: FAA Air Traffic Operations and Delay Data
**Coverage**: FY 1990 through present

## Data Availability
- Historical data (2015-2021): **Publicly accessible without login**
- Data finalized 20 days after month end
- Download format: Excel spreadsheet via email link

## Two Download Types

### 1. Summary Data Download (RECOMMENDED)
**Description**: Summary of OPSNET delays by facility by date
**Granularity**: One row per facility per day
**Best for**: Joining with daily flight data

### 2. Detail Data Download
**Description**: Individual delay entries with specific aircraft information
**Granularity**: One row per delayed aircraft
**Best for**: Detailed analysis of specific delay events

## Available Fields (30 total)

### Join Keys
- **LOCID**: FAA Location ID (4 characters) → Join to ORIGIN/DEST
- **YYYYMMDD**: Date (8 characters) → Join to FL_DATE
- **STATE**: State (2 characters)
- **REGION**: FAA region code

### Facility Information
- **CLASS_ID**: Facility classification (tower, TRACON, center)
- **FTYPE**: Facility type (8 categories)
- **OPS**: Operations count

### Delay Metrics (14 fields)
- **Total system impact delays**
- **Traffic management initiative delays**
- **Departure delays** (key for our analysis!)
- **Airborne holds**
- **Held delays**

### Delay by Aircraft Type
- Air carrier delays
- Air taxi delays
- General aviation delays
- Military delays

### Delay by Cause (KEY FEATURES)
- **Weather delays**
- **Volume/traffic delays**
- **Equipment outages**
- **Runway condition delays**

### Delay Initiative Types
- **EDCT_DEL/MIN**: Electronic Departure Clearance Timing delays
- **GS_DEL/MIN**: Ground Stop Initiative delays

## Integration Strategy

### Join Logic
```python
# Join OPSNET Summary data to flight data
flight_data
  .join(opsnet_summary,
        (flight_data.ORIGIN == opsnet_summary.LOCID) &
        (flight_data.FL_DATE == opsnet_summary.YYYYMMDD),
        how='left')
  .withColumnRenamed('weather_delays', 'origin_weather_delays')
  .withColumnRenamed('volume_delays', 'origin_volume_delays')
  .withColumnRenamed('equipment_delays', 'origin_equipment_delays')
```

### Feature Engineering
Create features:
- `origin_atc_weather_delay`: Weather-related ATC delays at origin
- `origin_atc_volume_delay`: Volume-related ATC delays at origin
- `origin_atc_equipment_delay`: Equipment outage delays at origin
- `origin_total_ops`: Total operations at origin facility
- Binary indicators: `has_origin_atc_delay`, `has_origin_ground_stop`

Repeat for destination airport.

### Temporal Alignment
- OPSNET data is **daily aggregate**
- Flight data is **per-flight with timestamps**
- Join on date to get facility-wide conditions for that day
- Features represent **airport operational stress** on flight day

## Data Download Steps

### Step 1: Access OPSNET
1. Navigate to https://www.aspm.faa.gov/opsnet/sys/main.asp
2. Select "Delays" module
3. Choose date range: **2015-01-01 to 2021-12-31**

### Step 2: Configure Download
1. Select airports: Use "All ASPM Airports" (77 major airports) or specify list
2. Choose "Summary Data Download" for facility-level aggregates
3. Submit request

### Step 3: Receive Data
1. System emails link to Excel spreadsheet
2. Download and convert to CSV
3. Save to `external_data/opsnet_summary_2015_2021.csv`

### Step 4: Data Processing
1. Standardize column names
2. Convert YYYYMMDD to date format
3. Handle missing values (likely for days with no delays)
4. Document data dictionary

## Expected Impact on Model

### New Predictive Features
- **ATC delay indicators**: Capture system-wide operational stress
- **Delay causes**: Distinguish between controllable (equipment) vs uncontrollable (weather) factors
- **Facility operations**: Proxy for airport congestion beyond flight-level data

### Complementary to Existing Features
- **Weather data**: OPSNET weather delays validate/complement NOAA weather
- **Network features**: OPSNET volume delays indicate hub congestion
- **Temporal features**: Daily ATC patterns reveal operational trends

## Next Steps
1. [ ] Access OPSNET portal and download 2015-2021 summary data
2. [ ] Convert Excel to CSV and save to external_data/
3. [ ] Create data dictionary mapping OPSNET fields to our schema
4. [ ] Write PySpark code to join OPSNET data with flight data
5. [ ] Validate join quality (check for missing airports, date mismatches)
6. [ ] Engineer features from OPSNET delays
7. [ ] Assess feature importance in baseline model

## Notes
- OPSNET data is **free and publicly available** for historical dates
- 77 ASPM airports cover major hubs (likely overlap with our dataset)
- Data represents **facility-wide** conditions, not flight-specific
- Some delay causes may overlap with existing DOT delay fields (compare for consistency)
