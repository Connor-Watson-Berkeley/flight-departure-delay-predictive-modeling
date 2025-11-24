# External Data Sources

This directory contains external event data that may impact flight delays.

## Data Sources Research

### Government Shutdown Data

**Finding:** No public API available for government shutdown data.

**Manual Data Source Options:**
- Congressional Research Service (CRS) reports
- Wikipedia: [Government shutdowns in the United States](https://en.wikipedia.org/wiki/Government_shutdowns_in_the_United_States)
- Congress.gov: [Past Government Shutdowns: Key Resources](https://www.congress.gov/crs-product/R41759)

**Shutdowns in 2015-2021 Timeframe:**
- **December 22, 2018 - January 25, 2019**: 35-day partial government shutdown (longest in US history)
  - Impact: FAA operations, TSA security, air traffic control funding affected

**Recommended Approach:**
Create a CSV file: `government_shutdowns.csv` with columns:
```
start_date,end_date,duration_days,type,impact_level
2018-12-22,2019-01-25,35,partial,high
```

### FAA Strike/Operational Disruption Data

**Finding:** No public API for strike data. The major PATCO strike was in 1981 (outside our timeframe).

**Available FAA Data Portals:**
- [FAA Data Portal](https://www.faa.gov/data) - General FAA datasets
- [OPSNET](https://www.aspm.faa.gov/opsnet/sys/main.asp) - Air traffic operations and delay data (FY 1990+)
- [NAS Status](https://nasstatus.faa.gov/) - Current National Airspace System status

**2015-2021 Operational Issues:**
No major strikes occurred, but consider:
- Government shutdown impact on ATC (Dec 2018 - Jan 2019)
- Staffing shortages at specific facilities
- Weather-related ground stops/delays (available in OPSNET)

**Recommended Approach:**
1. Use OPSNET data for facility-level delay causes
2. Create manual CSV for known operational events:
```
date,event_type,scope,affected_airports,description
2018-01-25,staffing_shortage,regional,"EWR,JFK,LGA",ATC staffing issues during shutdown
```

## Data Files

### Planned Data Structure

```
external_data/
├── README.md (this file)
├── government_shutdowns.csv
├── faa_operational_events.csv
└── sources.txt (documentation of where data came from)
```

## Integration Plan

These event datasets can be merged with flight data as binary indicator features:
- `is_gov_shutdown`: 1 if flight date falls within shutdown period, 0 otherwise
- `has_atc_event`: 1 if operational event affecting origin/destination airport, 0 otherwise

These features capture systemic disruptions that may not be reflected in weather or historical delay patterns.
