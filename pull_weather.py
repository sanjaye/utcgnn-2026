"""
Pull historical weather for Los Angeles from the Open-Meteo Historical Weather
(archive) API for every session date in the train, validation and test splits.

The splits are chronological by season (see utcgnn.experiment.split_sessions), so
together they cover every session date in the match file. Session dates are taken
the same way as utcgnn.data.load_data: the local calendar day of `timestamp`.

Categories included: temperature, humidity & pressure, precipitation, wind,
clouds, sun & solar radiation, and weather conditions.
(Agriculture/soil and upper-atmosphere variables are left out.)

Output:
  - data/la_daily_summary.csv                    one row per date with daily highs/lows/totals
  - data/la_weather_hourly/la_weather_YYYY-MM-DD.csv   one file per date, 24 hourly rows

Requirements:  pip install requests pandas
Usage:         python pull_weather.py [--matches data/morphed_matches.csv]
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# --- Settings ---------------------------------------------------------------
LATITUDE = 34.0522      # Los Angeles
LONGITUDE = -118.2437
TIMEZONE = "America/Los_Angeles"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DATA_DIR = Path("data")
HOURLY_DIR = DATA_DIR / "la_weather_hourly"
SUMMARY_PATH = DATA_DIR / "la_daily_summary.csv"
MATCHES_PATH = DATA_DIR / "morphed_matches.csv"

ARCHIVE_LAG_DAYS = 5    # the archive is usually ~5 days behind today

UNITS = {
    "temperature_unit": "fahrenheit",   # or "celsius"
    "wind_speed_unit": "mph",           # or "kmh", "ms", "kn"
    "precipitation_unit": "inch",       # or "mm"
}

# Hourly variables, grouped by category
HOURLY_VARIABLES = [
    # Temperature
    "temperature_2m",
    "apparent_temperature",
    "dew_point_2m",
    # Humidity & pressure
    "relative_humidity_2m",
    "vapour_pressure_deficit",
    "pressure_msl",
    "surface_pressure",
    # Precipitation
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    # Wind
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    # Clouds
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    # Sun & solar radiation
    "sunshine_duration",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "terrestrial_radiation",
    # Conditions
    "weather_code",
    "is_day",
]

# Daily summary variables, grouped by category
DAILY_VARIABLES = [
    # Temperature
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    # Precipitation
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    # Wind
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    # Sun & solar radiation
    "sunrise",
    "sunset",
    "daylight_duration",
    "sunshine_duration",
    "shortwave_radiation_sum",
    # Conditions
    "weather_code",
]


def session_dates(matches_path: Path = MATCHES_PATH):
    """All session dates (train + validation + test) in the match file, as ISO strings."""
    matches = pd.read_csv(matches_path, encoding="utf-8-sig")
    return sorted(matches["timestamp"].str.slice(0, 10).unique())


def _contiguous_ranges(dates):
    """Group sorted dates into (start, end) runs of consecutive days."""
    ranges = []
    start = prev = dates[0]
    for d in dates[1:]:
        if d - prev > timedelta(days=1):
            ranges.append((start, prev))
            start = d
        prev = d
    ranges.append((start, prev))
    return ranges


def _fetch_archive(start: date, end: date):
    """Fetch hourly and daily data for one date range. Returns (hourly_df, daily_df)."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": TIMEZONE,
        **UNITS,
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo error {resp.status_code}: {resp.text}")
    data = resp.json()

    hourly = pd.DataFrame(data["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly.insert(0, "date", hourly["time"].dt.date)

    daily = pd.DataFrame(data["daily"])
    daily["time"] = pd.to_datetime(daily["time"]).dt.date
    daily = daily.rename(columns={"time": "date"})

    return hourly, daily


def get_la_weather(dates, hourly_dir: Path = HOURLY_DIR,
                   summary_path: Path = SUMMARY_PATH) -> pd.DataFrame:
    """
    Download historical LA weather for the given dates and write one CSV per date.

    `dates` can be strings ("2024-07-04") or datetime.date objects, in any order.
    Returns the daily summary DataFrame.
    """
    wanted = sorted({pd.to_datetime(d).date() for d in dates})
    cutoff = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)

    too_recent = [d for d in wanted if d > cutoff]
    if too_recent:
        print(f"Skipping {len(too_recent)} date(s) not yet in the archive "
              f"(after {cutoff}): {[d.isoformat() for d in too_recent]}")
    wanted = [d for d in wanted if d <= cutoff]
    if not wanted:
        print("No historical dates to fetch.")
        return pd.DataFrame()

    hourly_dir.mkdir(parents=True, exist_ok=True)
    daily_frames = []

    # One request per run of consecutive dates keeps downloads small
    for start, end in _contiguous_ranges(wanted):
        print(f"Fetching {start} to {end} ...")
        hourly, daily = _fetch_archive(start, end)
        daily_frames.append(daily)

        for day, day_df in hourly.groupby("date"):
            path = hourly_dir / f"la_weather_{day.isoformat()}.csv"
            day_df.to_csv(path, index=False)
            print(f"  saved {path} ({len(day_df)} hourly rows)")

    summary = pd.concat(daily_frames, ignore_index=True).sort_values("date")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Saved daily summary for {len(summary)} dates to {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH)
    args = parser.parse_args()

    # Every session date in the train, validation and test splits.
    my_dates = session_dates(args.matches)
    print(f"{len(my_dates)} session dates from {my_dates[0]} to {my_dates[-1]}")

    get_la_weather(my_dates)
