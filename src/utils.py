import pandas as pd


def create_datetime(calendar: pd.DataFrame) -> pd.DataFrame:
    # Define start date (Monday)
    start_date = pd.Timestamp("2021-01-04")

    # Create date column (start of week)
    calendar["date"] = start_date + pd.to_timedelta(calendar["week_id"] * 7, unit="D")

    # Optional: ensure it's datetime64
    calendar["date"] = pd.to_datetime(calendar['date'], format='%Y-%m-%d %H:%M:%S')

    return calendar