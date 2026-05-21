from datetime import datetime, timedelta, date, time
import calendar

def max_day(month: int, year: int = 2020) -> int:
    return calendar.monthrange(year, month)[1]

def get_timestamp_from_sliders(month, day, hour):
    year=2020
    return datetime.combine(date(year, month, day), time(hour=hour))