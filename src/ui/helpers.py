from datetime import datetime, date, time
import calendar


def max_day(month: int, year: int = 2020) -> int:
    return calendar.monthrange(year, month)[1]

def get_timestamp_from_sliders(year, month, day, hour):
    return datetime.combine(date(year, month, day), time(hour=hour))
