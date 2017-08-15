from datetime import datetime as dt
import time

def local_time_as_epoc_seconds(year, month, day, hour, minute, second):
    """ Return the epoc given the local time
    :param year: Year
    :param month: Month
    :param day: Day
    :param hour: Hour
    :param minute: Minute
    :param second: Second
    :return: The epoc time
    """
    return time.mktime(dt(year, month, day, hour, minute, second).timetuple())*1000

def datetime_ns_as_epoc_seconds(mydatetime):
    """ Return the epoc given a date object (in ns precision)
    :return: The epoc time
    """
    return mydatetime.astype(int) / 1e9

