"""
U.S. market calendar utilities.
Provides next trading day based on NYSE calendar.
"""

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

class USMarketCalendar:
    """
    U.S. stock market calendar (NYSE holidays).
    """
    
    def __init__(self):
        self.calendar = USFederalHolidayCalendar()
        self.holidays = self.calendar.holidays(start='2000-01-01', end='2030-12-31')
        self.trading_day = CustomBusinessDay(holidays=self.holidays)
    
    def next_trading_day(self, date=None):
        """
        Return the next trading day.
        If today is a trading day, returns today; otherwise returns the next valid trading day.
        """
        if date is None:
            date = pd.Timestamp.today().normalize()
        else:
            date = pd.Timestamp(date).normalize()
        
        if self.is_trading_day(date):
            return date
        
        return date + self.trading_day
    
    def is_trading_day(self, date=None):
        """Check if the given date is a trading day."""
        if date is None:
            date = pd.Timestamp.today().normalize()
        else:
            date = pd.Timestamp(date).normalize()
        return (date.weekday() < 5) and (date not in self.holidays)
