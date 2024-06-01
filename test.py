import pandas as pd
from prophet import Prophet

class ProphetForecast:
    def __init__(self, dates, values, periods=24, freq='M'):
        self.df = pd.DataFrame({'ds': pd.to_datetime(dates), 'y': values})
        self.model = Prophet()
        self.periods = periods
        self.freq = freq
        
        self.fit_model()
        self.create_future_dataframe()
        self.make_predictions()
        
    def fit_model(self):
        self.model.fit(self.df)
    
    def create_future_dataframe(self):
        # Use the last date from the input data to start the future dates
        last_date = self.df['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=self.periods+1, freq=self.freq)[1:]
        self.future = pd.DataFrame({'ds': future_dates})
    
    def make_predictions(self):
        self.forecast = self.model.predict(self.future)
    
    def print_forecast(self, n):
        forecast_dict = {
            "dates": self.forecast['ds'].head(n).dt.strftime('%Y-%m-%d').tolist(),
            "forecasted_values": self.forecast['yhat'].head(n).tolist()
        }
        return forecast_dict

