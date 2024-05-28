import pandas as pd
from prophet import Prophet

class ProphetForecast:
    def __init__(self, dates, values):
        self.df = pd.DataFrame({'ds': pd.to_datetime(dates), 'y': values})
        self.model = Prophet()
        
        self.fit_model()
        self.create_future_dataframe()
        self.make_predictions()
        
    def fit_model(self):
        self.model.fit(self.df)
    
    def create_future_dataframe(self, periods=24, freq='M'):
        last_date = pd.to_datetime('2023-12-31')  # Ensure it starts from a specific date
        future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
        self.future = pd.DataFrame({'ds': future_dates})
    
    def make_predictions(self):
        self.forecast = self.model.predict(self.future)
    
    def print_forecast(self, n):
        forecast_dict = {
            "dates": self.forecast['ds'].head(n).dt.strftime('%Y-%m-%d').tolist(),
            "ordered": self.forecast['yhat'].head(n).tolist()
        }
        return forecast_dict
