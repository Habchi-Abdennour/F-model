import pandas as pd
from prophet import Prophet

# Suppress warning messages

class ProphetForecast:
    def __init__(self, dates, values):
        self.df = pd.DataFrame({'ds': pd.to_datetime(dates), 'y': values})
        self.model = Prophet()
        
        self.fit_model()
        self.create_future_dataframe()
        self.make_predictions()
        
    def fit_model(self):
        self.model.fit(self.df)
    
    def create_future_dataframe(self, periods=36, freq='MS'):
        self.future = self.model.make_future_dataframe(periods=periods, freq=freq)
    
    def make_predictions(self):
        self.forecast = self.model.predict(self.future)
    
    def print_forecast(self, n):
        forecast_dict = {
            "dates": self.forecast['ds'].tail(n).dt.strftime('%Y-%m-%d').tolist(),
            "ordered": self.forecast['yhat'].tail(n).tolist()
        }
        return(forecast_dict)

