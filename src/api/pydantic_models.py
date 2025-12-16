from pydantic import BaseModel

class CustomerData(BaseModel):
    CustomerId: str
    CurrencyCode: float
    CountryCode: float
    ProductCategory: float
    ChannelId: float
    Amount: float
    Value: float
    PricingStrategy: float
    Total_Transaction_Amount: float
    Average_Transaction_Amount: float
    Transaction_Count: float
    Std_Dev_Transaction_Amount: float
    Transaction_Hour: float
    Transaction_Day: float
    Transaction_Month: float
    Transaction_Year: float

class PredictionResponse(BaseModel):
    CustomerId: str
    risk_probability: float
