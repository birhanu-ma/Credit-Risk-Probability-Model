from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
  "CustomerId": "CustomerId_83",
  "CurrencyCode": -5.551115123125783e-17,
  "CountryCode": -5.551115123125783e-17,
  "ProductCategory": -0.1215820452191876,
  "ChannelId": -0.3546444573641953,
  "Amount": -0.1046755531397042,
  "Value": -0.3816849657965848,
  "PricingStrategy": -0.2980638802036633,
  "Total_Transaction_Amount": -0.6090364182048986,
  "Average_Transaction_Amount": -0.2835070905671694,
  "Transaction_Count": -0.5585921178765559,
  "Std_Dev_Transaction_Amount": -3.0751792152127257,
  "Transaction_Hour": -2.1366056943909224,
  "Transaction_Day": -0.1924793001391105,
  "Transaction_Month": 0.8126293406289661,
  "Transaction_Year": -0.9562260512840092
})
    
    # Assert that the response is valid
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert isinstance(data['risk_probability'], (float, int))

def test_predict_endpoint_missing_data():
    # Test endpoint with missing data
    response = client.post("/predict", json={
        "CustomerId": "C1",
        "CurrencyCode": 1,
        "CountryCode": 254,
        "ProductCategory": 3,
        "ChannelId": 2,
        # Missing other fields to simulate error
    })
    
    assert response.status_code == 422  # Unprocessable Entity (missing required fields)
