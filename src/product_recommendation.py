"""Product recommendation function for the group application."""
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES = ["engagement_score", "purchase_interest_score", "social_profile_count", "high_engagement", "social_media_platform", "review_sentiment"]

def predict_product(customer_id):
    data = pd.read_csv(BASE_DIR / "datasets" / "merged_dataset.csv")
    model = joblib.load(BASE_DIR / "models" / "product_model.pkl")
    try:
        customer_id = int(str(customer_id).strip().lstrip("Aa"))
    except ValueError:
        raise ValueError("Customer ID must be a number such as 151 or A151")
    customer = data.loc[data["customer_id"] == customer_id]
    if customer.empty:
        raise ValueError(f"Customer {customer_id} was not found")
    return str(model.predict(customer[FEATURES].head(1))[0])

if __name__ == "__main__":
    customer_id = input("Enter customer ID: ")
    try:
        print("Predicted product:", predict_product(customer_id))
    except ValueError as error:
        print(error)
