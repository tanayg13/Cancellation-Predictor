import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    print("--- Customer Cancellation Predictor ---")
    
    # 1. Load the Data
    try:
        df = pd.read_csv('../data/customer_data.csv')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Error: Could not find customer_data.csv. Ensure you are running this from the 'src' folder.")
        return

    # 2. Prepare the Data
    # 'X' is the features (what we use to predict)
    # 'y' is the target (what we are trying to predict: canceled)
    X = df[['months_active', 'support_tickets', 'monthly_bill', 'late_payments']]
    y = df['canceled']

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train the Model
    print("Training Machine Learning Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Test the Model's Accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Training Complete. Accuracy on test data: {accuracy * 100:.2f}%\n")

    # 5. Make a Real-World Prediction
    print("--- Run a New Prediction ---")
    print("Imagine a new customer with the following stats:")
    print("- 3 months active\n- 4 support tickets raised\n- $95 monthly bill\n- 2 late payments")
    
    # Create a DataFrame for the new customer with matching feature names
    new_customer = pd.DataFrame({
        'months_active': [3],
        'support_tickets': [4],
        'monthly_bill': [95],
        'late_payments': [2]
    })
    
    risk_prediction = model.predict(new_customer)

    if risk_prediction[0] == 1:
        print("\n⚠️ ALERT: High Risk of Cancellation! Recommend sending a discount offer.")
    else:
        print("\n✅ STATUS: Customer is stable. No action required.")

if __name__ == "__main__":
    main()