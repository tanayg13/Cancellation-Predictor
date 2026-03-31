# Customer Cancellation Predictor using ML Classification

## 📖 Overview
This project is an automated Machine Learning classification tool designed for Telecom and SaaS companies. It analyzes customer behavior data (support tickets, billing, account age) to predict if a user is at high risk of canceling their subscription. 

**The Business Problem:** Acquiring a new customer costs up to 5x more than retaining an existing one. By the time a customer clicks "Cancel Account," it is usually too late to save them.  

**The Solution:** This automated script flags "at-risk" customers *before* they leave, allowing customer success teams to intervene with targeted discounts or support.

## 🧑‍💻 How It Works
The system uses a **Random Forest Classifier** (`scikit-learn`) to find patterns in historical customer data. It evaluates:
* `months_active`: How long they have been a customer.
* `support_tickets`: Number of complaints/issues raised.
* `monthly_bill`: How much they are paying.
* `late_payments`: Financial friction.

## ⚙️ Project Structure
Ensure your local repository matches this exact structure before running the application:

```text
cancellation-predictor/
│
├── data/
│   └── customer_data.csv        # The historical dataset used for training
│
├── src/
│   └── predictor_model.py       # The main machine learning engine
│
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation (You are here)
```

## 💻 Installation & Setup
 **Step A: Clone the repository**  
         Download the code to your local machine using Git:  
         `git clone [https://github.com/tanayg13/cancellation-predictor.git](https://github.com/tanayg13/cancellation-predictor.git)
cd cancellation-predictor`  

**Step B: Install Dependencies**  
          Install the required machine learning libraries (pandas and scikit-learn) using the provided requirements file:  
          `pip install -r requirements.txt`  
          
**Step C: Running the model**  
          Navigate to the `src` directory and run the model:  
          `cd src`  
          `python predictor_model.py`  

## 🔃 What happens when you run it?
**1. Data Loading:** The script reads the `customer_data.csv` file.

**2. Training:** It splits the data, trains the Random Forest model, and prints the accuracy percentage.

**3. Simulation:** It runs a test prediction on a hypothetical customer and outputs an actionable alert.

## 🎉 Expected Terminal Output:  
```text
--- Customer Cancellation Predictor ---
Data loaded successfully.
Training Machine Learning Model...
Model Training Complete. Accuracy on test data: 100.00%

--- Run a New Prediction ---
Imagine a new customer with the following stats:
- 3 months active
- 4 support tickets raised
- $95 monthly bill
- 2 late payments

⚠️ ALERT: High Risk of Cancellation! Recommend sending a discount offer.
```  

## 🖥️ Technologies Used
* **Python 3**
* **Pandas:** For data manipulation and loading the CSV.
* **Scikit-Learn:** For the classification algorithm and accuracy metrics.

## 👤 Author
Tanay Gupta  
25BCE10082
