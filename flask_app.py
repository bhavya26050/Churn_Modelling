# from flask import Flask, render_template, request, jsonify
# import joblib
# import random

# app = Flask(__name__)

# # Load the pre-trained models
# ensemble_model = joblib.load('ensemble_model.pkl')
# preprocessor = joblib.load('preprocessor.pkl')

# # Initialize a dictionary to store user inputs
# user_data = {}

# # A few simple responses
# greetings = ["Hello! How can I help you today?", "Hi there! What would you like to know?", "Hello! Ask me anything!"]
# fallbacks = ["I'm sorry, I don't understand. Can you ask something else?", "I'm not sure how to help with that.", "Can you clarify your question?"]

# # Serve the chatbot HTML page
# @app.route('/')
# def index():
#     return render_template('chat.html')

# # Handle chat interactions
# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message').lower()

#     # Define a global variable to track prediction steps
#     global user_data

#     # Example response logic
#     if "hello" in user_message or "hi" in user_message or "hey" in user_message or "Ha bhai" in user_message:
#         bot_response = random.choice(greetings)

#     elif "predict churn" in user_message:
#         user_data = {}  # Reset the user data when a new prediction starts
#         bot_response = "Great! Let's get started. What's the customer's credit score?"

#     elif 'credit score' not in user_data:
#         try:
#             credit_score = int(user_message)
#             user_data['credit score'] = credit_score
#             bot_response = "Got it! What's the customer's age?"
#         except ValueError:
#             bot_response = "Please enter a valid number for the credit score."

#     elif 'age' not in user_data:
#         try:
#             age = int(user_message)
#             user_data['age'] = age
#             bot_response = "What's the customer's tenure (years with the company)?"
#         except ValueError:
#             bot_response = "Please enter a valid number for the age."

#     elif 'tenure' not in user_data:
#         try:
#             tenure = int(user_message)
#             user_data['tenure'] = tenure
#             bot_response = "What's the customer's balance?"
#         except ValueError:
#             bot_response = "Please enter a valid number for tenure."

#     elif 'balance' not in user_data:
#         try:
#             balance = float(user_message)
#             user_data['balance'] = balance
#             bot_response = "What's the number of products the customer uses?"
#         except ValueError:
#             bot_response = "Please enter a valid number for the balance."

#     elif 'num_of_products' not in user_data:
#         try:
#             # Ensure that user input is properly cast to an integer
#             num_of_products = int(user_message.strip())  # Strip any whitespace and cast to integer
#             user_data['num_of_products'] = num_of_products

#             # Now that we have all the necessary inputs, proceed with the prediction
#             bot_response = "Thank you! Let me predict if the customer will churn."

#             # Create the input array from user_data
#             user_input = [[
#                 user_data['credit score'], user_data['age'], user_data['tenure'], user_data['balance'], 
#                 user_data['num_of_products'], 1, 0, 0, 1, 0  # Dummy categorical variables, can be extended
#             ]]

#             # Preprocess the user input
#             transformed_input = preprocessor.transform(user_input)

#             # Make the prediction using the ensemble model
#             prediction = ensemble_model.predict(transformed_input)

#             if prediction[0] == 1:
#                 bot_response = "The customer is likely to churn."
#             else:
#                 bot_response = "The customer is unlikely to churn."

#         except ValueError:
#             bot_response = "Please enter a valid number for the number of products."

#     elif "bye" in user_message:
#         bot_response = "Goodbye! Have a great day!"

#     else:
#         bot_response = random.choice(fallbacks)

#     return jsonify({'response': bot_response})

# if __name__ == '__main__':
#     app.run(debug=True)





# from flask import Flask, render_template, request, jsonify
# import joblib
# import pandas as pd
# import random

# app = Flask(__name__)

# # Load the pre-trained models and dataset
# ensemble_model = joblib.load('ensemble_model.pkl')
# preprocessor = joblib.load('preprocessor.pkl')
# df = pd.read_csv('Churn_Modelling.csv')

# # Initialize a dictionary to store user inputs
# user_data = {}

# # Simple bot responses
# greetings = ["Hello! How can I help you today?", "Hi there! What would you like to know?", "Hello! Ask me anything!"]
# fallbacks = ["I'm sorry, I don't understand. Can you ask something else?", "I'm not sure how to help with that.", "Can you clarify your question?"]

# # Serve the chatbot HTML page
# @app.route('/')
# def index():
#     return render_template('chat.html')

# # Function to get a customer profile by ID
# def get_customer_profile(customer_id):
#     customer = df[df['CustomerId'] == customer_id]
#     return customer.iloc[0] if not customer.empty else None

# # Handle chat interactions
# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message').lower()
#     global user_data

#     # Greeting
#     if any(greet in user_message for greet in ["hello", "hi", "hey", "ha bhai"]):
#         bot_response = random.choice(greetings)

#     # Predict Churn
#     elif "predict churn" in user_message:
#         user_data = {}  # Reset user data for a new prediction
#         bot_response = "Let's begin! What's the customer's credit score?"

#     elif 'credit score' not in user_data:
#         try:
#             user_data['credit score'] = int(user_message)
#             bot_response = "Got it! What's the customer's age?"
#         except ValueError:
#             bot_response = "Please enter a valid number for the credit score."

#     elif 'age' not in user_data:
#         try:
#             user_data['age'] = int(user_message)
#             bot_response = "What's the customer's tenure (years with the company)?"
#         except ValueError:
#             bot_response = "Please enter a valid number for the age."

#     elif 'tenure' not in user_data:
#         try:
#             user_data['tenure'] = int(user_message)
#             bot_response = "What's the customer's balance?"
#         except ValueError:
#             bot_response = "Please enter a valid number for tenure."

#     elif 'balance' not in user_data:
#         try:
#             user_data['balance'] = float(user_message)
#             bot_response = "How many products does the customer use?"
#         except ValueError:
#             bot_response = "Please enter a valid number for the balance."

#     elif 'num_of_products' not in user_data:
#         try:
#             user_data['num_of_products'] = int(user_message)
#             user_input = [[
#                 user_data['credit score'], user_data['age'], user_data['tenure'],
#                 user_data['balance'], user_data['num_of_products'], 1, 0, 0, 1, 0
#             ]]
#             transformed_input = preprocessor.transform(user_input)
#             prediction = ensemble_model.predict(transformed_input)
#             bot_response = "The customer is likely to churn." if prediction[0] == 1 else "The customer is unlikely to churn."
#         except ValueError:
#             bot_response = "Please enter a valid number for the number of products."

#     # Customer Profile Overview
#     elif "customer profile" in user_message:
#         customer_id = int(input("Enter the customer ID: "))
#         customer = get_customer_profile(customer_id)
#         if customer is not None:
#             bot_response = f"Customer Profile:\n- Age: {customer['Age']}\n- Balance: {customer['Balance']}\n- Tenure: {customer['Tenure']}\n- Products: {customer['NumOfProducts']}"
#         else:
#             bot_response = "Customer not found."

#     # Geographical Analysis
#     elif "highest credit score" in user_message:
#         avg_credit = df.groupby('Geography')['CreditScore'].mean()
#         max_country = avg_credit.idxmax()
#         bot_response = f"{max_country} has the highest average credit score of {avg_credit[max_country]:.2f}."

#     # Product Usage Patterns
#     elif "average products" in user_message:
#         avg_products = df[df['IsActiveMember'] == 1]['NumOfProducts'].mean()
#         bot_response = f"Active customers use an average of {avg_products:.2f} products."

#     # Gender-Based Financial Insights
#     elif "higher balance" in user_message:
#         avg_balance_male = df[df['Gender'] == 'Male']['Balance'].mean()
#         avg_balance_female = df[df['Gender'] == 'Female']['Balance'].mean()
#         bot_response = f"Male customers have an average balance of {avg_balance_male:.2f}, while female customers have {avg_balance_female:.2f}."

#     # Customer Segmentation Insights
#     elif "segment customers" in user_message:
#         low_balance = len(df[df['Balance'] < 50000])
#         mid_balance = len(df[(df['Balance'] >= 50000) & (df['Balance'] < 100000)])
#         high_balance = len(df[df['Balance'] >= 100000])
#         bot_response = f"Low balance: {low_balance}, Mid balance: {mid_balance}, High balance: {high_balance}."

#     # Transaction Anomalies Detection
#     elif "unusual transactions" in user_message:
#         anomalies = df[df['Balance'] > 150000]
#         bot_response = f"There are {len(anomalies)} customers with unusually high balances."

#     # Exit
#     elif "bye" in user_message:
#         bot_response = "Goodbye! Have a great day!"

#     # Fallback response
#     else:
#         bot_response = random.choice(fallbacks)

#     return jsonify({'response': bot_response})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import joblib
# import random
# import re
# import pandas as pd

# app = Flask(__name__)

# # Load pre-trained models and data
# ensemble_model = joblib.load('ensemble_model.pkl')
# preprocessor = joblib.load('preprocessor.pkl')
# df = pd.read_csv('churn_Modelling.csv')  # Ensure this file is loaded correctly

# # Predefined greetings and fallback responses
# greetings = ["Hello! How can I help you today?", "Hi there! What would you like to know?"]
# fallbacks = ["I'm sorry, I don't understand. Can you ask something else?", "Can you clarify your question?"]

# # Intent handlers
# def get_avg_products():
#     avg_products = df['NumOfProducts'].mean()
#     return f"The average number of products per customer is {avg_products:.2f}."

# def get_active_customers():
#     active_count = df[df['IsActiveMember'] == 1].shape[0]
#     return f"There are {active_count} active customers."

# def get_avg_tenure_non_exited():
#     avg_tenure = df[df['Exited'] == 0]['Tenure'].mean()
#     return f"The average tenure of customers who have not exited is {avg_tenure:.2f} years."

# def get_high_credit_score_count():
#     count = df[df['CreditScore'] > 800].shape[0]
#     return f"There are {count} customers with a credit score above 800."

# def get_avg_balance_by_gender():
#     avg_balance_male = df[df['Gender'] == 'Male']['Balance'].mean()
#     avg_balance_female = df[df['Gender'] == 'Female']['Balance'].mean()
#     return (f"Male customers have an average balance of {avg_balance_male:.2f}, "
#             f"while female customers have {avg_balance_female:.2f}.")

# # Map user inputs to intent handlers
# intent_map = {
#     re.compile(r"average number of products", re.IGNORECASE): get_avg_products,
#     re.compile(r"how many active customers", re.IGNORECASE): get_active_customers,
#     re.compile(r"average tenure.*not exited", re.IGNORECASE): get_avg_tenure_non_exited,
#     re.compile(r"credit score above 800", re.IGNORECASE): get_high_credit_score_count,
#     re.compile(r"male or female.*higher balances", re.IGNORECASE): get_avg_balance_by_gender,
# }

# # Serve the chatbot HTML page
# @app.route('/')
# def index():
#     return render_template('chat.html')

# # Handle chat interactions
# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message', '').lower()

#     # Check for greetings
#     if any(greet in user_message for greet in ["hello", "hi", "hey"]):
#         return jsonify({'response': random.choice(greetings)})

#     # Match user message to an intent
#     for pattern, handler in intent_map.items():
#         if pattern.search(user_message):
#             response = handler()
#             return jsonify({'response': response})

#     # Default fallback response
#     return jsonify({'response': random.choice(fallbacks)})

# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, render_template, request, jsonify
# import joblib
# import random
# import pandas as pd

# app = Flask(__name__)

# # Load models and dataset
# ensemble_model = joblib.load('ensemble_model.pkl')
# preprocessor = joblib.load('preprocessor.pkl')
# df = pd.read_csv('churn_Modelling.csv')  # Ensure this file exists

# # Predefined responses
# greetings = ["Hello! How can I assist you today?", "Hi! What would you like to know?", "Hey! Ask me anything."]
# farewells = ["Goodbye! Have a great day!", "See you later!", "Take care!"]
# fallbacks = ["I'm not sure about that. Could you rephrase?", "I didn't get that. Can you try again?"]

# # Intent-based responses
# def get_avg_products():
#     return f"The average number of products per customer is {df['NumOfProducts'].mean():.2f}."

# def get_active_customers():
#     return f"There are {df[df['IsActiveMember'] == 1].shape[0]} active customers."

# def get_avg_tenure_non_exited():
#     return f"The average tenure of non-exited customers is {df[df['Exited'] == 0]['Tenure'].mean():.2f} years."

# def get_high_credit_score_count():
#     return f"There are {df[df['CreditScore'] > 800].shape[0]} customers with a credit score above 800."

# def get_avg_balance_by_gender():
#     male_balance = df[df['Gender'].str.lower() == 'male']['Balance'].mean()
#     female_balance = df[df['Gender'].str.lower() == 'female']['Balance'].mean()
#     return (f"Male customers have an average balance of {male_balance:.2f}, "
#             f"while female customers have an average balance of {female_balance:.2f}.")

# def get_country_highest_credit_score():
#     country = df.groupby('Geography')['CreditScore'].mean().idxmax()
#     return f"The country with the highest average credit score is {country}."

# def get_churn_percentage():
#     churn_rate = df['Exited'].mean() * 100
#     return f"The churn percentage is {churn_rate:.2f}%."

# def get_salary_above_threshold(threshold):
#     count = df[df['EstimatedSalary'] > threshold].shape[0]
#     return f"There are {count} customers with a salary above {threshold}."

# def get_correlation_products_loyalty():
#     correlation = df['NumOfProducts'].corr(df['IsActiveMember'])
#     return f"The correlation between the number of products and loyalty is {correlation:.2f}."

# # Route to serve the chat page
# @app.route('/')
# def index():
#     return render_template('chat.html')

# # Chat interaction handler
# @app.route('/chat', methods=['POST'])
# def chat():
#     user_message = request.json.get('message', '').lower().strip()

#     # Greeting
#     if any(greet in user_message for greet in ["hello", "hi", "hey"]):
#         return jsonify({'response': random.choice(greetings)})

#     # Farewell
#     if any(bye in user_message for bye in ["bye", "goodbye", "see you", "take care"]):
#         return jsonify({'response': random.choice(farewells)})

#     # Intent Matching
#     if "average number of products" in user_message:
#         response = get_avg_products()
#     elif "how many active customers" in user_message:
#         response = get_active_customers()
#     elif "average tenure of customers who have not exited" in user_message:
#         response = get_avg_tenure_non_exited()
#     elif "credit score above 800" in user_message:
#         response = get_high_credit_score_count()
#     elif "higher balances male or female" in user_message or "gender-based balance" in user_message:
#         response = get_avg_balance_by_gender()
#     elif "country with highest credit score" in user_message:
#         response = get_country_highest_credit_score()
#     elif "churn percentage" in user_message:
#         response = get_churn_percentage()
#     elif "salary above" in user_message:
#         try:
#             threshold = float(user_message.split()[-1])
#             response = get_salary_above_threshold(threshold)
#         except ValueError:
#             response = "Please provide a valid number for the salary threshold."
#     elif "correlation between products and loyalty" in user_message:
#         response = get_correlation_products_loyalty()
#     else:
#         # Fallback for unmatched queries
#         response = random.choice(fallbacks)

#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template, request, jsonify
import joblib
import random
import pandas as pd

app = Flask(__name__)

# Load models and dataset
ensemble_model = joblib.load('ensemble_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
df = pd.read_csv('churn_Modelling.csv')  # Ensure this file exists

# Predefined responses
greetings = ["Hello! How can I assist you today?", "Hi! What would you like to know?", "Hey! Ask me anything."]
farewells = ["Goodbye! Have a great day!", "See you later!", "Take care!"]
fallbacks = ["I'm not sure about that. Could you rephrase?", "I didn't get that. Can you try again?"]

# Intent-based responses
def get_avg_products():
    return f"The average number of products per customer is {df['NumOfProducts'].mean():.2f}."

def get_active_customers():
    return f"There are {df[df['IsActiveMember'] == 1].shape[0]} active customers."

def get_avg_tenure_non_exited():
    return f"The average tenure of non-exited customers is {df[df['Exited'] == 0]['Tenure'].mean():.2f} years."

def get_high_credit_score_count(threshold):
    """Returns the count of customers with a credit score above the given threshold."""
    count = df[df['CreditScore'] > threshold].shape[0]
    return f"There are {count} customers with a credit score above {threshold}."

def get_avg_balance_by_gender():
    male_balance = df[df['Gender'].str.lower() == 'male']['Balance'].mean()
    female_balance = df[df['Gender'].str.lower() == 'female']['Balance'].mean()
    return (f"Male customers have an average balance of {male_balance:.2f}, "
            f"while female customers have an average balance of {female_balance:.2f}.")

def get_country_highest_credit_score():
    country = df.groupby('Geography')['CreditScore'].mean().idxmax()
    return f"The country with the highest average credit score is {country}."

def get_churn_percentage():
    churn_rate = df['Exited'].mean() * 100
    return f"The churn percentage is {churn_rate:.2f}%."

def get_salary_above_threshold(threshold):
    count = df[df['EstimatedSalary'] > threshold].shape[0]
    return f"There are {count} customers with a salary above {threshold}."

def get_correlation_products_loyalty():
    correlation = df['NumOfProducts'].corr(df['IsActiveMember'])
    return f"The correlation between the number of products and loyalty is {correlation:.2f}."

# Route to serve the chat page
@app.route('/')
def index():
    return render_template('chat.html')

# Chat interaction handler
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').lower().strip()

    # Greeting
    if any(greet in user_message for greet in ["hello", "hi", "hey"]):
        return jsonify({'response': random.choice(greetings)})

    # Farewell
    if any(bye in user_message for bye in ["bye", "goodbye", "see you", "take care"]):
        return jsonify({'response': random.choice(farewells)})

    # Intent Matching
    if "average number of products" in user_message:
        response = get_avg_products()
    elif "how many active customers" in user_message:
        response = get_active_customers()
    elif "average tenure of customers who have not exited" in user_message:
        response = get_avg_tenure_non_exited()
    elif "credit score above" in user_message:
        try:
            threshold = int(user_message.split()[-1])
            response = get_high_credit_score_count(threshold)
        except ValueError:
            response = "Please provide a valid number for the credit score threshold."
    elif "higher balances male or female" in user_message or "gender-based balance" in user_message:
        response = get_avg_balance_by_gender()
    elif "country with highest credit score" in user_message:
        response = get_country_highest_credit_score()
    elif "churn percentage" in user_message:
        response = get_churn_percentage()
    elif "salary above" in user_message:
        try:
            threshold = float(user_message.split()[-1])
            response = get_salary_above_threshold(threshold)
        except ValueError:
            response = "Please provide a valid number for the salary threshold."
    elif "correlation between products and loyalty" in user_message:
        response = get_correlation_products_loyalty()
    else:
        # Fallback for unmatched queries
        response = random.choice(fallbacks)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
