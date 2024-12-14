import requests

def chatbot():
    print("Welcome to the AI Chatbot for Churn Prediction!")

    while True:
        try:
            # Collect user input
            credit_score = float(input("Enter credit score (or type 'exit' to quit): "))
            age = int(input("Enter age: "))
            balance = float(input("Enter balance: "))
            estimated_salary = float(input("Enter estimated salary: "))
            num_of_products = int(input("Enter number of products: "))
            geography = input("Enter geography (e.g., France, Spain, Germany): ")
            gender = input("Enter gender (Male/Female): ")
            has_cr_card = int(input("Has credit card? (1 for Yes, 0 for No): "))
            is_active_member = int(input("Is active member? (1 for Yes, 0 for No): "))

            # Create the input data
            input_data = {
                'CreditScore': credit_score,
                'Age': age,
                'Balance': balance,
                'EstimatedSalary': estimated_salary,
                'NumOfProducts': num_of_products,
                'Geography': geography,
                'Gender': gender,
                'HasCrCard': has_cr_card,
                'IsActiveMember': is_active_member
            }

            # Send request to the Flask server
            response = requests.post('http://127.0.0.1:5000/predict', json=input_data)

            if response.status_code == 200:
                print("Predictions:", response.json())
            else:
                print("Error: Could not get a prediction. Please try again.")

        except ValueError:
            print("Invalid input. Please enter the correct type of data.")
        except requests.exceptions.RequestException as e:
            print("Error communicating with the server:", e)
        
        # Option to continue or exit
        continue_chat = input("Do you want to make another prediction? (yes/no): ").strip().lower()
        if continue_chat != 'yes':
            print("Thank you for using the AI Chatbot for Churn Prediction!")
            break

if __name__ == '__main__':
    chatbot()
