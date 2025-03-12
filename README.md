
# 📊 Churn Modeling Project

This project focuses on predicting customer churn using a machine learning model. Customer churn refers to the percentage of customers that stop using a company's services during a specific time frame. Accurately predicting churn helps businesses enhance customer retention strategies and optimize their operations.

---

## 🚀 Features
- 📂 **Dataset Used:** Structured dataset containing customer demographics, account information, and usage statistics.
- 🤖 **Modeling Techniques:** Various machine learning algorithms applied to build a robust churn prediction model.
- 📈 **Evaluation Metrics:** Performance measured using Accuracy, Precision, Recall, and F1-score.
- 📊 **Data Visualization:** Performed Exploratory Data Analysis (EDA) to identify patterns and relationships in the data.
- 💬 **Chatbot Integration:** An AI chatbot interface for interacting with the churn model.

---

## 📂 Project Structure
```
├── Churn_Modelling.csv            # Dataset file
├── FIMIS_Research Paper.docx       # Research Paper related to the project
├── app.py                          # Main application file (Flask/Backend)
├── chat.html                       # HTML file for chatbot UI
├── chatbot.py                      # Chatbot script
├── fimis_model.py                  # Model training and prediction script
├── flask_app.py                    # Flask app for serving the model
├── requirements.txt                # Dependencies list
├── README.md                       # Project documentation (This file)
```

---

## 🔧 Installation
1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Churn_Modelling.git
```

2. **Navigate to the project directory:**
```bash
cd Churn_Modelling
```

3. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📌 Usage

### Data Preprocessing & Model Training
Run the model training script to prepare and train the model:
```bash
python fimis_model.py
```

### Running the Flask App
Start the Flask application:
```bash
python flask_app.py
```
The app will run at `http://127.0.0.1:5000/`.

### Chatbot Interaction
To interact with the model via chatbot:
- Run `chatbot.py` if it's a separate script.
- Open `chat.html` in your browser if it's served via `flask_app.py`.

---

## 🛠️ Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Flask, Joblib
- **Frontend:** HTML (for chatbot interface)
- **Platform:** Localhost / Google Colab Notebook (For experimentation and training)

---

## 💡 Future Work
- 🔨 Improve model performance through feature engineering and hyperparameter tuning.
- 🌟 Integrate more advanced machine learning techniques like ensemble methods or deep learning.
- 🌐 Enhance chatbot interactions and deploy the model online.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.
