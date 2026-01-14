# Credit Wise Loan System
📌 Project Overview

The Credit Wise Loan System is a Machine Learning–based web application designed to predict loan approval outcomes based on applicant financial and demographic information.
The system helps simulate real-world loan eligibility checks using supervised learning models and provides an interactive interface built with Streamlit.

🚀 Features

User-friendly web interface using Streamlit

Handles both numerical and categorical applicant data

Compares multiple Machine Learning models

Displays prediction results in real time

End-to-end ML pipeline (preprocessing → modeling → deployment)

🧠 Machine Learning Models Used

Logistic Regression

K-Nearest Neighbors (KNN)

Naive Bayes ✅ (Best performing model)

Models were evaluated using:

Accuracy

Precision

Recall

🛠️ Tech Stack

Programming Language: Python

Libraries & Frameworks:

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

Pickle

Streamlit

🌐 Live Demo

👉 Streamlit App:
https://creditwiseloansystem-bhu809.streamlit.app/

📂 Project Structure

```

├── requirements.txt
├── model
    └── loan_pipeline.pkl
├── anaconda_projects
    └── db
    │   └── project_filebrowser.db
├── .gitignore
├── src
    └── train_model.py
├── app.py
└── data
    └── loan_approval_data.csv

```
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/BhupatiNadar/Credit_Wise_LoanSystem.git
cd Credit_Wise_LoanSystem
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
streamlit run app.py

📈 Model Workflow

Data Collection

Data Preprocessing (Missing values, encoding, scaling)

Model Training

Model Evaluation

Best Model Selection (Naive Bayes)

Model Serialization using Pickle

Deployment with Streamlit

🎯 Learning Outcomes

Built an end-to-end ML project

Gained hands-on experience with multiple classification models

Learned model comparison and evaluation techniques

Deployed an ML application using Streamlit

🔮 Future Enhancements

Add more advanced models (Random Forest, XGBoost)

Improve UI/UX

Add model explainability (SHAP / feature importance)

Connect to a real-time database

🤝 Connect

If you have suggestions, feedback, or collaboration ideas, feel free to connect.

Author: Bhupati Nadar
💻 GitHub Repository

👉 https://github.com/BhupatiNadar/Credit_Wise_LoanSystem.git
