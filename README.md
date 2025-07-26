Shell AI Hackathon: Sustainable Fuel Blend Property Predictor
This project contains an interactive web application that predicts the properties of sustainable fuel blends using a machine learning model. It is the result of my participation in the Shell AI Hackathon for Sustainable and Efficient Energy.

The core of the project is a CatBoost model trained on a complex dataset of fuel component fractions and properties. The model leverages extensive feature engineering to capture non-linear interactions between components, ultimately predicting 10 final blend properties with high accuracy.

Live Demo
This application is deployed as a Streamlit web app.

[Link to your deployed app will go here]

Problem Statement
The challenge was to develop a model to accurately predict the final properties of complex fuel blends based on their constituent components and proportions. This is critical for accelerating the development and adoption of Sustainable Aviation Fuels (SAFs), helping to reduce the environmental impact of the aviation industry.

Features
Interactive Interface: Upload a CSV file with fuel component data to receive instant predictions.
Advanced ML Model: Utilizes a MultiOutputRegressor with CatBoost to handle the complexity of the data.
In-depth Feature Engineering: Creates over 150 new features to model chemical interactions, blend complexity, and component dominance.
Feature Importance: Visualizes the key drivers behind the model's predictions.
Downloadable Results: Export the predictions to a submission.csv file.
Technical Stack
Python: The core programming language.
Pandas & NumPy: For data manipulation and numerical operations.
Scikit-learn & CatBoost: For building and training the machine learning model.
Streamlit: For creating the interactive web application.
Joblib: For serializing and deserializing the trained model.
How to Run Locally
Clone the repository:
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the dependencies:
pip install -r requirements.txt

Download the data:
Place the train.csv file inside a dataset folder.
Train the model:
Run the training script once to create the model file.
python train_model.py

Run the Streamlit app:
streamlit run app.py

The application will open in your web browser.
