

# 🤖 AutoML Pro: Enterprise-Grade Automated Machine Learning

**AutoML Pro** is a state-of-the-art Automated Machine Learning engine designed to democratize AI. It automates the complete data science lifecycle—from raw data ingestion to deployable model export—making high-performance machine learning accessible to everyone, regardless of technical expertise.

---

## 🌟 Key Features

### 🧠 Intelligent Automation

* **Auto-Detection**: Automatically identifies if your problem is a **Classification** or **Regression** task based on your target variable.
* **Smart Preprocessing**: Handles missing values, encodes categorical variables, scales numeric features, and removes high-cardinality columns without user intervention.
* **Imbalance Handling**: Automatically detects class imbalance and adjusts class weights to ensure fair predictions.

### 🏎️ multiple Optimization Modes

* **⚡ Fast Mode**: (Tests ~4 models) Ideal for rapid prototyping and large datasets.
* **⚖️ Balanced Mode**: (Tests ~6 models) The perfect trade-off between speed and accuracy.
* **🎯 Precise Mode**: (Tests ~8 models) Runs exhaustive searches including computationally expensive models like SVMs for maximum accuracy.

### 🏆 Advanced Modeling

* **Smart Ensembles**: Automatically identifies the top 3 performing models and combines them into a **Voting Classifier/Regressor** to boost performance by 2-5%.
* **Hyperparameter Tuning**: Includes a dedicated module for fine-tuning specific algorithms using Grid Search with Cross-Validation.

### 📊 Rich Analytics

* **Interactive Dashboards**: Powered by **Plotly**, offering dynamic charts for model comparison, feature importance, and confusion matrices.
* **Comprehensive Metrics**: Evaluates models on Accuracy, F1-Score, R², MAE, MSE, and RMSE.
* **Downloadable Reports**: Export detailed performance reports and trained models for production use.

---

## 🛠️ Tech Stack

This project is built using a robust stack of modern data science libraries:

| Component | Technology | Purpose |
| --- | --- | --- |
| **Frontend** | **Streamlit** | Interactive web dashboard and UI components. |
| **Core Engine** | **Python 3.10+** | Primary programming language. |
| **ML Backend** | **Scikit-Learn** | Model training, preprocessing, and evaluation metrics. |
| **Data Handling** | **Pandas & NumPy** | High-performance data manipulation and analysis. |
| **Visualization** | **Plotly Express** | Interactive charts, radar plots, and heatmaps. |
| **Serialization** | **Pickle** | Saving and loading trained models and preprocessors. |

---

## 📂 Project Structure

The project follows a modular **Clean Architecture** pattern to ensure maintainability and scalability.

```text
AutoML-Pro/
│
├── automl_engine.py          # THE BRAIN: Core ML logic, preprocessing, and training pipelines.
├── app.py                    # THE FACE: Streamlit dashboard and UI logic.
├── requirements.txt          # List of python dependencies.
└── README.md                 # Project documentation.

```

* **`automl_engine.py`**: Contains classes for `DataAnalyzer`, `DataPreprocessor`, `ModelFactory`, and the main `AutoML` orchestrator. It handles all the "heavy lifting" of machine learning.
* **`app.py`**: Handles user interaction, file uploads, session state management, and renders the visualizations.

---

## 🚀 Installation & Setup

Follow these steps to run the application locally:

### 1. Clone the Repository

```bash
git clone https://github.com/Arjunyadavsoma/AI_Algorithm_Selector.git
cd automl-pro

```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Application

```bash
streamlit run app.py

```

---

## 📖 Usage Guide

### Step 1: Upload Data

* Launch the app and navigate to the sidebar.
* Upload your dataset in **CSV format**.
* *Tip: The app automatically samples large datasets (>50k rows) to ensure performance.*

### Step 2: Configure Training

* **Select Target**: Choose the column you want to predict from the dropdown.
* **Select Mode**: Choose between **Fast**, **Balanced**, or **Precise**.
* **Advanced Settings**: Optionally adjust the test set split size (default is 20%).

### Step 3: Train & Evaluate

* Click **🚀 Start AutoML Training**.
* Watch the real-time progress bar as the engine trains, evaluates, and ranks models.
* Once complete, explore the **Leaderboard** to see which algorithm won.

### Step 4: Visualize & Export

* Use the tabs to view **Confusion Matrices**, **Feature Importance**, and **Prediction distributions**.
* Go to the **Export** tab to download your trained model (`.pkl`) and a CSV report of the results.

---

## 🧪 Supported Algorithms

The engine automatically selects the appropriate algorithms based on your problem type:

### 🔵 Classification

* Logistic Regression
* Random Forest Classifier
* Decision Tree Classifier
* Gradient Boosting (GBM)
* Support Vector Machines (SVM)
* K-Nearest Neighbors (KNN)
* Naive Bayes
* **🏆 Smart Voting Ensemble**

### 🔴 Regression

* Linear Regression
* Ridge / Lasso Regression
* Random Forest Regressor
* Decision Tree Regressor
* Gradient Boosting Regressor
* Support Vector Regressor (SVR)
* **🏆 Smart Voting Ensemble**

---



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

**Author:** Arjun Yadav

**Version:** 3.0.0
