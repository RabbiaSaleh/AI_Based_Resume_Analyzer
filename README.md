# 📄 Resume Analyzer

### 🔍 An intelligent system to evaluate and screen resumes using Machine Learning

---

## 🚀 Project Overview

**Resume Analyzer** is a smart web application designed to automate the resume screening process. It helps recruiters or HR professionals by analyzing uploaded resumes (PDF or DOCX formats) and evaluating them against predefined criteria using machine learning models.

This application leverages:
- Natural Language Processing (NLP) for text extraction and feature analysis.
- A trained **Random Forest classifier** for candidate suitability scoring.
- A user-friendly **Flask-based web interface** for interaction.
- SQLite database integration to store analyzed results.

---

## 🧠 Features

- ✅ Upload and analyze resumes (PDF/DOCX)  
- ✅ Extract and vectorize text using TF-IDF  
- ✅ Predict candidate relevance using ML  
- ✅ Store and view results in a database  
- ✅ Stylish HTML interface using Flask templates  

---

## 📂 Project Structure

- resume_analyzer/
- ├── app.py # Main Flask app
- ├── db.py # Database functions (SQLite)
- ├── model/
- │ ├── rf_model.pkl # Pre-trained Random Forest model
- │ └── tfidf_vectorizer.pkl # TF-IDF vectorizer
- ├── static/ # CSS, JS, image assets
- ├── templates/ # HTML templates (Jinja2)
- ├── view_db.py # View analyzed results from DB
- └── .git/ # Git repo metadata


---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/resume_analyzer.git
   cd resume_analyzer

  ---
  
## 🧪 Tech Stack

- Layer:	Tech Used
- Frontend:	HTML, CSS, Bootstrap
- Backend:	Python, Flask
- Machine Learning:	scikit-learn, joblib
- Database:	SQLite
- NLP Tools:	spaCy, TF-IDF

---

## 📊 Example Use-Case

- 1. Upload a resume from your local machine.
- 2. The app processes and extracts text.
- 3. It utilizes machine learning to score the resume based on relevance.
- 4. The result is saved to the database and shown in a dashboard.

----

## 📷 Live Link

https://RabbiaSaleh.github.io/AI_Based_Resume_Analyzer/

---

## 🤖 Model Info

- 1. Trained using Random Forest Classifier
- 2. Text vectorization with TF-IDF
- 3. Suitable for binary classification of resume relevance

---

## 🤝 Contributing

We welcome contributions!
Feel free to fork the repo, create a new branch, and submit a pull request.


---

## 📜 License

This project is licensed under the MIT License.
