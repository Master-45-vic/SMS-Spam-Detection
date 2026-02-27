# 📩 Spam Detection using Google Word2Vec + XGBoost

This project is a spam classification web app built using:

- Google Pretrained Word2Vec (300d)
- Average Word Embeddings
- XGBoost Classifier
- Streamlit Web Interface

---

## 🚀 Features

- Semantic word embeddings using pretrained Word2Vec
- XGBoost model for classification
- Clean modular architecture
- Confidence score display
- Streamlit-based UI

---

## 🧠 Model Pipeline

1. Text preprocessing (tokenization, stopword removal, lemmatization)
2. Convert words → Average Word2Vec embedding
3. XGBoost prediction
4. Spam / Ham output

---

## 📈 Model Performance

Accuracy: 96%

Precision: 98%

Recall: 98%

F1 Score: 98%

---

## 📂 Project Structure


├── SMS Spam Classification.ipynb 
├── SMSSpamCollection.txt   
├── app.py                  
├── embeddings.py            
├── google_w2v_model.pkl     
├── model.py                 
├── preprocessing.py         
└── requirements.txt         

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Master-45-vic/SMS-Spam-Detection-.git
cd SMS-Spam-Detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download Google Word2Vec model

Download the pretrained GoogleNews Word2Vec model separately
and place it in the project root folder:

GoogleNews-vectors-negative300.bin

(Note: File is ~1.5GB and not included in repository)

### 4️⃣ Run the application

```bash
streamlit run app.py
```

## 📊 Dataset

SMS Spam Collection Dataset from Kaggle.

## 👨‍💻 Author

Prashanth M

⭐ If you like this project

Give it a star on GitHub!






