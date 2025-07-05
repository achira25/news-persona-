# 📰 Personalized News Finder

**IR+ML powered news classification and recommendation system** built using **Streamlit**, **Machine Learning**, and **Information Retrieval** techniques.

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-Enabled-red?style=flat&logo=streamlit" />
  <img src="https://img.shields.io/badge/ML-Logistic%20Regression%20|%20KNN-green" />
</div>

---

## 🔍 Features

- ✅ Load and explore the BBC News dataset
- ✅ Train and evaluate **Logistic Regression** and **K-Nearest Neighbors (KNN)** classifiers
- ✅ View category-wise performance metrics (Precision, Recall, F1-score)
- ✅ Get personalized article recommendations using:
  - Category + article selection
  - Free-text query with cosine similarity
- ✅ Interactive and visual dashboard via **Streamlit**

---

## 📁 Project Structure

```bash
.
├── .devcontainer/
│   └── devcontainer.json      # Environment configuration for GitHub Codespaces or VS Code Dev Containers
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
└── README.md                  # You're reading it!
