# Iris Classification Project (KNN)

This GitHub-ready project contains the full **Iris Classification** workflow using **KNN**, including evaluation using **accuracy, confusion matrix, precision, recall, and F1-score**.

---

##  Project Structure
```
Iris-Classification-Project/
│
├── README.md
├── iris_classification.ipynb
├── Iris.csv
└── requirements.txt
```

---

##  Overview
This project demonstrates supervised machine learning on the **Iris flower dataset**. The algorithm used is:

- **K-Nearest Neighbors (KNN)**


Evaluation includes:
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score
- Classification Report

---

##  How to Run
1. Upload all files to your GitHub repo.
2. Open the notebook in Google Colab.
3. Upload `Iris.csv` in Colab.
4. Run all cells.

---

##  Google Colab Code (Notebook Content)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("Iris.csv")

if "Id" in df.columns:
    df = df.drop("Id", axis=1)

X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- KNN Model ----
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred_knn)

print("KNN Accuracy:", knn_acc)

# Confusion Matrix (KNN)
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, cmap="Blues", fmt="d")
plt.title("KNN Confusion Matrix")
plt.show()

# Classification Report (KNN)
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

```

---

##  requirements.txt
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

##  Author
**Charvi Sharma**

---

