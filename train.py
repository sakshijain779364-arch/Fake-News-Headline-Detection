import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("fake_news.csv")

X = df["headline"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save files
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")
print("✅ Model and vectorizer saved!")