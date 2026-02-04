import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the larger dataset
print("📂 Loading dataset...")
df = pd.read_csv("english_fake_news_2212_numeric.csv")

X = df["headline"]
y = df["label"]

print(f"✅ Dataset loaded: {len(df)} rows")
print(f"   - Real (0): {(y == 0).sum()}")
print(f"   - Fake (1): {(y == 1).sum()}")

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n📊 Training set: {len(X_train)}")
print(f"📊 Test set: {len(X_test)}")

# Better TF-IDF Vectorizer with optimized parameters
print("\n🔧 Creating vectorizer with better parameters...")
vectorizer = TfidfVectorizer(
    max_features=2000,           # Increased from default
    stop_words="english",
    ngram_range=(1, 2),          # Bigrams help catch patterns
    min_df=2,                    # Ignore very rare words
    max_df=0.8,                  # Ignore very common words
    sublinear_tf=True            # Better scaling
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✅ Vectorizer created: {X_train_vec.shape[1]} features")

# Train with better Logistic Regression
print("\n🤖 Training model...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',     # Handle class imbalance
    solver='lbfgs'
)
model.fit(X_train_vec, y_train)

# Evaluate on test set
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n📈 TEST SET RESULTS:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# Cross-validation for robustness
print(f"\n🔄 Cross-validation (5-fold)...")
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
print(f"   CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives (Real):  {cm[0][0]}")
print(f"   False Positives (FP):   {cm[0][1]}")
print(f"   False Negatives (FN):   {cm[1][0]}")
print(f"   True Positives (Fake):  {cm[1][1]}")

# Save files
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")

print(f"\n✅ Model and vectorizer saved!")
print(f"\n💾 Files created:")
print(f"   - vectorizer.pkl")
print(f"   - model.pkl")
