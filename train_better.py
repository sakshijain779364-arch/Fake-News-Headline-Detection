import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("📂 Loading dataset...")
df = pd.read_csv("english_fake_news_2212_numeric.csv")

# Combine headline + body_text for better features
df['text'] = df['headline'] + " " + df['body_text']

X = df["text"]
y = df["label"]

print(f"✅ Dataset loaded: {len(df)} rows")
print(f"   - Real (0): {(y == 0).sum()}")
print(f"   - Fake (1): {(y == 1).sum()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n📊 Training set: {len(X_train)}")
print(f"📊 Test set: {len(X_test)}")

# Improved TF-IDF
print("\n🔧 Creating advanced vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english",
    ngram_range=(1, 3),          # Trigrams for better context
    min_df=3,
    max_df=0.7,
    sublinear_tf=True,
    strip_accents='unicode'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✅ Vectorizer created: {X_train_vec.shape[1]} features")

# Try Random Forest (better for text)
print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)

# Evaluate
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

# Cross-validation
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

# Save
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")

print(f"\n✅ Better model trained and saved!")
