import pandas as pd
from pathlib import Path
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 📂 Load Data
data_dir = Path(r"C:\drive D\coincent\assignment_2\text_classification\data")
file_paths = sorted(data_dir.glob("full-*.parquet"))
df = pd.concat([pd.read_parquet(fp) for fp in file_paths], ignore_index=True)

print("Total records:", len(df))
print(df.head())

# 🏷️ Filter & Label Sentiment
df = df[['title', 'average_rating']].dropna()

def label_sentiment(rating):
    if rating >= 4.0:
        return "positive"
    elif rating <= 2.0:
        return "negative"
    else:
        return "neutral"

df['sentiment'] = df['average_rating'].apply(label_sentiment)
df = df[df['sentiment'] != 'neutral']
print(df['sentiment'].value_counts())

# 🧹 Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

df['clean_title'] = df['title'].apply(preprocess_text)

# 🧠 Feature Extraction
X = df['clean_title']
y = df['sentiment']

vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# 🔀 Split Data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 🤖 Train Model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# 📊 Evaluate
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# 🧪 Detailed Metrics
y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
