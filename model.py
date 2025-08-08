import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from scipy.sparse import vstack
feedback_data = []

def clean_text(text):
    text = text.lower()

    # remove hashtags, links
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)

    # remove spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

import pandas as pd

# upload CSV
df = pd.read_csv('C:/Users/Nextixt/Documents/tweet_emotions.csv')

# clean content
df['clean_content'] = df['content'].apply(clean_text)

# delete NaN
df = df.dropna()
# delete tweet_id
df = df.drop(columns = 'tweet_id')
df = df[df['sentiment'] != 'empty']
df = df.drop(columns = 'content')

# splitting our dataset
X = df['clean_content']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
# defining vectorizer
vectorizer = TfidfVectorizer(
    max_features=30000,            
    ngram_range=(1, 2),           
    strip_accents='unicode',       
    lowercase=True
)
# transforming data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# defining RFC
RFC = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
# fitting model
RFC.fit(X_train_vec, y_train)

# predicting and reciving accuracy
y_pred = RFC.predict(X_test_vec)

def explain_and_learn(text, vectorizer, RFC, top_k=5):
    # vectorizing
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    
    # most important things
    feature_names = vectorizer.get_feature_names_out()
    importances = RFC.feature_importances_
    nonzero_indices = vec.nonzero()[1]
    words_with_scores = [(feature_names[i], importances[i]) for i in nonzero_indices]
    sorted_words = sorted(words_with_scores, key=lambda x: x[1], reverse=True)
    top_words = sorted_words[:top_k]
    
    # Predicting
    pred = RFC.predict(vec)[0]
    print(f"\n📍 Predicted emotion: {pred}")

    # Explanation
    print("\n🧠 Explanation:")
    for word, score in top_words:
        print(f"{word}: importance = {score:.4f}")

    # Feedback
    feedback = input("✅ Is this correct? (y/n): ").strip().lower()

    if feedback == 'y':
        print("🎉 Great! Model was correct.")
    elif feedback == 'n':
        true_label = input("👉 What was the correct emotion? ").strip().lower()

        
        global X_train_vec, y_train

        X_train_vec = vstack([X_train_vec, vec])

        y_train = pd.concat([y_train, pd.Series([true_label])], ignore_index=True)

        # Retrain model
        RFC.fit(X_train_vec, y_train)
        print("🔁 Model has been updated with your correction.")
    else:
        print("❓ Invalid feedback. Skipping...")
while True:
    text = input("\n📝 Enter your text (or 'exit'): ")
    if text.strip().lower() == "exit":
        break

    explain_and_learn(text, vectorizer, RFC)




