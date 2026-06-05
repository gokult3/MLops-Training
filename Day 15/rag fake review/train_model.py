import pandas as pd

df = pd.read_csv("reviews.csv")

print(df.head())
print(df.shape)
import pandas as pd
import re

df = pd.read_csv("reviews.csv")

df = df.dropna()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df["review_text"] = df["review_text"].apply(clean_text)

print(df.head())
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("reviews.csv")

X = df["review_text"]
y = df["label"]

vectorizer = TfidfVectorizer(max_features=5000)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

pickle.dump(model, open("fake_review_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))