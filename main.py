import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

print("Welcome To Language Identification With Multilingual Machine By AKKIREDDY CHANDINI\n")


def language_detection():
    data = pd.read_csv("dataset.csv")
    X = np.array(data["Text"])
    y = np.array(data["language"])

    cv = CountVectorizer()
    X_vec = cv.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.33, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("Accuracy:", model.score(X_test, y_test))

    text = input("Enter text: ")
    text_vec = cv.transform([text])
    print("Detected Language:", model.predict(text_vec)[0])


def english_to_korean():
    data = pd.read_csv("EK.csv")                  # was correct
    X = np.array(data["english"])
    y = np.array(data["korean"])
    model, cv, le = train_translation_model(X, y)
    predict_translation(model, cv, le, "Korean")


def english_to_portuguese():
    data = pd.read_csv("EP.csv")                  # FIX: was read_excel("EPO.xlsx")
    X = np.array(data["english"])                 # FIX: was data["English"]
    y = np.array(data["portuguese"])              # FIX: was data["Portuguese"]
    model, cv, le = train_translation_model(X, y)
    predict_translation(model, cv, le, "Portuguese")


def english_to_french():
    data = pd.read_csv("EF.csv")
    X = np.array(data["english"])                 # FIX: was data["English"]
    y = np.array(data["french"])                  # FIX: was data["French"]
    model, cv, le = train_translation_model(X, y)
    predict_translation(model, cv, le, "French")


def english_to_urdu():
    data = pd.read_csv("EU.csv")                  # FIX: was read_excel("EU.xlsx")
    X = np.array(data["english"])                 # FIX: was data["English"]
    y = np.array(data["urdu"])                    # FIX: was data["Urdu"]
    model, cv, le = train_translation_model(X, y)
    predict_translation(model, cv, le, "Urdu")


def train_translation_model(X, y):
    y = np.where(pd.isna(y), "Unknown", y)        # FIX: moved here so ALL languages benefit

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, _, y_train, _ = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )

    cv = CountVectorizer()
    X_train_vec = cv.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model, cv, le


def predict_translation(model, cv, le, lang):
    text = input(f"Enter English sentence for {lang}: ")
    text_vec = cv.transform([text])
    pred = model.predict(text_vec)[0]
    result = le.inverse_transform([pred])[0]
    print(f"Translated ({lang}): {result}")


while True:
    print("""
================ MENU ================
A - Language Detection
B - English to Korean
C - English to Portuguese
D - English to French
E - English to Urdu
F - Exit
""")
    choice = input("Enter choice: ").upper()

    try:
        if choice == 'A':
            language_detection()
        elif choice == 'B':
            english_to_korean()
        elif choice == 'C':
            english_to_portuguese()
        elif choice == 'D':
            english_to_french()
        elif choice == 'E':
            english_to_urdu()
        elif choice == 'F':
            print("Exiting... ధన్యవాదాలు 🙏")
            break
        else:
            print("Invalid choice. Try again.")
    except Exception as e:
        print("\nERROR OCCURRED:")
        print(e)
