import joblib

def predict(data):
    tree_clf = joblib.load("student_model.sav")
    return tree_clf.predict(data)

