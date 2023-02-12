import joblib
import sqlite3
import numpy as np


def predict(data):
    tree_clf = joblib.load("student_model.sav")
    return tree_clf.predict(data)


def get_data_from_database(column, regNo):
    conn = sqlite3.connect("Students.db")
    cursor = conn.cursor()
    cursor.execute("SELECT " + column + " FROM students" + " WHERE RegNo = " + regNo)
    data = np.array(cursor.fetchall())
    conn.close()
    return data
