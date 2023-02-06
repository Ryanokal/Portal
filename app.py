import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect
from Prediction import predict

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    conn = sqlite3.connect("Login.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    c.close()
    conn.close()
    if user:
        return redirect("/dashboard")
    else:
        return render_template("index.html", error="Wrong username or password.")


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    conn = sqlite3.connect('Students.db')
    c = conn.cursor()
    c.execute("SELECT name FROM students WHERE name=?", (query,))
    result = c.fetchone()
    c.close()
    conn.close()
    if result:
        #Predict the result of the student here - either drop out or graduating - use database
        result = predict(np.array([]))
        
        #Using Progressive Bar

        return redirect("/dashboard?student_name=" + result[0])
    else:
        return redirect("/dashboard?student_name=No student found")


if __name__ == '__main__':
    app.run()
