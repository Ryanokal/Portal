import sqlite3

conn = sqlite3.connect('Students.db')
c = conn.cursor()

c.execute('''CREATE TABLE students (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 age INTEGER NOT NULL,
                 address TEXT NOT NULL
             );''')

conn.commit()
conn.close()
