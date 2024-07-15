import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox
from tkinter import font as tkfont

# Load dataset
dataset = pd.read_csv('C:/Users/Ajay kumar/Downloads/Diabetes.csv')

# Preprocessing
x = dataset.drop(columns='Outcome', axis=1)
y = dataset['Outcome']
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Check accuracy
train_pred = classifier.predict(x_train)
train_acc = accuracy_score(y_train, train_pred)
print("Training accuracy:", train_acc)

test_pred = classifier.predict(x_test)
test_acc = accuracy_score(y_test, test_pred)
print("Test accuracy:", test_acc)

# Prediction function
def predict_diabetes(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(std_data)
    return 'Diabetic' if prediction[0] else 'Not Diabetic'

# GUI function
def show_prediction():
    try:
        input_data = [float(entry.get()) for entry in entries]
        result = predict_diabetes(input_data)
        result_label.config(text=f'Result: The person is {result}', fg='green' if result == 'Not Diabetic' else 'red')
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numeric values")

# Create main window
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("500x600")
root.configure(bg='#f0f8ff')

# Title and font
title_font = tkfont.Font(family='Helvetica', size=18, weight='bold')
label_font = tkfont.Font(family='Helvetica', size=12)

# Title label
title_label = tk.Label(root, text="Diabetes Prediction App", font=title_font, bg='#f0f8ff')
title_label.pack(pady=20)

# Input frame
input_frame = tk.Frame(root, bg='#f0f8ff')
input_frame.pack(pady=10)

# Input labels and entries
labels = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
entries = []

for i, label in enumerate(labels):
    tk.Label(input_frame, text=label, font=label_font, bg='#f0f8ff').grid(row=i, column=0, padx=10, pady=5, sticky='w')
    entry = tk.Entry(input_frame, font=label_font, width=20)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

# Predict button
predict_button = tk.Button(root, text='Predict', font=label_font, command=show_prediction, bg='#87cefa', width=20)
predict_button.pack(pady=20)

# Result label
result_label = tk.Label(root, text='', font=label_font, bg='#f0f8ff')
result_label.pack(pady=10)

root.mainloop()
