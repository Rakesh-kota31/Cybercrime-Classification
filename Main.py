import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # need to see
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import read_excel
import joblib

# Load the SVM model and TF-IDF vectorizer
svm_model = joblib.load("svm_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    
    text = text.lower()
    
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    processed_text = ' '.join(tokens)
    
    return processed_text

# Function to classify cyberattack
def classify_attack():
    # Get input values from GUI fields
    incident = incident_entry.get()
    offender = offender_var.get()
    harm = harm_entry.get()
    victim = victim_var.get()
    year = year_var.get()
    location = location_entry.get()
    state = state_var.get()

    data = {
        'Incident': incident,
        'Offender': offender,
        'Victim': victim,
        'Harm' : harm,
        'Year': year,
        'Location': location,
        'Amount': 0,
        'State': state
        }
    # Preprocess text
    incident = preprocess_text(incident)
    offender = preprocess_text(offender)
    harm = preprocess_text(harm)
    update_label.config(text=f"Preprocessing")
    victim = preprocess_text(victim)

    # Combine text
    text = incident + ' ' + offender + ' ' + harm + ' ' + victim

    root.after(5000, apply_tfidf, text, data)

def apply_tfidf(text, data):
    # Apply TF-IDF vectorization
    vectorized_text = tfidf_vectorizer.transform([text])
    update_label.config(text=f"Applying TF-IDF")
    # Update the GUI after 1 second
    root.after(5000, apply_svm, vectorized_text, data)

def apply_svm(vectorized_text, data):
    # Apply SVM model
    prediction = svm_model.predict(vectorized_text)
    update_label.config(text=f"Applying SVM ALgorithm")
    data['Attack'] = prediction

    existing_data = pd.read_excel("Dataset.xlsx")

    df = pd.DataFrame(data)

    # Assume new_data is the DataFrame containing the data you want to append
    # Concatenate the existing data with the new data
    combined_data = pd.concat([existing_data, df], ignore_index=True)

    # Write the combined data back to the Excel file, overwriting the existing file
    with pd.ExcelWriter("Dataset.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        combined_data.to_excel(writer, index=False, sheet_name="Sheet1")
    
    # Update the GUI
    messagebox.showinfo("Prediction", f"The predicted cyberattack is: {prediction[0]}")
    update_label.config(text=f"Predicted attack: {prediction[0]}")


def classify_crimes():
    df = read_excel('Dataset.xlsx')

    cols = ['Incident', 'Offender', 'Victim', 'Harm']
    
    for column in cols:
        df[column] = df[column].fillna("")
        df[column] = df[column].apply(preprocess_text)

    X = df.drop(columns=['Attack'])
    Y = df['Attack']

    X_total = X['Incident'] + ' ' + X['Offender'] + ' ' + X['Harm'] + ' ' + X['Victim']

    X_total_tfidf = tfidf_vectorizer.transform(X_total)
    predictions_svm_total = svm_model.predict(X_total_tfidf)

    X_results_total = X.copy()
    X_results_total['Attack'] = Y
    X_results_total['Predicted Attack'] = predictions_svm_total

    return X_results_total


# Function to display attacks by year
def show_attacks_by_year():
    # Your code for attacks by year graph
    # attacks_by_year_total = ...
    X_svm_results_total = classify_crimes()

    attacks_by_year_total = X_svm_results_total.pivot_table(index='Year', columns='Predicted Attack', aggfunc='size', fill_value=0)

    attacks_by_year_total['Total Attacks'] = attacks_by_year_total.sum(axis=1)

    attacks_by_year_total = attacks_by_year_total.rename(columns={
        'Social Engineering': 'SE',
        'Industrial Espionage': 'IE',
        'ID Theft': 'ID theft',
        'Malware': 'Malware'
    })

    attacks_by_year_total.columns.name = 'Attack'
    attacks_by_year_total.index.name = 'Year'


    attacks_by_year_total.drop('Total Attacks', axis=1, inplace=True)

    colors = ['#FFD700', '#FFA07A', '#98FB98', '#87CEEB']

    ax = attacks_by_year_total.plot(kind='bar', stacked=True, color=colors)

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 0:
            ax.text(x + width / 2, y + height / 2, int(height), ha='center', va='center')

    plt.legend(title='Attack', loc='upper left', labels=['ID Theft','Industrial Espionage', 'Malware', 'Social Engineering',])

    plt.xlabel('Year')
    plt.ylabel('Number of Attacks')
    plt.title('Number of Attacks by Year')

    plt.show()
    #plt.figure(figsize=(8, 6))

# Function to display attacks by state
def show_attacks_by_state():
    # Your code for attacks by state graph
    # attacks_by_state_total = ...
    X_svm_results_total = classify_crimes()
    
    attacks_by_state_total = X_svm_results_total.pivot_table(index='State', columns='Predicted Attack', aggfunc='size', fill_value=0)

    attacks_by_state_total['Total Attacks'] = attacks_by_state_total.sum(axis=1)

    attacks_by_state_total = attacks_by_state_total.rename(columns={
        'Social Engineering': 'SE',
        'Industrial Espionage': 'IE',
        'ID Theft': 'ID theft',
        'Malware': 'Malware'
    })

    attacks_by_state_total.columns.name = 'Attack'
    attacks_by_state_total.index.name = 'Year'

    attacks_by_state_total.drop('Total Attacks', axis=1, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#FFD700', '#FFA07A', '#98FB98', '#87CEEB']

    attacks_by_state_total.plot(kind='bar', stacked=True, color=colors, ax=ax)

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 0:
            ax.text(x + width / 2, y + height / 2, int(height), ha='center', va='center')

    ax.legend(title='Attack', loc='upper left', labels=['ID Theft','Industrial Espionage', 'Malware', 'Social Engineering'])

    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Attacks')
    ax.set_title('Number of Attacks by State')

    plt.show()
    #plt.figure(figsize=(10, 6))


# Create the main window
root = tk.Tk()
root.title("Cybercrime Prediction")
root.geometry("800x600")  # Set window size

# Create a frame to contain all the widgets
frame = ttk.Frame(root, padding="20")
frame.grid(row=0, column=0, sticky="nsew")

# Incident
incident_label = ttk.Label(frame, text="Incident:", font=('Helvetica', 14))
incident_label.grid(row=0, column=0, sticky="w", pady=(10, 5))
incident_entry = ttk.Entry(frame, font=('Helvetica', 14), width=50)
incident_entry.grid(row=0, column=1, columnspan=2, pady=(10, 5))

# Offender
offender_label = ttk.Label(frame, text="Offender:", font=('Helvetica', 14))
offender_label.grid(row=1, column=0, sticky="w", pady=(5, 5))
offender_var = tk.StringVar(root)
offender_var.set("Hacker")  # Default value
offender_dropdown = ttk.Combobox(frame, textvariable=offender_var, values=["Hacker", "Criminal", "Terrorist"], font=('Helvetica', 14), state="readonly")
offender_dropdown.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(5, 5))

# Harm
harm_label = ttk.Label(frame, text="Harm:", font=('Helvetica', 14))
harm_label.grid(row=2, column=0, sticky="w", pady=(5, 5))
harm_entry = ttk.Entry(frame, font=('Helvetica', 14), width=50)
harm_entry.grid(row=2, column=1, columnspan=2, pady=(5, 5))

# Victim
victim_label = ttk.Label(frame, text="Victim:", font=('Helvetica', 14))
victim_label.grid(row=3, column=0, sticky="w", pady=(5, 5))
victim_var = tk.StringVar(root)
victim_var.set("Company")  # Default value
victim_dropdown = ttk.Combobox(frame, textvariable=victim_var, values=["Company", "Individual", "Software"], font=('Helvetica', 14), state="readonly")
victim_dropdown.grid(row=3, column=1, columnspan=2, sticky="ew", pady=(5, 5))

# Year
year_label = ttk.Label(frame, text="Year:", font=('Helvetica', 14))
year_label.grid(row=4, column=0, sticky="w", pady=(5, 5))
year_var = tk.StringVar(root)
year_var.set("2020")  # Default value
year_dropdown = ttk.Combobox(frame, textvariable=year_var, values=["2020", "2021", "2022", "2023"], font=('Helvetica', 14), state="readonly")
year_dropdown.grid(row=4, column=1, columnspan=2, sticky="ew", pady=(5, 5))

# Location
location_label = ttk.Label(frame, text="Location:", font=('Helvetica', 14))
location_label.grid(row=5, column=0, sticky="w", pady=(5, 5))
location_entry = ttk.Entry(frame, font=('Helvetica', 14), width=50)
location_entry.grid(row=5, column=1, columnspan=2, pady=(5, 5))

# State
state_label = ttk.Label(frame, text="State:", font=('Helvetica', 14))
state_label.grid(row=6, column=0, sticky="w", pady=(5, 5))
state_var = tk.StringVar(root)
state_var.set("Andhra Pradesh")  # Default value
state_dropdown = ttk.Combobox(frame, textvariable=state_var, values=["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"], font=('Helvetica', 14), state="readonly")
state_dropdown.grid(row=6, column=1, columnspan=2, sticky="ew", pady=(5, 10))


# Title label
title_label = ttk.Label(root, text="CyberCrime Classification", font=('Helvetica', 16, 'bold'))
title_label.place(relx=0.5, rely=0.02, anchor="center")

# Predict button
predict_button = ttk.Button(frame, text="Predict Cyberattack", command=classify_attack, width=20, style='my.TButton')
predict_button.grid(row=7, column=0, columnspan=3, pady=(10, 20))

# Show Graph button for Attacks by Year
show_attacks_by_year_button = ttk.Button(frame, text="Show Attacks by Year", command=show_attacks_by_year, width=30, style='my.TButton')
show_attacks_by_year_button.grid(row=8, column=0, columnspan=3, pady=(10, 10))

# Show Graph button for Attacks by State
show_attacks_by_state_button = ttk.Button(frame, text="Show Attacks by State", command=show_attacks_by_state, width=30, style='my.TButton')
show_attacks_by_state_button.grid(row=9, column=0, columnspan=3, pady=(10, 10))

# Update label
update_label = ttk.Label(root, text="", font=('Helvetica', 14))
update_label.place(relx=0.5, rely=0.95, anchor="center")

# Apply the same style to the buttons and the label
root.option_add('*TButton*font', ('Helvetica', 12))

# Define a custom style for the button to increase its size
root.tk_setPalette(background='#ececec', foreground='#333333',
                   activeBackground='#c9c9c9', activeForeground='#333333',
                   highlightBackground='#d9d9d9', highlightColor='black')

style = ttk.Style(root)
style.configure('my.TButton', font=('Helvetica', 14), padding=10)


# Run the application
root.mainloop()

