import tkinter as tk
from tkinter import ttk
from tweeter_sentiment_analysis import analyze_sentiment,top_ten_frequent_word

words = top_ten_frequent_word()
print(words)
def submit():
    # Function to handle form submission
    print("Form submitted")
    keyword = name_entry.get()
    print("Selected option:", dropdown_var.get())
    analyze_sentiment(keyword)

# Create main window
root = tk.Tk()
root.title("Form")

# Create and pack a frame for the form fields
form_frame = tk.Frame(root)
form_frame.pack(padx=20, pady=20)

# Dropdown menu
dropdown_label = tk.Label(form_frame, text="Dropdown:")
dropdown_label.grid(row=4, column=0, sticky="w")
dropdown_var = tk.StringVar(root)
dropdown = ttk.Combobox(form_frame, textvariable=dropdown_var, state="readonly")
dropdown['values'] = ('Option 1', 'Option 2', 'Option 3')
dropdown.grid(row=0, column=1)
dropdown.current(0)

# Name field
name_label = tk.Label(form_frame, text="Name:")
name_label.grid(row=0, column=0, sticky="w")
name_label = tk.Label(form_frame, text=words)
name_label.grid(row=1, column=0, sticky="w")
name_entry = tk.Entry(form_frame)
name_entry.grid(row=4, column=1)

# Submit button
submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.pack()

root.mainloop()
