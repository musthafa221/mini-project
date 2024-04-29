import tkinter as tk
from tkinter import ttk
from ipl_team_analysis_model import perform_engagement_analysis,perform_hashtag_analysis,perform_sentiment_analysis,perform_tweet_activity_analysis,perform_word_cloud_analysis


# Define function to handle analysis selection
def select_analysis():
    selected_analysis = analysis_var.get()
    if selected_analysis == 'Sentiment Analysis':
        perform_sentiment_analysis()
    elif selected_analysis == 'Engagement Analysis':
        perform_engagement_analysis()
    elif selected_analysis == 'Tweet Activity Analysis':
        perform_tweet_activity_analysis()
    elif selected_analysis == 'Word Cloud Analysis':
        perform_word_cloud_analysis()
    elif selected_analysis == 'Hashtag Analysis':
        perform_hashtag_analysis()

# Create Tkinter window
window = tk.Tk()
window.title("Analysis Selection")

# Define options for analysis selection
analysis_options = ['Sentiment Analysis', 'Engagement Analysis', 'Tweet Activity Analysis', 'Word Cloud Analysis', 'Hashtag Analysis']

# Create a Tkinter variable to store selected analysis
analysis_var = tk.StringVar(window)
analysis_var.set(analysis_options[0])  # Set default value

# Create dropdown menu for analysis selection
analysis_dropdown = ttk.Combobox(window, textvariable=analysis_var, values=analysis_options, state="readonly")
analysis_dropdown.pack(pady=10)

# Create button to trigger analysis selection
analyze_button = tk.Button(window, text="Analyze", command=select_analysis)
analyze_button.pack(pady=5)

# Run the Tkinter event loop
window.mainloop()
