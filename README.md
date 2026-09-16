<H3>NAME: Vasanth P</H3>
<H3>REGISTER NO. 212224230295</H3>
<H3>DATE: 16-09-2026</H3>
<H1 Align="center">Project Based Experiment<H1>
<H3>Objective:<H3>
  
To perform sentiment analysis on Facebook data using Natural Language Processing (NLP) techniques and classify the feedback into Positive, Negative, and Neutral sentiments, and filter the data containing only Neutral feedback.
  
<H3>Program:</H3>
  
```
# Install required library
!pip install vaderSentiment

# Import libraries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Read the Facebook dataset
df = pd.read_csv("pseudo_facebook.csv")

# Display the first five records
print("First 5 Records:")
print(df.head())

# Create feedback based on Facebook activity
def create_feedback(row):
    if row["likes_received"] > 100:
        return "I received many likes on Facebook"
    elif row["likes_received"] < 10:
        return "I received very few likes on Facebook"
    else:
        return "I received an average number of likes on Facebook"

df["Feedback"] = df.apply(create_feedback, axis=1)

# Create sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to classify sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))

    if score["compound"] >= 0.05:
        return "Positive"
    elif score["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Feedback"].apply(get_sentiment)

# Display sentiment results
print("\nSentiment Results:")
print(df[["Feedback", "Sentiment"]].head(20))

# Count each sentiment
print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())

# Filter only neutral feedback
neutral_data = df[df["Sentiment"] == "Neutral"]

# Display neutral feedback
print("\nNeutral Feedback:")
print(neutral_data[["Feedback", "Sentiment"]].head(20))

# Display number of neutral records
print("\nNumber of Neutral Feedback:", len(neutral_data))

# Save neutral feedback into a new CSV file
neutral_data.to_csv("neutral_facebook_feedback.csv", index=False)

print("\nNeutral feedback saved successfully.")
  
```
<H3>Output:</H3>

<img width="920" height="442" alt="image" src="https://github.com/user-attachments/assets/f29b6220-b835-45bf-9c33-d2f096699d73" />
<img width="610" height="470" alt="image" src="https://github.com/user-attachments/assets/2753dd54-9218-422b-8eca-cb65102b3ec8" />
<img width="532" height="105" alt="image" src="https://github.com/user-attachments/assets/37edb722-cb03-48ba-b066-8cffafb15222" />
<img width="427" height="48" alt="image" src="https://github.com/user-attachments/assets/93a695aa-0bea-4491-a134-65f202ec5fd5" />

<H3>Inference:</H3>
This project helped me understand how sentiment analysis can be performed on Facebook data using Natural Language Processing techniques. I learned how to use the VADER sentiment analyzer to classify feedback into Positive, Negative, and Neutral categories. I also learned how to filter the dataset to obtain only Neutral feedback and save the filtered data as a separate CSV file. This experiment improved my understanding of data preprocessing, sentiment classification, and data filtering using Python.
