# Cleaning text steps
# 1) Create a file that takes text from it
# 2) Convert the letters all into lowercase
# 3) Remove punctuation marks like .,!? etc.

import string
from textblob import TextBlob
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
text = open('input.txt', encoding='utf-8').read()

# Sentiment Analysis using TextBlob
blob = TextBlob(text)
print(blob.sentiment)

# Cleaning text by putting all the characters to lowercase and removing punctuation
lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('','',string.punctuation))

# Tokenize words by splitting each of the words into an string array
tokenized_words = word_tokenize(cleaned_text, "english")

final_words = []
# If the word is not one of the stop words given above, we add the word into the array
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)

# NLP Emotions Analysis
# 1) Check if the word in the final word list is present in the emotion.txt
#   - Open the emotion file
#   - Loop through each line and clear it
#   - Extract the word and emotion using split

emotion_list = []

with open ('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace('\n', '').replace(',', '').replace("'",'').strip()
        word, emotion = clear_line.split(':')

        if word in final_words:
            emotion_list.append(emotion)

# Prints emotion list of words categorized by emotion
w = Counter(emotion_list)

# Use NLTK Sentiment Analyzer
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    print(score)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        print("Negative sentiment")
    elif pos > neg:
        print("Positive sentiment")
    else:
        print("Neutral vibe")

sentiment_analyse(cleaned_text)

# After analyzing all the emotions, system will generate a graph of these emotions created
fig , ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()