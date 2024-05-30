import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from transformers import BartForConditionalGeneration, BartTokenizer

text = """How to reach out for a referral?

Few basic things we can keep in mind

🔹Add a relevant subject line
‘Developer interested in joining ABC org’ works
‘Fresher seeking referral for Job Id 123’ works
‘Hi sir’ or ‘Hi ma’am’ might not work

🔹Do your Homework. (yes, you read that right)
- Check if their org is currently hiring for roles suitable for you.
- Bonus : send them the exact Job Id or link.

🔹Introduce yourself - give a 2-3 lines brief intro (professionally relevant).

🔹Attach your resume - It will save your and their time by not going back and forth asking for it. Share a PDF or a PDF link.

🔹Reach out to more & more people. But remember, your alumni have a higher chance of reverting back to you.

ps : if you are someone who is currently in a position to give referrals, feel free to add your expectations from the candidates when they reach out❤️

Follow Vikram Gaur
"""

# Extractive summarization
def extractive_summary(text, num_sentences=2):
    stop_words = set(stopwords.words('english') + list(punctuation))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]

    freq_dist = nltk.FreqDist(filtered_words)
    sentences = sent_tokenize(text)

    sentence_scores = {sent: sum(freq_dist[word] for word in word_tokenize(sent.lower()) if word in freq_dist) for sent in sentences}
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    
    return ' '.join(summarized_sentences)

# Extract key sentences
extracted_summary = extractive_summary(text)
print("Extractive Summary:", extracted_summary)

# Abstractive summarization
model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def abstractive_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=5, length_penalty=1.0, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Generate abstractive summary from extractive summary
abstractive_summ = abstractive_summary(extracted_summary)
print("Abstractive Summary:", abstractive_summ)
