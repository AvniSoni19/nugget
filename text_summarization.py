import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from transformers import BartForConditionalGeneration, BartTokenizer

# # Download NLTK data files
# nltk.download('punkt')
# nltk.download('stopwords')

def extractive_summary(text, num_sentences=5):
    stop_words = set(stopwords.words('english') + list(punctuation))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]

    freq_dist = nltk.FreqDist(filtered_words)
    sentences = sent_tokenize(text)

    sentence_scores = {sent: sum(freq_dist[word] for word in word_tokenize(sent.lower()) if word in freq_dist) for sent in sentences}
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    return ' '.join(summarized_sentences)

text = """Two youths died and a woman was injured on May 29 after a vehicle in the convoy of the Bharatiya Janata Party (BJP) Lok Sabha candidate from Kaiserganj,
 Karan Bhushan Singh, ran over them near Baikunth Degree College on Colonelganj-Huzoorpur road in Gonda district. The local police seized the vehicle and 
 taken the bodies into custody for postmortem. It is not known whether Mr. Singh was in the vehicle or convoy or not.
The injured woman has been admitted to hospital and police personnel have been deployed in the area to ensure law and order."""

extracted_summary = extractive_summary(text)
print("Extracted Summary:", extracted_summary)


model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def abstractive_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

abstractive_summ = abstractive_summary(extracted_summary)
print("Abstractive Summary:", abstractive_summ)
