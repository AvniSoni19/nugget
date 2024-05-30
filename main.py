from data.load_dataset import load_data
from data.preprocess import preprocess_data
from model.fine_tuning import fine_tune_model
from model.generate_summary import generate_summary
from transformers import BartTokenizer

def main():
    train_dataset, valid_dataset, _ = load_data()
    
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    tokenized_train_dataset = preprocess_data(train_dataset, tokenizer)
    tokenized_valid_dataset = preprocess_data(valid_dataset, tokenizer)
    
    model = fine_tune_model(tokenized_train_dataset, tokenized_valid_dataset, model_name)
    
    # Example text for summarization
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

Follow Vikram Gaur"""
    summary = generate_summary(model, tokenizer, text)
    print("Summary:", summary)

if __name__ == "__main__":
    main()


# from datasets import load_dataset
# from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
# from transformers import pipeline

# # Load Dataset
# dataset = load_dataset('cnn_dailymail', '3.0.0')

# # Initialize Tokenizer and Model
# model_name = 'facebook/bart-large-cnn'
# tokenizer = BartTokenizer.from_pretrained(model_name)
# model = BartForConditionalGeneration.from_pretrained(model_name)

# # Tokenize Data
# def preprocess_data(dataset, tokenizer, max_input_length=1024, max_target_length=128):
#     def tokenize_function(examples):
#         inputs = [doc for doc in examples['article']]
#         model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

#         with tokenizer.as_target_tokenizer():
#             labels = tokenizer(examples['highlights'], max_length=max_target_length, truncation=True)

#         model_inputs['labels'] = labels['input_ids']
#         return model_inputs

#     tokenized_dataset = dataset.map(tokenize_function, batched=True)
#     return tokenized_dataset

# train_dataset, valid_dataset, test_dataset = dataset['train'], dataset['validation'], dataset['test']
# tokenized_train_dataset = preprocess_data(train_dataset, tokenizer)
# tokenized_valid_dataset = preprocess_data(valid_dataset, tokenizer)

# # Define Training Arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy='epoch',
#     learning_rate=3e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     save_total_limit=1,
#     fp16=True,
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_valid_dataset,
# )

# # Train the Model
# trainer.train()

# # Generate Summary
# def generate_summary(model, tokenizer, text, max_input_length=512, max_output_length=150):
#     inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=max_input_length, truncation=True)
#     summary_ids = model.generate(inputs, max_length=max_output_length, min_length=5, length_penalty=1.0, num_beams=2, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

# # Example Text
# text = """How to reach out for a referral? Few basic things we can keep in mind..."""
# summary = generate_summary(model, tokenizer, text)
# print("Summary:", summary)
