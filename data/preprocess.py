from transformers import BartTokenizer

def preprocess_data(dataset, tokenizer, max_input_length=1024, max_target_length=128):
    def tokenize_function(examples):
        inputs = [doc for doc in examples['article']]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['highlights'], max_length=max_target_length, truncation=True)

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset
