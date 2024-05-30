from transformers import BartForConditionalGeneration, Trainer, TrainingArguments

def fine_tune_model(tokenized_train_dataset, tokenized_valid_dataset, model_name='facebook/bart-large-cnn'):
    model = BartForConditionalGeneration.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
    )

    trainer.train()
    return model
