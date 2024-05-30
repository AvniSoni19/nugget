def generate_summary(model, tokenizer, text, max_input_length=512, max_output_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=max_input_length, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_output_length, min_length=5, length_penalty=1.0, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
