from datasets import load_metric

def evaluate_model(model, tokenized_dataset, tokenizer, metric_name='rouge'):
    metric = load_metric(metric_name)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(decoded_pred) for decoded_pred in decoded_preds]
        decoded_labels = ["\n".join(decoded_label) for decoded_label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return result

    results = model.evaluate(tokenized_dataset, metric_key_prefix=metric_name, compute_metrics=compute_metrics)
    return results
