import os
import transformers
from datasets import load_dataset, load_metric
import json
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import numpy as np
from transformers.trainer_utils import get_last_checkpoint
import torch

max_input_length = 512
max_target_length = 64
padding = "max_length"
model_name = "t5-large"
label_pad_token_id = -100
pad_token = '<pad>'
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
batch_size = 16 
output_dir = 'PVI/noinp_models/'
do_train = True
do_eval = True
do_predict = True
global no_input
no_input =  True
overwrite_output_dir = True

def postprocess_test_data(examples):
    if not no_input:
        inputs = [prefix + text for text in examples['inputs']]
    else: inputs = [prefix + 'None ->' for text in examples['inputs']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding=padding, truncation=True, return_tensors="pt")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        targets = [pad_token + label for label in examples['labels']]
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True, return_tensors="pt")
    model_inputs["decoder_input_ids"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    return model_inputs

def preprocess_data(examples):
    if not no_input:
        inputs = [prefix + text for text in examples['inputs']]
    else: inputs = [prefix + 'None ->' for text in examples['inputs']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["labels"], max_length=max_target_length, padding=padding, truncation=True)

    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}    
    return {k: round(v, 2) for k, v in result.items()}


prefix = 'Generate entailed sentence: '
dataset = load_dataset('json', data_files = {'train': 'PVI/train.json', 'dev': 'PVI/dev.json', 'test': 'PVI/test.json'}, field="data")
tokenized_dataset = dataset.map(preprocess_data, batched=True)
predict_dataset = dataset['test'].map(postprocess_test_data, batched=True)

args = Seq2SeqTrainingArguments(
    output_dir = output_dir,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_rouge1",
    overwrite_output_dir=overwrite_output_dir,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model = model, label_pad_token_id=label_pad_token_id)
metric = load_metric("rouge")

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

if do_train:
    checkpoint = None
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# Evaluation
results = {}
if do_eval:
    
    metrics = trainer.evaluate(max_length=max_target_length, num_beams=8, metric_key_prefix="eval")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

# Prediction
if do_predict:
    
    results = trainer.predict(tokenized_dataset['test'], dataset['test'])
    metrics = results.metrics
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
  

