#! /opt/anaconda3/bin/python3

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

DATA_DIR = "../../Datasets" 
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = "./multilingual_translator"
BATCH_SIZE = 2 
EPOCHS = 1 
MAX_LENGTH = 128 

LANG_TAGS = {
    "en": "en_XX",
    "hi": "hi_IN",
    "ur": "ur_PK"
}

def load_parallel_data(file_src, file_tgt, src_lang, tgt_lang):
    data_pairs = []
    with open(file_src, "r", encoding="utf-8") as fsrc, open(file_tgt, "r", encoding="utf-8") as ftgt:
        for src_line, tgt_line in zip(fsrc, ftgt):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if not src_line or not tgt_line:
                continue
            tagged_src_line = f"<<{LANG_TAGS[tgt_lang]}>> {src_line}"
            data_pairs.append({"src": tagged_src_line, "tgt": tgt_line})
    return data_pairs


def build_dataset():
    all_samples = []

    en_hi_dir = os.path.join(DATA_DIR, "English-Hindi")
    file_en = os.path.join(en_hi_dir, "NLLB.en-hi.en")
    file_hi = os.path.join(en_hi_dir, "NLLB.en-hi.hi")
    all_samples += load_parallel_data(file_en, file_hi, "en", "hi")
    all_samples += load_parallel_data(file_hi, file_en, "hi", "en")

    en_ur_dir = os.path.join(DATA_DIR, "English-Urdu")
    file_en = os.path.join(en_ur_dir, "NLLB.en-ur.en")
    file_ur = os.path.join(en_ur_dir, "NLLB.en-ur.ur")
    all_samples += load_parallel_data(file_en, file_ur, "en", "ur")
    all_samples += load_parallel_data(file_ur, file_en, "ur", "en")

    hi_ur_dir = os.path.join(DATA_DIR, "Hindi-Urdu")
    file_hi = os.path.join(hi_ur_dir, "NLLB.hi-ur.hi")
    file_ur = os.path.join(hi_ur_dir, "NLLB.hi-ur.ur")
    all_samples += load_parallel_data(file_hi, file_ur, "hi", "ur")
    all_samples += load_parallel_data(file_ur, file_hi, "ur", "hi")

    dataset = Dataset.from_list(all_samples)
    return dataset

def tokenize_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["src"], 
        max_length=MAX_LENGTH, 
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["tgt"], 
            max_length=MAX_LENGTH, 
            truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    raw_dataset = build_dataset()
    print(f"Total parallel samples: {len(raw_dataset)}")

    raw_dataset = raw_dataset.shuffle(seed=42)
    train_size = int(0.98 * len(raw_dataset))
    train_dataset = raw_dataset.select(range(train_size))
    eval_dataset = raw_dataset.select(range(train_size, len(raw_dataset)))

    def preprocess(examples):
        return tokenize_function(examples, tokenizer)

    train_dataset = train_dataset.map(preprocess, batched=True)
    eval_dataset = eval_dataset.map(preprocess, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    print("Training complete. Model saved to:", OUTPUT_DIR)
