#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "./multilingual_translator"
LANG_TAGS = {
    "en": "en_XX",
    "hi": "hi_IN",
    "ur": "ur_PK"
}

def translate_sentence(
    text, 
    source_lang="en", 
    target_lang="hi", 
    model_dir=MODEL_DIR, 
    max_length=128
):
    """
    Translate text from source_lang to target_lang using the fine-tuned model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).eval()

    tagged_source = f"<<{LANG_TAGS[target_lang]}>> {text}"

    inputs = tokenizer(tagged_source, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=4,
            max_length=max_length
        )
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


if __name__ == "__main__":
    en_text = "Hello, how are you?"
    hi_translation = translate_sentence(en_text, source_lang="en", target_lang="hi")
    print("EN -> HI:", hi_translation)

    hi_text = "मैं ठीक हूँ।"
    ur_translation = translate_sentence(hi_text, source_lang="hi", target_lang="ur")
    print("HI -> UR:", ur_translation)

    ur_text = "میں ٹھیک ہوں۔"
    en_translation = translate_sentence(ur_text, source_lang="ur", target_lang="en")
    print("UR -> EN:", en_translation)
