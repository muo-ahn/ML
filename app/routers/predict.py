from fastapi import APIRouter, Depends, HTTPException
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from app.dependencies import get_model_eng, get_model_kor, get_tokenizer_eng, get_tokenizer_kor
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class TextData(BaseModel):
    text: str
    language: str

@router.post("/predict")
def get_prediction(data: TextData):
    logger.debug(f"Received text: {data.text}")
    text = "summarize: " + data.text
    output_text = ""

    if data.language == "eng" : 
        model = TFT5ForConditionalGeneration(Depends(get_model_eng))
        tokenizer = T5Tokenizer(Depends(get_tokenizer_eng))

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="tf").input_ids
        logger.debug(f"Tokenized input: {inputs}")

        # Generate output text
        outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
        logger.debug(f"Model output: {outputs}")

        # Decode the output tokens into text
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Decoded output: {output_text}")

    elif data.language == "kor" :
        model = get_model_kor()
        tokenizer = get_tokenizer_kor()

        inputs = tokenizer(data.text, return_tensors="pt", padding="max_length", truncation=True, max_length=1026)
        logger.debug(f"Tokenized input: {inputs}")

        output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            length_penalty=1.0,
            max_length=300,
            min_length=12,
            num_beams=6,
            repetition_penalty=1.5,
            no_repeat_ngram_size=15,
            )
        logger.debug(f"Model output: {output}")

        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.debug(f"Decoded output: {output_text}")

    return {"summary" : output_text}