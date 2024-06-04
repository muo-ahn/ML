from fastapi import APIRouter, Depends, HTTPException
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from app.dependencies import get_model, get_tokenizer
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class TextData(BaseModel):
    text: str

@router.post("/predict")
def get_prediction(data: TextData, model: TFT5ForConditionalGeneration = Depends(get_model), tokenizer: T5Tokenizer = Depends(get_tokenizer)):
    logger.debug(f"Received text: {data.text}")
    text = "summarize: " + data.text

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="tf").input_ids
    logger.debug(f"Tokenized input: {inputs}")

    # Generate output text
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
    logger.debug(f"Model output: {outputs}")

    # Decode the output tokens into text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.debug(f"Decoded output: {output_text}")

    return {"summary": output_text}
