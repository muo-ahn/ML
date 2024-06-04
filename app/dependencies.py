import logging
from transformers import TFT5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)

model = None
tokenizer = None

def get_model():
    return model

def get_tokenizer():
    return tokenizer

async def startup_event():
    global model, tokenizer
    model_directory = "../ML/summarization_eng_t5"
    logger.debug(f"Loading model from {model_directory}")
    model = TFT5ForConditionalGeneration.from_pretrained(model_directory)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    logger.info("Model and Tokenizer loaded at startup")
