import logging
from transformers import TFT5ForConditionalGeneration, T5Tokenizer, PreTrainedTokenizerFast, BartForConditionalGeneration

logger = logging.getLogger(__name__)

model_eng = None
tokenizer_eng = None
model_kor = None
tokenizer_kor = None

def get_model_eng():
    return model_eng

def get_model_kor():
    return model_kor

def get_tokenizer_eng():
    return tokenizer_eng

def get_tokenizer_kor():
    return tokenizer_kor

async def startup_event():
    logger.info("MODEL LOADING STARTS")
    global model_eng, model_kor, tokenizer_eng, tokenizer_kor
    
    model_directory = "../ML/summarization_eng_t5"
    logger.debug(f"Loading model from {model_directory}")
    
    model_eng = TFT5ForConditionalGeneration.from_pretrained(model_directory)
    logger.info(f"English model : {model_eng}")
    tokenizer_eng = T5Tokenizer.from_pretrained("t5-small")
    
    model_kor = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")
    logger.info(f"Korean model : {model_kor}")
    tokenizer_kor = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
    
    logger.info("Model and Tokenizer loaded at startup")
