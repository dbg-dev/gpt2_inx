from loguru import logger
from transformers import GPT2Tokenizer
from gpt2_inx.pipelines.model import from_hf
from gpt2_inx.pipelines.inference import generate


def main():
    from gpt2_inx.configs.hyperparams import GPT2_124M
    hfmodel = "gpt2"
    
    model = from_hf(GPT2_124M, hfmodel)
    tokenizer = GPT2Tokenizer.from_pretrained(hfmodel)

    prompt = "Hello, I am"
    
    model.eval()  # put model into evaluation mode. will switch off dropouts
    inputs = tokenizer(prompt, return_tensors="np")["input_ids"]
    logits = generate(model, inputs, 10)
    act = tokenizer.decode(logits, skip_special_tokens=True)

    logger.info(f" output = {act}")


if __name__ == "__main__":
    main()
