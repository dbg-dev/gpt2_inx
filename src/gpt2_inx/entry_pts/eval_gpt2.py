from loguru import logger
from transformers import GPT2Tokenizer
from jax.numpy import array
from gpt2_inx.configs.hyperparams import GPT2_124M
from gpt2_inx.pipelines.model import from_hf
from gpt2_inx.samplers import sample_greedy


def main():
    hfmodel = "gpt2"
    model = from_hf(GPT2_124M, hfmodel)
    tokenizer = GPT2Tokenizer.from_pretrained(hfmodel)

    prompt = "Hello, I am"
    model.eval()  # put model into evaluation mode. will switch off dropouts
    inputs = array(tokenizer(prompt, return_tensors="np")["input_ids"])
    logits = sample_greedy(model, inputs)

    act = tokenizer.decode(logits[0].tolist(), skip_special_tokens=True)
    logger.info(f" output = {act}")


if __name__ == "__main__":
    main()
