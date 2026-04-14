
from loguru import logger
from transformers import GPT2Tokenizer
from jax import Array

from gpt2_inx.configs.hyperparams import GPT2_124M
from gpt2_inx.configuration import CheckpointConfig, RuntimeConfig
from gpt2_inx.pipelines.model import from_hf
from gpt2_inx.pipelines.data import prepare
from gpt2_inx.trainer import train, CheckpointIO
from gpt2_inx.dataloader import make_eval_loader, make_train_loader
from gpt2_inx.metrics import cross_entropy_loss
from gpt2_inx.pipelines.inference import generate

def main():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    model_id = "gpt2"
    splits = [0.85, 0.1, 0.05]
    model = from_hf(GPT2_124M, model_id)

    # tokenizer = tiktoken.get_encoding(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    eos_id = tokenizer.eos_token_id
    
    config = RuntimeConfig(
        project_name = "nnx-trainer",
        batch_size=8,
        drop_remainder=True,
        n_epochs=2, 
        learning_rate=5e-5, 
        weight_decay=0.1,
        log_every = 10,
        eval_every = 100,
        prefetch_size=2,
        n_threads=2,
        n_workers=0,
    )

    def eval_fn(x: Array, y: Array) -> dict[str, Array]:
        loss = cross_entropy_loss(x, y)
        return {"loss": loss}

    train_ds, val_ds, test_ds = prepare(url, splits, tokenizer)
    n_train = len(train_ds[0])

    checkpt = CheckpointIO(
        CheckpointConfig() # go with defaults
    )
    train_loader = make_train_loader(train_ds, config)

    logger.info("Training model {}", model_id)
    model, _ = train(
        model,
        train_loader,
        loss_fn = cross_entropy_loss,
        config = config,
        n_train=n_train,
        eval_loader_fn = lambda: make_eval_loader(val_ds, config),
        eval_fn = eval_fn,
        checkpoint = checkpt
    )

    logger.info("Test model responses")
    for inx, exp_resp in test_ds[:3]:
        enc_inx = tokenizer(inx, return_tensors="np")["input_ids"]
        l = len(enc_inx)
        x = generate(model, enc_inx, max_new_tokens=64)
        act_resp = tokenizer.decode(x[x != eos_id])

        resp_header = "\n\n### Response:\n"
        logger.info("Instruction = {}", inx)
        logger.info("exp response = {}", exp_resp.replace(resp_header, ""))
        logger.info("act response = {}", act_resp.partition(resp_header)[2])


if __name__ == "__main__":
    main()
