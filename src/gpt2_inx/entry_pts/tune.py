from loguru import logger
from gpt2_inx.configs.hyperparams import GPT2_124M
from gpt2_inx.pipelines.model import from_hf
from gpt2_inx.pipelines.data import prepare
from gpt2_inx.training.trainer import train, TrainerConfig
from gpt2_inx.training.loss_functions import cross_entropy_loss


def main():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    model_id = "gpt2"

    config = TrainerConfig(
        batch_size = 8,
        n_epochs = 2,
        learning_rate = 5e-5,
        weight_decay = 0.1,
        seq_len = 128
    )
    tds, vds = prepare(url, model_id)

    # setup model and trainer
    hfmodel = model_id
    model = from_hf(GPT2_124M, hfmodel)

    logger.info("Running model {}", hfmodel)
    _ = train(model, tds, vds, cross_entropy_loss ,config)

if __name__ == "__main__":
    main()
