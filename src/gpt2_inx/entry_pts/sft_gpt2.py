from gpt2_inx.pipelines.training import finetune
import jax


def main():
    print(jax.devices()) 
    print(jax.default_backend())
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    model_id = "gpt2"

    _ = finetune(url, model_id)


if __name__ == "__main__":
    main()
