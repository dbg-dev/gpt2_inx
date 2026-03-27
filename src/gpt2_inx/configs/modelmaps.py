##
## Configuration for loading HF FlaxGPT2 model
##


def hfgpt2_to_local(num_layers: int): 

    def ith_layer_map(i: int):
        # LHS = Flax.nnx parameters names, which are tuples
        # RHS = incoming PyTorch paths are delimited strings
        return {
            ("blocks", i, "att", "layers", 0, "bias"): f"transformer.h.{i}.ln_1.bias",
            ("blocks", i, "att", "layers", 0, "scale"): f"transformer.h.{i}.ln_1.scale",
            ("blocks", i, "att", "layers", 1, "qkv", "bias"): f"transformer.h.{i}.attn.c_attn.bias",
            (
                "blocks",
                i,
                "att",
                "layers",
                1,
                "qkv",
                "kernel",
            ): f"transformer.h.{i}.attn.c_attn.kernel",
            ("blocks", i, "att", "layers", 1, "lin", "bias"): f"transformer.h.{i}.attn.c_proj.bias",
            (
                "blocks",
                i,
                "att",
                "layers",
                1,
                "lin",
                "kernel",
            ): f"transformer.h.{i}.attn.c_proj.kernel",
            ("blocks", i, "ff", "layers", 0, "bias"): f"transformer.h.{i}.ln_2.bias",
            ("blocks", i, "ff", "layers", 0, "scale"): f"transformer.h.{i}.ln_2.scale",
            (
                "blocks",
                i,
                "ff",
                "layers",
                1,
                "mlp",
                "layers",
                0,
                "bias",
            ): f"transformer.h.{i}.mlp.c_fc.bias",
            (
                "blocks",
                i,
                "ff",
                "layers",
                1,
                "mlp",
                "layers",
                0,
                "kernel",
            ): f"transformer.h.{i}.mlp.c_fc.kernel",
            (
                "blocks",
                i,
                "ff",
                "layers",
                1,
                "mlp",
                "layers",
                2,
                "bias",
            ): f"transformer.h.{i}.mlp.c_proj.bias",
            (
                "blocks",
                i,
                "ff",
                "layers",
                1,
                "mlp",
                "layers",
                2,
                "kernel",
            ): f"transformer.h.{i}.mlp.c_proj.kernel",
        }

    all_layers = {k: v for i in range(num_layers) for k, v in ith_layer_map(i).items()}

    non_layer_maps = {
        ("final_norm", "bias"): "transformer.ln_f.bias",
        ("final_norm", "scale"): "transformer.ln_f.scale",
        ("embed", "pos_embedding", "embedding"): "transformer.wpe.embedding",
        ("embed", "tok_embedding", "embedding"): "transformer.wte.embedding",
    }

    return all_layers | non_layer_maps
