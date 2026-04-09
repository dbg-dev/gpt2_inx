##
## Configuration for loading HF FlaxGPT2 model
##

def hfgpt2_to_local(num_layers: int):

    def ith_layer_map(i: int):
        # LHS = Flax.nnx parameters names, which are tuples
        # RHS = incoming PyTorch paths are delimited strings
        return {
            ('blocks', i, 'att', 'layers', 0, 'bias'): f"transformer.h.{i}.ln_1.bias",
            ('blocks', i, 'att', 'layers', 0, 'scale'):  f"transformer.h.{i}.ln_1.weight",
            ('blocks', i, 'att', 'layers', 1, 'qkv', 'bias'): f"transformer.h.{i}.attn.c_attn.bias",
            ('blocks', i, 'att', 'layers', 1, 'qkv', 'kernel'): f"transformer.h.{i}.attn.c_attn.weight",
            ('blocks', i, 'att', 'layers', 1, 'lin', 'bias'):  f"transformer.h.{i}.attn.c_proj.bias",
            ('blocks', i, 'att', 'layers', 1, 'lin', 'kernel'): f"transformer.h.{i}.attn.c_proj.weight", 
            ('blocks', i, 'ff', 'layers', 0, 'bias'):  f"transformer.h.{i}.ln_2.bias", 
            ('blocks', i, 'ff', 'layers', 0, 'scale'):  f"transformer.h.{i}.ln_2.weight",
            ('blocks', i, 'ff', 'layers', 1, 'mlp', 'layers', 0, 'bias'): f"transformer.h.{i}.mlp.c_fc.bias",
            ('blocks', i, 'ff', 'layers', 1, 'mlp', 'layers', 0, 'kernel'): f"transformer.h.{i}.mlp.c_fc.weight",
            ('blocks', i, 'ff', 'layers', 1, 'mlp', 'layers', 2, 'bias'): f"transformer.h.{i}.mlp.c_proj.bias",
            ('blocks', i, 'ff', 'layers', 1, 'mlp', 'layers', 2, 'kernel'): f"transformer.h.{i}.mlp.c_proj.weight"
        }

    all_layers = {
        k: v
        for i in range(num_layers)
        for k, v in ith_layer_map(i).items()
    }

    non_layer_maps = {
        ('layernorm', 'bias'): "transformer.ln_f.bias", 
        ('layernorm', 'scale'): "transformer.ln_f.weight", 
        ('lm_head', 'kernel'): "lm_head.weight", 
        ('embed', 'pos_embedding', 'embedding'): "transformer.wpe.weight", 
        ('embed', 'tok_embedding', 'embedding'): "transformer.wte.weight"
    }

    return all_layers | non_layer_maps