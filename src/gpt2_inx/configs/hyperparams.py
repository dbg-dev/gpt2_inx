from gpt2_inx.models.params import hyparams

GPT2_124M = hyparams(
    vocab_size=50257,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    ctx_len=1024,
    ff_hidden_dim=(4 * 768),  # 4 x em.
    use_bias=True,
)
