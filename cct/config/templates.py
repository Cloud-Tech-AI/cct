TEMPLATES = {
    "cct_2": {
        "encoder_config": {
            "num_layers": 2,
            "nheads": 2,
            "ffn_scaling_dim": 1,
            "dmodel": 128,
        },
        "tokenizer_config": {
            "cbam": True,
        },
    },
    "cct_4": {
        "encoder_config": {
            "num_layers": 4,
            "nheads": 2,
            "ffn_scaling_dim": 1,
            "dmodel": 128,
        },
        "tokenizer_config": {
            "cbam": True,
        },
    },
    "cct_6": {
        "encoder_config": {
            "num_layers": 6,
            "nheads": 4,
            "ffn_scaling_dim": 2,
            "dmodel": 256,
        },
        "tokenizer_config": {
            "cbam": True,
        },
    },
    "cct_7": {
        "encoder_config": {
            "num_layers": 7,
            "nheads": 4,
            "ffn_scaling_dim": 2,
            "dmodel": 256,
        },
        "tokenizer_config": {
            "cbam": True,
        },
    },
    "cct_8": {
        "encoder_config": {
            "num_layers": 8,
            "nheads": 4,
            "ffn_scaling_dim": 2,
            "dmodel": 256,
        },
        "tokenizer_config": {
            "cbam": True,
        },
    },
    "cct_14": {
        "encoder_config": {
            "num_layers": 14,
            "nheads": 6,
            "ffn_scaling_dim": 3,
            "dmodel": 384,
        },
        "tokenizer_config": {
            "cbam": True,
        },
    },
    "cct_16": {
        "encoder_config": {
            "num_layers": 16,
            "nheads": 6,
            "ffn_scaling_dim": 3,
            "dmodel": 384,
        },
        "tokenizer_config": {
            "cbam": True,
        },
    },
}
