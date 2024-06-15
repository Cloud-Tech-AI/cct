class EncoderConfig:
    def __init__(self, **kwargs) -> None:
        self.dmodel = kwargs.get("dmodel", 64)
        self.nheads = kwargs.get("nheads", 12)
        self.num_layers = kwargs.get("num_layers", 6)
        self.ffn_scaling_dim = kwargs.get("ffn_scaling_dim", 4)

    def to_dict(self):
        return {
            "dmodel": self.dmodel,
            "nheads": self.nheads,
            "num_layers": self.num_layers,
            "ffn_scaling_dim": self.ffn_scaling_dim,
        }


class TokenizerConfig:
    def __init__(self, **kwargs) -> None:
        self.input_channels = kwargs.get("input_channels", 3)
        self.num_conv_layers = kwargs.get("num_conv_layers", 1)
        self.intermediate_conv_dim = kwargs.get("intermediate_conv_dim", 64)

        self.kernel_size = kwargs.get("kernel_size", 3)
        self.stride = kwargs.get("stride", max(1, (self.kernel_size // 2) - 1))
        self.padding = kwargs.get("padding", max(1, self.kernel_size // 2))

        self.pooling_kernel_size = kwargs.get("pooling_kernel_size", 3)
        self.pooling_stride = kwargs.get("pooling_stride", 2)
        self.pooling_padding = kwargs.get("pooling_padding", 1)

        self.cbam = kwargs.get("cbam", True)
        self.activation = kwargs.get("activation", None)
        self.max_pool = kwargs.get("max_pool", True)
        self.conv_bias = kwargs.get("conv_bias", False)

    def to_dict(self):
        return {
            "input_channels": self.input_channels,
            "num_conv_layers": self.num_conv_layers,
            "intermediate_conv_dim": self.intermediate_conv_dim,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "pooling_kernel_size": self.pooling_kernel_size,
            "pooling_stride": self.pooling_stride,
            "pooling_padding": self.pooling_padding,
            "cbam": self.cbam,
            "activation": self.activation,
            "max_pool": self.max_pool,
            "conv_bias": self.conv_bias,
        }


class CCTConfig(TokenizerConfig, EncoderConfig):
    def __init__(self, tokenizer_config: dict, encoder_config: dict) -> None:
        TokenizerConfig.__init__(self, **tokenizer_config)
        EncoderConfig.__init__(self, **encoder_config)

    def to_dict(self):
        return {
            **TokenizerConfig.to_dict(self),
            **EncoderConfig.to_dict(self),
        }
