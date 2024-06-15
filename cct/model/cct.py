from typing import Optional
import torch.nn as nn

from ..config.configurations import CCTConfig
from ..config.templates import TEMPLATES
from ..tokenizer.tokenizer import ConvTokenizer
from .encoder import TransformerEncoder


class CCT(nn.Module):
    def __init__(
        self,
        tokenizer_config: dict = {},
        encoder_config: dict = {},
        model_name: Optional[str] = None,
    ):
        super(CCT, self).__init__()
        if model_name is not None:
            assert model_name in TEMPLATES, f"{model_name} is not a valid model config."
            tokenizer_config = (
                tokenizer_config | TEMPLATES[model_name]["tokenizer_config"]
                if tokenizer_config
                else TEMPLATES[model_name]["tokenizer_config"]
            )
            encoder_config = (
                encoder_config | TEMPLATES[model_name]["encoder_config"]
                if encoder_config
                else TEMPLATES[model_name]["encoder_config"]
            )
        self.config = CCTConfig(tokenizer_config, encoder_config).to_dict()
        self.tokenizer = ConvTokenizer(
            input_channels=self.config["input_channels"],
            output_channels=self.config["dmodel"],
            num_conv_layers=self.config["num_conv_layers"],
            intermediate_conv_dim=self.config["intermediate_conv_dim"],
            kernel_size=self.config["kernel_size"],
            stride=self.config["stride"],
            padding=self.config["padding"],
            pooling_kernel_size=self.config["pooling_kernel_size"],
            pooling_stride=self.config["pooling_stride"],
            pooling_padding=self.config["pooling_padding"],
            cbam=self.config["cbam"],
            activation=self.config["activation"],
            max_pool=self.config["max_pool"],
            conv_bias=self.config["conv_bias"],
        )
        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(
                    dmodel=self.config["dmodel"],
                    nheads=self.config["nheads"],
                    ffn_scaling_dim=self.config["ffn_scaling_dim"],
                )
                for _ in range(self.config["num_layers"])
            ]
        )

    def forward(self, src):
        src = self.tokenizer(src)
        for layer in self.encoder:
            src = layer(src)
        return src


if __name__ == "__main__":
    model = CCT(model_name="cct_2")
