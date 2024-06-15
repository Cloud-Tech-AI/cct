# Compact Convolution Transformers

This repository contains the unofficial PyTorch implementation of [Compact Convolution Transformers (CCT)](https://arxiv.org/pdf/2104.05704) with [Convolution Block Attention Module (CBAM)](https://arxiv.org/abs/1807.06521).

## Content

- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [Citation](#citation)


## Model Architecture

Use Convolutions with CBAM to tokenize the image and then use the transformer encoder to process the tokens.<br>
<img src="/docs/architecture.png"/><br>

Compact Convolution Transformer (CCT)<br>
<img src="/docs/cct.png"/><br>

Convolutional Block Attention Module (CBAM)<br>
<img src="/docs/cbam.png"/><br>


## Installation

```
git clone https://github.com/Cloud-Tech-AI/cct.git
cd cct
pip install .
```


## Usage
```
import torch

from cct import CCT

model = CCT(
    model_name='cct_2',
    tokenizer_config={'cbam': True}
)
img = torch.randn(1, 3, 224, 224)
output = model(img)
```

## References

- The official implementation for CCT [here](https://github.com/SHI-Labs/Compact-Transformers)
- The official implementation for CBAM [here](https://github.com/Jongchan/attention-module)


## Citation

```bibtex
@article{DBLP:journals/corr/abs-2104-05704,
  author       = {Ali Hassani, Steven Walton, Nikhil Shah, Abulikemu Abuduweili, Jiachen Li, Humphrey Shi},
  title        = {Escaping the Big Data Paradigm with Compact Transformers},
  year         = {2021},
  url          = {https://arxiv.org/abs/2104.05704},
}
```

```bibtex
@article{DBLP:journals/corr/abs-1807-06521,
  author       = {Sanghyun Woo, Jongchan Park, Joon{-}Young Lee, In So Kweon},
  title        = {{CBAM:} Convolutional Block Attention Module},
  journal      = {CoRR},
  year         = {2018},
  url          = {http://arxiv.org/abs/1807.06521},
}
```
