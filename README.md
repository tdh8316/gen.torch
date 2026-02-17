# Gen.torch

This repository contains PyTorch implementations of various generative models.

Each model is implemented from scratch using only PyTorch and standard libraries, without relying on high-level frameworks, such as HuggingFace Transformers.
The goal is to provide clear and educational codebases for understanding how these models work under the hood.

## Usage

To use a specific model, navigate to its directory and run the model download scripts.
For example, to run the Gemma-3 model:

```bash
cd gemma-3
python download_gemma_3.py
python gemma_3.py
```

This will download the pre-trained model weights from Huggingface and run a simple inference example.
