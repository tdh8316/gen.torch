# LLM.torch

This repository contains PyTorch implementations of various large language models (LLMs).

Each model is implemented from scratch using only PyTorch and standard libraries, without relying on high-level frameworks. The goal is to provide clear and educational codebases for understanding how these models work under the hood.

## Usage

To use a specific model, navigate to its directory and run the model download scripts. For example, to run the Gemma-3 model:

```bash
cd gemma-3
python download_gemma_3.py
python gemma_3.py
```
This will download the pre-trained weights and run a simple inference example.

