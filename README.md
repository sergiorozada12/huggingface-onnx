# Huggingface to ONNX 

This repository shows how to port a Huggingface's MarianMT torch model into ONNX. In this case, we have optimized [this](https://huggingface.co/Helsinki-NLP/opus-mt-es-ca) Spanish to Catalan translation model. MarianMT models need specific routines to be saved as usable ONNX models. Then, the models run using a greedy decoding routine using ONNX Runtime.

The core scripts of this repository are:
* ```src/onnx.py``` - Routines to save the encoder/decoder into ONNX format, optimize (offlice), and quantize them.
* ```src/models.py``` - Routines to implement ONNX Runtime sessions with the encoder/decoder and to run a greedy decodification.
* ```src/wrappers``` - Huggingface models wrapped to manage ```None``` inputs in ONNX.

Then, the models can be run via:
* ```run_torch_model.py``` - Runs a torch model
* ```run_onnx_converter.py``` - Runs the torch-to-ONNX converter
* ```run_onnx_model.py``` - Runs an ONNX model
* ```run_time_benchmarck.py``` - Runs a time benchmarck between models

# Installation

To replicate the development environement just run:

```
pip install -r requirements.txt
```
