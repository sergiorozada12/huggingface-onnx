from transformers import AutoTokenizer
from onnxruntime import InferenceSession
from transformers import MarianMTModel, MarianTokenizer


tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-ca')
session = InferenceSession("onnx/test.onnx")
# ONNX Runtime expects NumPy arrays as input
inputs = tokenizer("Hola qué tal soy científico de datos", return_tensors="np")
outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
print(outputs)