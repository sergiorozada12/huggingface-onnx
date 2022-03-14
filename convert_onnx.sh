python -m transformers.onnx \
    --model=Helsinki-NLP/opus-mt-es-ca \
    --feature=seq2seq-lm \
    --quantize \
    onnx/