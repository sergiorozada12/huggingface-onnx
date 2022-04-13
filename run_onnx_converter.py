from src.onnx import OnnxConverter
from src.config import NAME, FEATURE, OPSET, ATOL, MAX_LENGTH, BATCH_SIZE, EMBEDDING_SIZE

OUTPUT_PATH = 'onnx/test.onnx'

if __name__ == '__main__':
    OnnxConverter(
        name=NAME,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        embedding_size=EMBEDDING_SIZE
    ).convert_to_onnx()