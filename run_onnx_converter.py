from src.onnx import OnnxConverter
from src.config import NAME, FEATURE, OPSET, ATOL

OUTPUT_PATH = 'onnx/test.onnx'

if __name__ == '__main__':
    OnnxConverter(
        name=NAME,
        batch_size=1,
        max_length=50,
        embedding_size=512
    ).convert_to_onnx()