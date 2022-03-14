from src.onnx import OnnxConverter
from src.config import NAME, FEATURE, OPSET, ATOL

OUTPUT_PATH = 'onnx/test.onnx'

if __name__ == '__main__':
    OnnxConverter.convert_to_onnx(
        name=NAME,
        feature=FEATURE,
        output_path=OUTPUT_PATH,
        opset=OPSET,
        atol=ATOL
    )