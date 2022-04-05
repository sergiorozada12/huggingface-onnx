from pathlib import Path
import onnx

from transformers.onnx.convert import export, validate_model_outputs
from transformers.onnx.features import FeaturesManager
from transformers import MarianTokenizer

class OnnxConverter():
    @staticmethod
    def convert_to_onnx(name, feature, output_path, opset, atol):
        tokenizer = MarianTokenizer.from_pretrained(name)
        model = FeaturesManager.get_model_from_feature(feature, name)

        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
        onnx_config = model_onnx_config(model.config)

        _, onnx_outputs = export(tokenizer, model, onnx_config, opset, Path(output_path))
        validate_model_outputs(onnx_config, tokenizer, model, Path(output_path), onnx_outputs, atol)

        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)

    def optimize_onnx_model():
        from onnxruntime_tools import optimizer
        from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions

        # disable embedding layer norm optimization for better model size reduction
        opt_options = BertOptimizationOptions('bert')
        opt_options.enable_embed_layer_norm = False

        opt_model = optimizer.optimize_model(
            'onnx/bert-base-cased.onnx',
            'bert', 
            num_heads=12,
            hidden_size=768,
            optimization_options=opt_options)
        opt_model.save_model_to_file('bert.opt.onnx')