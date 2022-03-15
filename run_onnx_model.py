from src.utils import read_txt, save_txt
from src.config import NAME, NAME_ONNX, REGEX
from src.models import TranslationModelOnnx

TEXT = read_txt('data/test_text.txt')

if __name__ == '__main__':
    model = TranslationModelOnnx(name_tokenizer=NAME, name_model=NAME_ONNX, split_regex=REGEX)
    text_translated = model.translate(TEXT)

    save_txt('data/test_text_translated_onnx.txt', text_translated)
