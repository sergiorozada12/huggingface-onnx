from src.utils import read_txt, save_txt
from src.config import NAME, REGEX
from src.model import TranslationModel

TEXT = read_txt('data/test_text.txt')

if __name__ == '__main__':
    model = TranslationModel(name=NAME, split_regex=REGEX)
    text_translated = model.translate(TEXT)
    save_txt('data/test_text_translated.txt', text_translated)

