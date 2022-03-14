from src.config import NAME, REGEX
from src.model import TranslationModel

text = 'Hola, mi nombre es Sergio y soy científico de datos'

if __name__ == '__main__':
    model = TranslationModel(name=NAME, split_regex=REGEX)
    print(model.translate(text))
