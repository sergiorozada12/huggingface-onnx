def read_txt(path):
    with open(path, 'r') as f:
        return f.read()

def save_txt(path, text):
    with open(path, 'w') as f:
        return f.write(text)