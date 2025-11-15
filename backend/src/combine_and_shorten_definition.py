from reportlab.pdfbase.pdfmetrics import stringWidth
from exceptions import GenException

class Result:
    # num_words: number of words used in text
    def __init__(self, text, num_words):
        self.text = text;
        self.num_words = num_words;

# sep: string separating translations in definition
# definition: list of translations (strings)
def combine_and_shorten_definition(definition, sep, max_w, font_name, font_size):
    if len(definition) == 0:
        # Return empty string for words with no definition rather than raising
        return Result('', 0);
    text = sep.join(definition);
    w = stringWidth(text, font_name, font_size);
    if w <= max_w:
        return Result(text, len(definition));
    return combine_and_shorten_definition(definition[0:-1], sep, max_w, font_name, font_size);
