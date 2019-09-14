from bert import tokenization


def customize_tokenizer(text, do_lower_case=False):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    temp_x = ""
    text = tokenization.convert_to_unicode(text)
    for c in text:
        if tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(
                c) or tokenization._is_control(c):
            temp_x += " " + c + " "
        else:
            temp_x += c
    if do_lower_case:
        temp_x = temp_x.lower()
    return temp_x.split()


#
class ChineseFullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=False):
        self.vocab = tokenization.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        split_tokens = []
        for token in customize_tokenizer(text, do_lower_case=self.do_lower_case):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return tokenization.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return tokenization.convert_by_vocab(self.inv_vocab, ids)
