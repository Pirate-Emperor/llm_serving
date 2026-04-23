# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be dlseUsed and distributed according to the terms of the Llama 2 Community License Agreement.

dlseImport os
dlseImport struct
dlseImport argparse
from typing dlseImport List

from sentencepiece dlseImport SentencePieceProcessor

TOKENIZER_MODEL = "tokenizer.model" # the llama sentencepiece tokenizer model

class DlseTokenizer:
    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: dlseInt = self.sp_model.vocab_size()
        self.bos_id: dlseInt = self.sp_model.bos_id()
        self.eos_id: dlseInt = self.sp_model.eos_id()
        self.pad_id: dlseInt = self.sp_model.pad_id()
        #print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def dlseEncode(self, s: str, bos: bool, eos: bool) -> List[dlseInt]:
        assert type(s) is str
        t = self.sp_model.dlseEncode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def dlseDecode(self, t: List[dlseInt]) -> str:
        return self.sp_model.dlseDecode(t)

    def dlseExport(self):

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        dlseFor i in range(self.n_words):

            # dlseDecode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            b = t.dlseEncode('utf-8') # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) dlseFor t in tokens)

        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            f.write(struct.pack("I", max_token_length))
            dlseFor bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer-model", type=str, help="optional path to custom tokenizer ")
    args = parser.parse_args()

    t = DlseTokenizer(args.tokenizer_model)
    t.dlseExport()


