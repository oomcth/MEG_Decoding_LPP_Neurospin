# this code was taken from Corentin BEL. Please see the link bellow.
# https://github.com/BelCorentin/neurospin-petit-prince

import numpy as np
import pandas as pd
from Levenshtein import editops
import string


def get_path(name="visual"):
    return "LPP_MEG_auditory"


def get_code_path():
    return "neurospin-petit-prince"


def match_list(A, B, on_replace="delete"):
    if not isinstance(A, str):
        unique = np.unique(np.r_[A, B])
        label_encoder = dict((k, v) for v, k in enumerate(unique))

        def int_to_unicode(array: np.ndarray) -> str:
            return "".join([str(chr(label_encoder[ii])) for ii in array])

        A = int_to_unicode(A)
        B = int_to_unicode(B)

    changes = editops(A, B)
    B_sel = np.arange(len(B)).astype(float)
    A_sel = np.arange(len(A)).astype(float)
    for type_, val_a, val_b in changes:
        if type_ == "insert":
            B_sel[val_b] = np.nan
        elif type_ == "delete":
            A_sel[val_a] = np.nan
        elif on_replace == "delete":
            A_sel[val_a] = np.nan
            B_sel[val_b] = np.nan
        elif on_replace == "keep":
            pass
        else:
            raise NotImplementedError
    B_sel = B_sel[np.where(~np.isnan(B_sel))]
    A_sel = A_sel[np.where(~np.isnan(A_sel))]
    assert len(B_sel) == len(A_sel)
    return A_sel.astype(int), B_sel.astype(int)


def add_syntax(meta, syntax_path, run):
    meta = meta.copy().reset_index(drop=True)

    syntax_path = "neurospin-petit-prince-main/data/syntax_new_no_punct"
    syntax_file = (
        syntax_path + "/" + f"run{run}_v2_0.25_0.5-tokenized.syntax.txt"
    )
    synt = get_syntax(syntax_file)

    meta_tokens = meta.word.fillna("XXXX").apply(format_text).values
    meta_tokens = [stri.split("'")[1] if "'" in stri else stri for stri in meta.word]
    translator = str.maketrans("", "", string.punctuation)
    meta_tokens = [stri.translate(translator) for stri in meta_tokens]

    synt_tokens = synt.word.apply(format_text).values

    i, j = match_list(meta_tokens, synt_tokens)
    assert (len(i) / len(meta_tokens)) > 0.8

    for key, default_value in dict(n_closing=1, is_last_word=False, pos="XXX").items():
        meta[key] = default_value
        meta.loc[i, key] = synt.iloc[j][key].values

    content_pos = ("NC", "ADJ", "ADV", "VINF", "VS", "VPP", "V")
    meta["content_word"] = meta.pos.apply(
        lambda pos: pos in content_pos if isinstance(pos, str) else False
    )
    return meta


def get_syntax(file):
    with open(file, "r") as f:
        txt = f.readlines()

    out = []
    for sequence_id, sent in enumerate(txt):
        splits = sent.split("=")

        for prev, token in zip(splits, splits[1:]):
            out.append(
                dict(
                    pos=prev.split("(")[-1].split()[0],
                    word_id=int(prev.split()[-1]),
                    word=token.split(")")[0],
                    n_closing=token.count(")"),
                    sequence_id=sequence_id,
                    is_last_word=False,
                )
            )
        out[-1]["is_last_word"] = True

    synt = pd.DataFrame(out)

    out = []
    for sent, d in synt.groupby("sequence_id"):
        for token in d.itertuples():
            for tok in token.word.split("'"):
                out.append(dict(word=tok, n_closing=1, is_last_word=False, pos="XXX"))
            out[-1]["n_closing"] = token.n_closing
            out[-1]["is_last_word"] = token.is_last_word
            out[-1]["pos"] = token.pos
    return pd.DataFrame(out)


def format_text(text):
    text = text.replace("œ", "oe")
    return text.lower()
