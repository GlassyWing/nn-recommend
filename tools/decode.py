import pandas as pd


def next_comps(preds, confidence: float, vocab_comps):
    masks = preds > confidence
    indices = (preds * masks).transpose()
    df = pd.DataFrame(indices)
    for i in df.index[df[0] > 0]:
        yield (vocab_comps.words[i - 1], preds[0, i])
