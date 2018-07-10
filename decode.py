import pandas as pd


def next_comps(preds, confidence: float, vocab_comps):
    """
    根据概率分布找到下一个可能使用的构件
    :param preds:
    :param confidence:
    :param vocab_comps:
    :return:
    """
    masks = preds > confidence
    indices = (preds * masks).transpose()
    df = pd.DataFrame(indices)
    for i in df.index[df[0] > 0]:
        yield (vocab_comps.words[i - 1], preds[0, i])
