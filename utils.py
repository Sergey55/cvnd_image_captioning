"""Utility method"""

from typing import List
from vocabulary import Vocabulary

def result_to_text(words: List[int], vocabulary: Vocabulary):
    """Convert list of word indexes to string

    Args:
        words: List of word indexec
        vocabulary: Vocabulary to convert

    Returns:
        Result sentence
    """
    sentence_words = []

    for idx in words:
        if idx == 0 or idx == 2:
            continue
        elif idx == 1:
            break
        else:
            sentence_words.append(vocabulary.idx2word[idx])

    sentence_words[0] = sentence_words[0].capitalize()

    # Punctuation mark is the last element of array. 
    sentence = ' '.join(sentence_words)[:-2] + sentence_words[-1]

    return sentence
