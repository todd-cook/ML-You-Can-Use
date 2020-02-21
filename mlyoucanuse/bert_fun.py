"""`bert_fun.py` - some useful utility methods for interacting with BERT models"""
from copy import deepcopy
import logging
import math
from itertools import chain
from typing import Tuple, List, Dict

import torch
from nltk import word_tokenize
from transformers import BertForMaskedLM, BertTokenizer

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


# pylint: disable=too-many-locals


def get_word_probabilities(
    sentence: str, bert_model: BertForMaskedLM, bert_tokenizer: BertTokenizer
) -> Tuple[Tuple[str, Tuple[str, ...], Tuple[float, ...]]]:
    """
    Returns the probability of each word in a sentence.
    Returns a sequence of subtokens and their probabilities.

    :param sentence:
    :param bert_model:
    :param bert_tokenizer:
    :return:
    """
    whole_tokens = word_tokenize(sentence)
    bert_token_map = {
        idx: bert_tokenizer.encode(whole_token, add_special_tokens=False)
        for idx, whole_token in enumerate(whole_tokens)
    }
    start_token_id = bert_tokenizer.encode("[CLS]", add_special_tokens=False)
    end_token_id = bert_tokenizer.encode("[SEP]", add_special_tokens=False)
    total_tokens = len(list(chain.from_iterable(bert_token_map.values())))
    LOG.debug("# bert tokens: %s # whole tokens: %s", total_tokens, len(whole_tokens))
    torch.set_grad_enabled(False)
    word_probas = []
    for idx in bert_token_map:
        LOG.debug("idx %s", idx)
        bert_model.eval()
        with torch.no_grad():
            tmp_token_map = deepcopy(bert_token_map)
            curr_slot_len = len(tmp_token_map[idx])
            tmp_token_map[idx] = bert_tokenizer.encode(
                " [MASK] " * curr_slot_len, add_special_tokens=False
            )
            the_tokens = list(chain.from_iterable(tmp_token_map.values()))
            indexed_tokens = list(start_token_id) + the_tokens + list(end_token_id)
            LOG.debug("index tokens %s", ",".join([str(tmp) for tmp in indexed_tokens]))
            predictions = _make_predictions(indexed_tokens, bert_model)
        # get true index for predictions
        starting_position = (
            len(
                list(
                    chain.from_iterable(
                        [vals for key, vals in bert_token_map.items() if key < idx]
                    )
                )
            )
            + 1
        )
        if curr_slot_len > 1:
            subtokens = []
            probas = []
            for col, orig_token_id in enumerate(bert_token_map[idx]):
                orig_word_proba = predictions[starting_position + col][
                    orig_token_id
                ].item()
                subtokens.append(bert_tokenizer.convert_ids_to_tokens(orig_token_id))
                probas.append(orig_word_proba)
                LOG.debug(
                    "token %s %s %s",
                    bert_tokenizer.convert_ids_to_tokens(orig_token_id),
                    col,
                    orig_word_proba,
                )
            word_probas.append((whole_tokens[idx], tuple(subtokens), tuple(probas)))
        else:
            orig_token_id = bert_token_map[idx][0]
            orig_word_proba = predictions[starting_position][orig_token_id].item()
            word_probas.append(
                (
                    whole_tokens[idx],
                    tuple(bert_tokenizer.convert_ids_to_tokens([orig_token_id])),
                    tuple([orig_word_proba]),
                )
            )
    return tuple(word_probas)  # type: ignore


def get_alternate_words(
    sentence: str,
    word_index: int,
    bert_model: BertForMaskedLM,
    bert_tokenizer: BertTokenizer,
    top: int = 10,
) -> Tuple[float, List[Tuple[float, str]]]:
    """
    Get N alternate words for a particular word in a sentence.

    :param sentence:
    :param word_index: zero based index
    :param bert_model:
    :param bert_tokenizer:
    :param top:
    :return:
    """
    whole_tokens = word_tokenize(sentence)
    bert_token_map = {
        idx: bert_tokenizer.encode(whole_token, add_special_tokens=False)
        for idx, whole_token in enumerate(whole_tokens)
    }
    bert_token_map[word_index] = bert_tokenizer.encode(
        "[MASK]", add_special_tokens=False
    )
    total_tokens = len(list(chain.from_iterable(bert_token_map.values())))
    LOG.debug("total bert tokens: %s whole tokens: %s", total_tokens, len(whole_tokens))
    torch.set_grad_enabled(False)
    predictions = _make_predictions(
        _prepare_predictions(bert_token_map, bert_tokenizer), bert_model
    )
    # to find the true index of the desired word; count all of the tokens and subtokens before
    the_word_idx = (
        len(
            list(
                chain.from_iterable(
                    [vals for key, vals in bert_token_map.items() if key < word_index]
                )
            )
        )
        + 1
    )
    probas, indices = torch.topk(  # pylint: disable=no-member
        predictions[the_word_idx], top
    )
    alt_words = bert_tokenizer.convert_ids_to_tokens(indices.tolist())
    return list(zip(alt_words, probas.tolist()))  # type: ignore


def _prepare_predictions(
    bert_token_map: Dict[int, List[int]], bert_tokenizer: BertTokenizer
):
    """
    Helper method
    :param start_token_id:
    :param end_token_id:
    :param bert_token_map:
    :param bert_model:
    :param bert_tokenizer:
    :return:
    """
    start_token_id = bert_tokenizer.encode("[CLS]", add_special_tokens=False)
    end_token_id = bert_tokenizer.encode("[SEP]", add_special_tokens=False)
    the_tokens = list(chain.from_iterable(bert_token_map.values()))
    LOG.debug(bert_tokenizer.convert_ids_to_tokens(the_tokens))
    indexed_tokens = list(start_token_id) + the_tokens + list(end_token_id)
    LOG.debug(indexed_tokens)
    return indexed_tokens


def _make_predictions(indexed_tokens: List[int], bert_model: BertForMaskedLM):
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        # pylint: disable=not-callable,no-member
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor(
            [
                torch.zeros(  # type: ignore
                    len(indexed_tokens), dtype=int
                ).tolist()
            ]
        )
        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0].squeeze(0)
        return softmax(predictions)


def get_word_in_sentence_probability(
    sentence: str,
    word: str,
    bert_model: BertForMaskedLM,
    bert_tokenizer: BertTokenizer,
    word_index: int = -1,
) -> Tuple[Tuple[str, float], ...]:
    """
    Given a sentence, and a slot, determine what the probability is for a given word.
    Reports subword tokenization probabilities.

    :param sentence: A sentence providing context for the word.
    :param word: the word for which you would like the probability; if it's not in the sentence
    you provide, you will have to also pass a position argument.
    :param bert_model: an instance of BertForMaskedLM (preferably cased, large)
    :param bert_tokenizer: a BertTokenizer (preferably cased, large)
    :param word_index: The location in the sentence for which you would like the probability of
     the `word` parameter. Zero-based index of the words.
    :return: a tuple of tuples of (token:str, probability:float) and the probability value
    represents the raw softmax value, not a value 0-100.0
    """
    whole_tokens = word_tokenize(sentence)
    if word_index == -1:
        word_index = whole_tokens.index(word)
    bert_token_map = {
        idx: bert_tokenizer.encode(whole_token, add_special_tokens=False)
        for idx, whole_token in enumerate(whole_tokens)
    }  # type: Dict[int,List[int]]
    mask_token_id = bert_tokenizer.encode("[MASK]", add_special_tokens=False)
    tokens_to_predict = bert_tokenizer.encode(word, add_special_tokens=False)
    bert_token_map[word_index] = mask_token_id * len(tokens_to_predict)  # type: ignore
    LOG.debug(
        "total bert tokens: %s whole tokens: %s",
        len(list(chain.from_iterable(bert_token_map.values()))),
        len(whole_tokens),
    )
    torch.set_grad_enabled(False)
    bert_model.eval()
    # to find the true index of the desired word; count all of the tokens and subtokens before
    starting_position = (
        len(
            list(
                chain.from_iterable(
                    [
                        vals for key, vals in bert_token_map.items() if key < word_index
                    ]  # type : ignore
                )
            )
        )
        + 1
    )
    predictions = _make_predictions(
        _prepare_predictions(bert_token_map, bert_tokenizer), bert_model
    )
    if len(tokens_to_predict) == 1:
        return tuple([predictions[starting_position][tokens_to_predict].item()])
    return tuple(
        [
            (
                bert_tokenizer.convert_ids_to_tokens(tmp),
                predictions[starting_position + idx][tmp].item(),  # type: ignore
            )
            for idx, tmp in enumerate(tokens_to_predict)
        ]
    )  # type: ignore


def sum_log_probabilities(
    results: Tuple[Tuple[str, Tuple[str, ...], Tuple[float, ...]]]
) -> float:
    """
    Get the sum of the log probabilities for a sentence provided by a BERT model.

    "We extract the probability of a sentence from BERT, by iteratively masking every word in the sentence and then summing the log probabilities. While this approach is far from ideal, it has been shown (Wang and Cho, 2019) that it approximates the log-likelihood of a sentence."
    See: `The Unreasonable Effectiveness of Transformer Language Models in Grammatical Error Correction` by Dimitris Alikaniotis, Vipul Raheja [https://arxiv.org/abs/1906.01733]
    See: `BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model` by Alex Wang, Kyunghyun Cho [https://arxiv.org/pdf/1902.04094.pdf]

    :param results: a tuple collection of tuples string, strings, floats; representing
    the original word, the subword tokens--if any, and the probabilities of the second strings.
    :return: a float of the probability, which should be in the range of 0-100, usually.
    """
    _, _, data = zip(*results)
    return sum([math.log(tmp * 100) for tmp in chain.from_iterable(data)])  # type: ignore
