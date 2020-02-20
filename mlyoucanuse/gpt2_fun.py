"""`gpt2_fun.py` - functions for using GPT2"""
from typing import Tuple

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def predict_next_token(
    words: str, gpt2_model: GPT2LMHeadModel, gpt2_tokenizer: GPT2Tokenizer, top=3
) -> Tuple[Tuple[str, float], ...]:
    """
    Predict the next token, given a some starting words.
    :param words: a string of a few words
    :param gpt2_model: GPT2LMHeadModel preferably
    :param gpt2_tokenizer: GPT2Tokenizer
    :param top: the number of probable tokens to return
    :return: a tuple of tuples (token, probability)
    """
    tokens_tensor = torch.tensor(  # pylint: disable=not-callable
        gpt2_tokenizer.encode(words, add_special_tokens=True)
    ).unsqueeze(
        0
    )  # Batch size 1
    soft = torch.nn.Softmax(dim=1)
    gpt2_model.eval()
    with torch.no_grad():
        predictions = gpt2_model(tokens_tensor)[0].squeeze(0)
        predictions = soft(predictions)
        values, indices = torch.topk(  # pylint: disable=no-member
            predictions[-1, :], top
        )
        id_prob = list(zip(indices, values))
    return tuple(
        [  # type: ignore
            (gpt2_tokenizer.decode(int(tmp[0])).strip(), float(tmp[1]))
            for tmp in id_prob
        ]
    )
