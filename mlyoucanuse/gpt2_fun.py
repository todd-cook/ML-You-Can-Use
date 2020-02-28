# Copyright 2020 Todd Cook
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`gpt2_fun.py` - functions for using GPT2"""
import logging
from typing import Tuple

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def predict_next_token(
    words: str, gpt2_model: GPT2LMHeadModel, gpt2_tokenizer: GPT2Tokenizer, top: int = 3
) -> Tuple[Tuple[str, float], ...]:
    """
    Predict the next token, given a some starting words.
    :param words: a string of a few words (max tokens: 1023)
    :param gpt2_model: GPT2LMHeadModel preferably
    :param gpt2_tokenizer: GPT2Tokenizer
    :param top: the number of probable tokens to return
    :return: a tuple of tuples (token, probability)

    ## OOME on circleci :-(
    # >>> gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # >>> gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    # >>> _ = gpt2_model.eval()
    # >>> predict_next_token('I am looking', gpt2_model, gpt2_tokenizer)
    # (('forward', 0.3665640652179718), ('for', 0.35346919298171997), ('to', 0.08423731476068497))

    """
    tokens_tensor = torch.tensor(  # pylint: disable=not-callable
        gpt2_tokenizer.encode(words, add_special_tokens=True)
    ).unsqueeze(
        0
    )  # Batch size 1
    if tokens_tensor.shape[1] > 1023:
        LOG.warning(
            "Too many tokens, should be 1023 or less, found %s", tokens_tensor.shape[1]
        )
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
