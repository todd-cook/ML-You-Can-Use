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
"""`text_classification_modeler.py` """

import logging
from typing import List, Dict, Optional

from keras.activations import relu
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Embedding,
    Conv1D,
    Dropout,
    concatenate,
    GlobalMaxPooling1D,
)

from mlyoucanuse.embeddings import (
    get_embeddings_index,
    create_embeddings_matrix,
    get_embeddings_layer,
)

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals


class TextClassificationModeler:
    """
    Configurable Single layer CNN with defaults suitable for general Text Classification.

    Based on ideas in:
    A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks
    for Sentence Classification
    by Ye Zhang, Byron Wallace
    https://arxiv.org/abs/1510.03820

    Convolutional Neural Networks for Sentence Classification
    by Yoon Kim
    https://arxiv.org/abs/1408.5882
    """

    def __init__(
        self,
        max_sequence_len: int,
        n_grams: List[int] = None,
        num_filters: int = 100,
        outputs: int = 1,
        loss_function: str = "binary_crossentropy",
        embedding_dimensions: int = 300,
        embeddings_name: Optional[str] = None,
        vocab_map: Dict[str, int] = None,  #: tokenizer.word_index
        cache_dir: Optional[str] = None,
        compile_model: bool = True,
        freeze_embeddings: bool = True,
        init_embeddings_by_variance: bool = True,
        dropout_rate: float = 0.5,
    ) -> None:
        """

        :param max_sequence_len: The maximum sequence length of text to use for classification.
        :param n_grams: a list of integers denoting n-grams to collect on
        :param num_filters: the number of filters; acts as hidden layers containing decision
        information. Recommended range: 100  to 600.
        :param outputs: integer indicating the number of desired outputs. Default is one for
        binary classification. For multi class classification, set the number of labels.
        :param loss_function: defaults to binary for single output; categorical_crossentropy
        for multiple outputs
        :param embedding_dimensions: the number of the dimensions of the embedding
        :param embeddings_name: A human readable name, for pretty diagrams
        :param vocab_map: a dictionary mapping unique words to integers
        :param cache_dir: the location where embeddings are stored
        :param compile_model: whether or not to compile the model automatically; set to false to
        set your own custom metrics or callbacks.
        :param freeze_embeddings: Whether or not to freeze the embeddings; trainable embeddings may
        produce better results
        :param init_embeddings_by_variance: Whether or not to initialize unmappped embeddings
        by the underlying variance of the known embeddings.
        :param dropout_rate: Decimal percentage of dropout to apply to the layers.
        """
        self.model = None
        self.max_sequence_len = max_sequence_len
        self.n_grams = [2, 3, 4, 5] if not n_grams else n_grams
        self.num_filters = num_filters
        self.outputs = outputs
        if not loss_function and outputs > 1:
            self.loss_function = "categorical_crossentropy"
        else:
            self.loss_function = loss_function
        self.embedding_dimensions = embedding_dimensions
        self.embeddings_name = embeddings_name
        self.freeze_embeddings = freeze_embeddings
        self.vocab_map = vocab_map
        self.dropout_rate = dropout_rate
        self.compile_model = compile_model
        self.cache_dir = cache_dir
        self.init_embeddings_by_variance = init_embeddings_by_variance
        if self.embeddings_name and self.embedding_dimensions and self.vocab_map:
            embedding_index = get_embeddings_index(
                embedding_name=self.embeddings_name,
                embedding_dimensions=self.embedding_dimensions,
            )
            embedding_matrix = create_embeddings_matrix(
                embeddings_index=embedding_index,
                vocabulary=self.vocab_map,
                embeddings_dimensions=self.embedding_dimensions,
                init_by_variance=self.init_embeddings_by_variance,
            )
            self.embedding_layer_to_use = get_embeddings_layer(
                embeddings_matrix=embedding_matrix,
                name=self.embeddings_name,
                max_len=self.max_sequence_len,
                trainable=self.freeze_embeddings,
            )

    def _get_central_layers(
        self, x_input: Input, suffix: str, n_grams: List[int], feature_maps: int = 100
    ):
        """

        :param x_input: The input layer
        :param max_len: the maximum length of the input
        :param suffix: the suffix to apply to the name of layers
        :param n_grams: the number of n-grams
        :param feature_maps: the number of filters or feature maps
        :return: branches of the intermediary layers
        """
        branches = []
        for val in n_grams:
            branch = Conv1D(
                filters=feature_maps,
                kernel_size=val,
                strides=1,
                activation=relu,
                name="Conv_" + suffix + "_" + str(val),
            )(x_input)
            if self.dropout_rate:
                branch = Dropout(
                    self.dropout_rate, name="Dropout_" + suffix + "_" + str(val)
                )(branch)
            branch = GlobalMaxPooling1D(
                name="GlobalMaxPool_" + suffix + "_" + str(val)
            )(branch)
            branches.append(branch)
        return branches

    def build_model(self, embedding_layer: Embedding) -> Model:
        """

        :param embedding_layer:
        :param max_len: the max length of the input; number of words
        :return: keras Model


        >>> import numpy as np
        >>> from mlyoucanuse.embeddings import create_embeddings_matrix, get_embeddings_layer
        >>> myembeddings_idxs = {'quick': np.ones(10), 'brown' : np.zeros(10) }
        >>> vocab = {'the': 1, 'quick': 2, 'brown': 3, 'fox': 4}
        >>> embeds = create_embeddings_matrix(embeddings_index=myembeddings_idxs, vocabulary=vocab,
        ... embeddings_dimensions=10, init_by_variance=False)
        >>> myembeds_layer = get_embeddings_layer(embeds, 'myembeds', max_len=20, trainable=False)
        >>> modeler = TextClassificationModeler(max_sequence_len=20 ,embedding_dimensions=10 )
        >>> model = modeler.build_model(myembeds_layer)
        >>> type(model) == Model
        True

        """
        feed = Input(shape=(self.max_sequence_len,), dtype="int32", name="main_input")
        embedding_input = embedding_layer(feed)

        branches = self._get_central_layers(
            embedding_input, "static", self.n_grams, self.num_filters
        )
        sandwich = concatenate(branches, axis=-1, name=f"Concatenate_{len(branches)}")

        if self.outputs == 1:
            output = Dense(1, activation="sigmoid", name="output")(sandwich)
        else:
            output = Dense(self.outputs, activation="softmax", name="output")(sandwich)
        model = Model(inputs=feed, outputs=output)
        if self.compile_model:
            model.compile(loss=self.loss_function, optimizer="adam", metrics=["acc"])
        self.model = model
        return model

    def build_dual_embeddings_model(
        self, embedding_layer_channel_1: Embedding, embedding_layer_channel_2: Embedding
    ) -> Model:
        """
        Split input into dynamic and static channels. This model has the possibility of allowing
        users to capture the advantages of static and dynamic embeddings; however, this is an area
        of continued exploration.
        :param embedding_layer_channel_1: a layer of static/frozen embeddings
        :param embedding_layer_channel_2: a layer trainable embeddings (static but unfrozen) or
        embeddings randomly initialized
        :param max_len: The maximum sequence length of the input
        :return: the model
        """

        input_dynamic = Input(
            shape=(self.max_sequence_len,), dtype="int32", name="input_dynamic"
        )
        embedding_input_1 = embedding_layer_channel_1(input_dynamic)
        branches_dynamic = self._get_central_layers(
            embedding_input_1, "static", self.n_grams, self.num_filters
        )
        z_dynamic = concatenate(branches_dynamic, axis=-1)

        input_static = Input(
            shape=(self.max_sequence_len,), dtype="int32", name="input_static"
        )
        embedding_input_2 = embedding_layer_channel_2(input_static)
        branches_static = self._get_central_layers(
            embedding_input_2, "dynamic", self.n_grams, self.num_filters
        )
        z_static = concatenate(branches_static, axis=-1)

        sandwich = concatenate([z_static, z_dynamic], axis=-1)

        if self.outputs == 1:
            output = Dense(1, activation="sigmoid", name="output")(sandwich)
        else:
            output = Dense(self.outputs, activation="softmax", name="output")(sandwich)

        model = Model(inputs=[input_static, input_dynamic], outputs=output)
        if self.compile_model:
            model.compile(loss=self.loss_function, optimizer="adam", metrics=["acc"])
        self.model = model
        return self.model
