# Copyright 2024 The TensorFlow Recommenders Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A pre-built ranking model."""

from typing import cast, Dict, Optional, Sequence, Tuple, Union

import tensorflow as tf

from absl import logging
from tensorflow_recommenders import layers
from tensorflow_recommenders import models
from tensorflow_recommenders import tasks
from tensorflow_recommenders.layers import feature_interaction as feature_interaction_lib

class MaskedAUC(tf.keras.metrics.AUC):
  def __init__(self, padding_label=-1, **kwargs):
    super().__init__(from_logits=False, **kwargs)
    self.padding_label = padding_label

  def update_state(self, y_true, y_pred, sample_weight=None):
    sample_weight = tf.cast(y_true >= 0, tf.float32)

    return super().update_state(y_true, y_pred, sample_weight)



class Ranking(models.Model):
  """A configurable ranking model.

  This class represents a sensible and reasonably flexible configuration for a
  ranking model that can be used for tasks such as CTR prediction.

  It can be customized as needed, and its constituent blocks can be changed by
  passing user-defined alternatives.

  For example:
  - Pass
    `feature_interaction = tfrs.layers.feature_interaction.DotInteraction()`
    to train a DLRM model, or pass
    ```
    feature_interaction = tf.keras.Sequential([
      tf.keras.layers.Concatenate(),
      tfrs.layers.feature_interaction.Cross()
    ])
    ```
    to train a DCN model.
  - Pass `task = tfrs.tasks.Ranking(loss=tf.keras.losses.BinaryCrossentropy())`
    to train a CTR prediction model, and
    `tfrs.tasks.Ranking(loss=tf.keras.losses.MeanSquaredError())` to train
    a rating prediction model.

  Changing these should cover a broad range of models, but this class is not
  intended to cover all possible use cases.  For full flexibility, inherit
  from `tfrs.models.Model` and provide your own implementations of
  the `compute_loss` and `call` methods.
  """

  def __init__(
      self,
      embedding_layer: tf.keras.layers.Layer,
      bottom_stack: Optional[tf.keras.layers.Layer] = None,
      feature_interaction: Optional[tf.keras.layers.Layer] = None,
      top_stack: Optional[tf.keras.layers.Layer] = None,
      concat_dense: bool = True,
      task: Optional[tasks.Task] = None) -> None:
    """Initializes the model.

    Args:
      embedding_layer: The embedding layer is applied to categorical features.
        It expects a string-to-tensor (or SparseTensor/RaggedTensor) dict as
        an input, and outputs a dictionary of string-to-tensor of feature_name,
        embedded_value pairs.
        {feature_name_i: tensor_i} -> {feature_name_i: emb(tensor_i)}.
      bottom_stack: The `bottom_stack` layer is applied to dense features before
        feature interaction. If None, an MLP with layer sizes [256, 64, 16] is
        used. For DLRM model, the output of bottom_stack should be of shape
        (batch_size, embedding dimension).
      feature_interaction: Feature interaction layer is applied to the
        `bottom_stack` output and sparse feature embeddings. If it is None,
        DotInteraction layer is used.
      top_stack: The `top_stack` layer is applied to the `feature_interaction`
        output. The output of top_stack should be in the range [0, 1]. If it is
        None, MLP with layer sizes [512, 256, 1] is used.
      concat_dense: Weather to concatenate the interaction output with dense
        embedding vector again before feeding into the top stack
      task: The task which the model should optimize for. Defaults to a
        `tfrs.tasks.Ranking` task with a binary cross-entropy loss, suitable for
        tasks like click prediction.
    """

    super().__init__()

    self._embedding_layer = embedding_layer
    self._concat_dense = concat_dense
    self._bottom_stack = (
        bottom_stack
        if bottom_stack
        else layers.blocks.MLP(units=[256, 64, 16], final_activation="relu")
    )
    self._top_stack = (
        top_stack
        if top_stack
        else layers.blocks.MLP(units=[512, 256, 1], final_activation="sigmoid")
    )
    self._feature_interaction = (
        feature_interaction
        if feature_interaction
        else feature_interaction_lib.DotInteraction()
    )

    if task is not None:
      self._task = task
    else:
      self._task = tasks.Ranking(
          loss=tf.keras.losses.BinaryCrossentropy(
              reduction=tf.keras.losses.Reduction.NONE, from_logits=False
          ),
          metrics=[MaskedAUC(name="masked_auc")],
      )

  def compute_loss(self,
                   inputs: Union[
                       # Tuple of (features, labels).
                           Dict[str, tf.Tensor],
                       # Tuple of (features, labels, sample weights).
                       Tuple[
                           Dict[str, tf.Tensor],
                           tf.Tensor,
                           Optional[tf.Tensor]
                       ]
                   ],
                   training: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Computes the loss and metrics of the model.

    Args:
      inputs: A data structure of tensors of the following format:
        ({"dense_features": dense_tensor,
          "sparse_features": sparse_tensors},
          label_tensor), or
        ({"dense_features": dense_tensor,
          "sparse_features": sparse_tensors},
          label_tensor,
          sample_weight tensor).
      training: Whether the model is in training mode.

    Returns:
      Loss tensor.

    Raises:
      ValueError if the the shape of the inputs is invalid.
    """

    # We need to work around a bug in mypy - tuple narrowing
    # based on length checks doesn't work.
    # See https://github.com/python/mypy/issues/1178 for details.
    features = inputs
    labels = inputs['clicked']
    outputs = self(features, training=training)

    loss = self._task(labels, outputs, sample_weight=None)
    loss = tf.reduce_mean(loss)
    return loss / tf.distribute.get_strategy().num_replicas_in_sync, labels, outputs

  def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Executes forward and backward pass, returns loss.

    Args:
      inputs: Model function inputs (features and labels).

    Returns:
      loss: Scalar tensor.
    """
    dense_features = inputs["dense_features"]
    sparse_features = inputs["sparse_features"]
    sparse_embeddings = self._embedding_layer(sparse_features)
    # Combine a dictionary into a vector and squeeze dimension from
    # (batch_size, 1, emb) to (batch_size, emb).
    sparse_embeddings = tf.nest.flatten(sparse_embeddings)

    sparse_embedding_vecs = [
        tf.squeeze(sparse_embedding) for sparse_embedding in sparse_embeddings
    ]
    #tf._logging.info(f'dense feature shape: {dense_features.shape}')
    dense_embedding_vec = self._bottom_stack(dense_features)
    #tf._logging.info(f'dense embedding vec shape: {dense_embedding_vec.shape}')
    interaction_args = sparse_embedding_vecs + [dense_embedding_vec]
    #tf._logging.info(f'interaction args shape: {interaction_args.shape}')
    interaction_output = self._feature_interaction(interaction_args)

    feature_interaction_output = interaction_output

    prediction = self._top_stack(feature_interaction_output)

    return tf.reshape(prediction, [-1])

  @property
  def embedding_trainable_variables(self) -> Sequence[tf.Variable]:
    """Returns trainable variables from embedding tables.

    When training a recommendation model with embedding tables, sometimes it's
    preferable to use separate optimizers/learning rates for embedding
    variables and dense variables.
    `tfrs.experimental.optimizers.CompositeOptimizer` can be used to apply
    different optimizers to embedding variables and the remaining variables.
    """
    return self._embedding_layer.trainable_variables

  @property
  def dense_trainable_variables(self) -> Sequence[tf.Variable]:
    """Returns all trainable variables that are not embeddings."""
    dense_vars = []
    for layer in self.layers:
      if layer != self._embedding_layer:
        dense_vars.extend(layer.trainable_variables)
    return dense_vars

