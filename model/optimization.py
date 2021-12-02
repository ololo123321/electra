# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Functions and classes related to optimization (weight updates).
Modified from the original BERT code to allow for having separate learning
rates for different layers of the network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import tensorflow.compat.v1 as tf
import tensorflow.contrib as tf_contrib
from horovod.tensorflow.compression import Compression


def create_optimizer(
        loss,
        learning_rate,
        num_train_steps,
        weight_decay_rate=0.0,
        warmup_steps=0,
        warmup_proportion=0,
        lr_decay_power=1.0,
        layerwise_lr_decay_power=-1,
        n_transformer_layers=None,
        hvd=None,
        use_fp16=False,
        num_accumulation_steps=1,
        allreduce_post_accumulation=False
):
    """
    Creates an optimizer and training op.
    """
    compression = Compression.fp16 if use_fp16 else Compression.none

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=lr_decay_power,
        cycle=False
    )
    warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)
    learning_rate *= tf.minimum(1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))

    if layerwise_lr_decay_power > 0:
        learning_rate = _get_layer_lrs(learning_rate, layerwise_lr_decay_power, n_transformer_layers)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=weight_decay_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
    )

    if hvd is not None and (num_accumulation_steps == 1 or (not allreduce_post_accumulation)):
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            sparse_as_dense=True,
            compression=compression
        )
    if use_fp16:
        loss_scale_manager = tf_contrib.mixed_precision.ExponentialUpdateLossScaleManager(
            init_loss_scale=2 ** 32,
            incr_every_n_steps=1000,
            decr_every_n_nan_or_inf=2,
            decr_ratio=0.5
        )
        optimizer = tf_contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)

    tvars = tf.trainable_variables()
    # if hvd.rank() == 0:
    #     print("*****Trainable variables*****")
    #     for v in tvars:
    #         print(v)
    #     print("*****************************")

    grads_and_vars = optimizer.compute_gradients(loss * 1.0 / num_accumulation_steps, tvars)

    if num_accumulation_steps > 1:
        local_step = tf.get_variable(
            name="local_step",
            shape=[],
            dtype=tf.int32,
            trainable=False,
            initializer=tf.zeros_initializer()
        )
        batch_finite = tf.get_variable(
            name="batch_finite",
            shape=[],
            dtype=tf.bool,
            trainable=False,
            initializer=tf.ones_initializer()
        )
        accum_vars = [
            tf.get_variable(
                name=tvar.name.split(":")[0] + "/accum",
                shape=tvar.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()
            ) for tvar in tvars
        ]

        reset_step = tf.cast(tf.math.equal(local_step % num_accumulation_steps, 0), dtype=tf.bool)
        local_step = tf.cond(
            reset_step,
            lambda: local_step.assign(tf.ones_like(local_step)),
            lambda: local_step.assign_add(1)
        )

        grads, tvars, accum_vars = zip(*[
            (g, v, g_acc) for (g, v), g_acc in zip(grads_and_vars, accum_vars) if g is not None
        ])

        if use_fp16:
            # оказывается, это условие может быть кучу перых шагов false, а затем будет всю дорогу true
            all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads])
            # если возобновить обучение из чекпоинта, то снова первые дохера шагов градиенты будут накапливаться,
            # что повлечёт скачок лосса
            # сделано так для продолжения обучения
            # all_are_finite = tf.constant(True, dtype=tf.bool)
        else:
            all_are_finite = tf.constant(True, dtype=tf.bool)

        batch_finite = tf.cond(
            reset_step,
            lambda: batch_finite.assign(tf.math.logical_and(tf.constant(True, dtype=tf.bool), all_are_finite)),
            lambda: batch_finite.assign(tf.math.logical_and(batch_finite, all_are_finite))
        )

        # This is how the model was pre-trained.
        # ensure global norm is a finite number
        # to prevent clip_by_global_norm from having a hizzy fit.
        (clipped_grads, _) = tf.clip_by_global_norm(
            grads, clip_norm=1.0,
            use_norm=tf.cond(
                all_are_finite,
                lambda: tf.global_norm(grads),
                lambda: tf.constant(1.0)
            )
        )

        accum_vars = tf.cond(
            reset_step,
            lambda: [v.assign(grad) for v, grad in zip(accum_vars, clipped_grads)],
            lambda: [v.assign_add(grad) for v, grad in zip(accum_vars, clipped_grads)]
        )

        def update(accum_vars):
            if allreduce_post_accumulation and hvd is not None:
                accum_vars = [
                    hvd.allreduce(
                        tf.convert_to_tensor(accum_var),
                        compression=compression
                    )
                    if isinstance(accum_var, tf.IndexedSlices)
                    else hvd.allreduce(
                        accum_var,
                        compression=compression
                    )
                    for accum_var in accum_vars
                ]
            return optimizer.apply_gradients(list(zip(accum_vars, tvars)), global_step=global_step)

        update_step = tf.identity(
            tf.cast(tf.math.equal(local_step % num_accumulation_steps, 0), dtype=tf.bool),
            name="update_step"
        )
        update_op = tf.cond(
            update_step,
            lambda: update(accum_vars),
            lambda: tf.no_op()
        )
        new_global_step = tf.cond(
            tf.math.logical_and(update_step, tf.cast(hvd.allreduce(tf.cast(batch_finite, tf.int32)), tf.bool)),
            lambda: global_step + 1,
            lambda: global_step
        )
        new_global_step = tf.identity(new_global_step, name='step_update')
        train_op = tf.group(update_op, [global_step.assign(new_global_step)])
    else:
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        grads, tvars = list(zip(*grads_and_vars))

        if use_fp16:
            all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads])
        else:
            all_are_finite = tf.constant(True, dtype=tf.bool)

        # This is how the model was pre-trained.
        # ensure global norm is a finite number
        # to prevent clip_by_global_norm from having a hizzy fit.
        (clipped_grads, _) = tf.clip_by_global_norm(
            grads, clip_norm=1.0,
            use_norm=tf.cond(
                all_are_finite,
                lambda: tf.global_norm(grads),
                lambda: tf.constant(1.0)
            )
        )
        train_op = optimizer.apply_gradients(list(zip(clipped_grads, tvars)), global_step=global_step)
        new_global_step = tf.cond(all_are_finite, lambda: global_step + 1, lambda: global_step)
        new_global_step = tf.identity(new_global_step, name='step_update')
        train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _apply_gradients(self, grads_and_vars, learning_rate):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))
            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self.weight_decay_rate > 0:
                if self._do_use_weight_decay(param_name):
                    update += self.weight_decay_rate * param

            update_with_lr = learning_rate * update
            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])

        return assignments

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        if isinstance(self.learning_rate, dict):
            key_to_grads_and_vars = {}
            for grad, var in grads_and_vars:
                update_for_var = False
                for key in self.learning_rate:
                    if key in var.name:
                        update_for_var = True
                        if key not in key_to_grads_and_vars:
                            key_to_grads_and_vars[key] = []
                        key_to_grads_and_vars[key].append((grad, var))
                if not update_for_var:
                    raise ValueError("No learning rate specified for variable", var)
            assignments = []
            for key, key_grads_and_vars in key_to_grads_and_vars.items():
                assignments += self._apply_gradients(key_grads_and_vars,
                                                     self.learning_rate[key])
        else:
            assignments = self._apply_gradients(grads_and_vars, self.learning_rate)
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


def _get_layer_lrs(learning_rate, layer_decay, n_layers):
    """Have lower learning rates for layers closer to the input."""
    key_to_depths = collections.OrderedDict({
        "/embeddings/": 0,
        "/embeddings_project/": 0,
        "task_specific/": n_layers + 2,
    })
    for layer in range(n_layers):
        key_to_depths["encoder/layer_" + str(layer) + "/"] = layer + 1
    return {
        key: learning_rate * (layer_decay ** (n_layers + 2 - depth))
        for key, depth in key_to_depths.items()
    }
