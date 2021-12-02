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

"""Pre-trains an ELECTRA model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import collections
import json
import time

import tensorflow.compat.v1 as tf
from tensorflow.core.protobuf import rewriter_config_pb2

from configure_pretraining import PretrainingConfig
from model import modeling
from model import optimization
from pretrain import pretrain_data
from pretrain import pretrain_helpers
from util import training_utils
from util import utils

import horovod.tensorflow as hvd
hvd.init()


class PretrainingModel(object):
    """Transformer pre-training using the replaced-token-detection task."""

    def __init__(self, config: PretrainingConfig, features, is_training, init_checkpoint):
        # Set up model config
        self._config = config
        self._bert_config = training_utils.get_bert_config(config)
        if config.debug:
            self._bert_config.num_hidden_layers = 3
            self._bert_config.hidden_size = 144
            self._bert_config.intermediate_size = 144 * 4
            self._bert_config.num_attention_heads = 4

        compute_type = modeling.infer_dtype(config.use_fp16)
        custom_getter = modeling.get_custom_getter(compute_type)

        with tf.variable_scope(tf.get_variable_scope(), custom_getter=custom_getter):
            # Mask the input
            masked_inputs = pretrain_helpers.mask(
                config,
                pretrain_data.features_to_inputs(features),
                config.mask_prob
            )

            # Generator
            embedding_size = self._bert_config.hidden_size if config.embedding_size is None else config.embedding_size
            if config.uniform_generator:
                mlm_output = self._get_masked_lm_output(masked_inputs, None)
            elif config.electra_objective and config.untied_generator:
                generator = self._build_transformer(
                    name="generator",
                    inputs=masked_inputs,
                    is_training=is_training,
                    use_fp16=config.use_fp16,
                    bert_config=get_generator_config(config, self._bert_config),
                    embedding_size=None if config.untied_generator_embeddings else embedding_size,
                    untied_embeddings=config.untied_generator_embeddings
                )
                mlm_output = self._get_masked_lm_output(masked_inputs, generator)
            else:
                generator = self._build_transformer(
                    name="electra",
                    inputs=masked_inputs,
                    is_training=is_training,
                    use_fp16=config.use_fp16,
                    embedding_size=embedding_size
                )
                mlm_output = self._get_masked_lm_output(masked_inputs, generator)
            fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)
            self.mlm_output = mlm_output
            self.total_loss = config.gen_weight * mlm_output.loss

            utils.log("Generator is built!")

            # Discriminator
            self.disc_output = None
            if config.electra_objective:
                discriminator = self._build_transformer(
                    name="electra",
                    inputs=fake_data.inputs,
                    is_training=is_training,
                    use_fp16=config.use_fp16,
                    embedding_size=embedding_size
                )
                utils.log("Discriminator is built!")
                self.disc_output = self._get_discriminator_output(
                    inputs=fake_data.inputs,
                    discriminator=discriminator,
                    labels=fake_data.is_fake_tokens
                )
                self.total_loss += config.disc_weight * self.disc_output.loss

        if init_checkpoint and hvd.rank() == 0:
            print("Loading checkpoint", init_checkpoint)
            assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
                tvars=tf.trainable_variables(),
                init_checkpoint=init_checkpoint,
                prefix=""
            )
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # Evaluation
        eval_fn_inputs = {
            "input_ids": masked_inputs.input_ids,
            "masked_lm_preds": mlm_output.preds,
            "mlm_loss": mlm_output.per_example_loss,
            "masked_lm_ids": masked_inputs.masked_lm_ids,
            "masked_lm_weights": masked_inputs.masked_lm_weights,
            "input_mask": masked_inputs.input_mask
        }
        if config.electra_objective:
            eval_fn_inputs.update({
                "disc_loss": self.disc_output.per_example_loss,
                "disc_labels": self.disc_output.labels,
                "disc_probs": self.disc_output.probs,
                "disc_preds": self.disc_output.preds,
                "sampled_tokids": tf.argmax(fake_data.sampled_tokens, -1, output_type=tf.int32)
            })
        eval_fn_keys = eval_fn_inputs.keys()
        eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]

        def metric_fn(*args):
            """Computes the loss and accuracy of the model."""
            d = dict(zip(eval_fn_keys, args))
            metrics = dict()
            metrics["masked_lm_accuracy"] = tf.metrics.accuracy(
                labels=tf.reshape(d["masked_lm_ids"], [-1]),
                predictions=tf.reshape(d["masked_lm_preds"], [-1]),
                weights=tf.reshape(d["masked_lm_weights"], [-1])
            )
            metrics["masked_lm_loss"] = tf.metrics.mean(
                values=tf.reshape(d["mlm_loss"], [-1]),
                weights=tf.reshape(d["masked_lm_weights"], [-1])
            )
            if config.electra_objective:
                metrics["sampled_masked_lm_accuracy"] = tf.metrics.accuracy(
                    labels=tf.reshape(d["masked_lm_ids"], [-1]),
                    predictions=tf.reshape(d["sampled_tokids"], [-1]),
                    weights=tf.reshape(d["masked_lm_weights"], [-1])
                )
                if config.disc_weight > 0:
                    metrics["disc_loss"] = tf.metrics.mean(d["disc_loss"])
                    metrics["disc_auc"] = tf.metrics.auc(
                        d["disc_labels"] * d["input_mask"],
                        d["disc_probs"] * tf.cast(d["input_mask"], tf.float32)
                    )
                    metrics["disc_accuracy"] = tf.metrics.accuracy(
                        labels=d["disc_labels"],
                        predictions=d["disc_preds"],
                        weights=d["input_mask"]
                    )
                    metrics["disc_precision"] = tf.metrics.accuracy(
                        labels=d["disc_labels"],
                        predictions=d["disc_preds"],
                        weights=d["disc_preds"] * d["input_mask"]
                    )
                    metrics["disc_recall"] = tf.metrics.accuracy(
                        labels=d["disc_labels"],
                        predictions=d["disc_preds"],
                        weights=d["disc_labels"] * d["input_mask"]
                    )

            return metrics

        self.eval_metrics = (metric_fn, eval_fn_values)

    def _get_masked_lm_output(self, inputs: pretrain_data.Inputs, model):
        """Masked language modeling softmax layer."""
        masked_lm_weights = inputs.masked_lm_weights
        with tf.variable_scope("generator_predictions"):
            if self._config.uniform_generator:
                logits = tf.zeros(self._bert_config.vocab_size)
                logits_tiled = tf.zeros(modeling.get_shape_list(inputs.masked_lm_ids) + [self._bert_config.vocab_size])
                logits_tiled += tf.reshape(logits, [1, 1, self._bert_config.vocab_size])
                logits = logits_tiled
            else:
                relevant_hidden = pretrain_helpers.gather_positions(
                    model.get_sequence_output(), inputs.masked_lm_positions)
                hidden = tf.layers.dense(
                    relevant_hidden,
                    units=modeling.get_shape_list(model.get_embedding_table())[-1],
                    activation=modeling.get_activation(self._bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(self._bert_config.initializer_range)
                )
                hidden = modeling.layer_norm(hidden)
                output_bias = tf.get_variable(
                    "output_bias",
                    shape=[self._bert_config.vocab_size],
                    initializer=tf.zeros_initializer()
                )
                # logits = tf.matmul(hidden, model.get_embedding_table(), transpose_b=True)
                logits = tf.matmul(hidden, tf.cast(model.get_embedding_table(), hidden.dtype), transpose_b=True)
                logits = tf.nn.bias_add(logits, tf.cast(output_bias, logits.dtype))

            logits = tf.cast(logits, tf.float32)  # чтоб выход sofmax-а был стабильным
            oh_labels = tf.one_hot(inputs.masked_lm_ids, depth=self._bert_config.vocab_size, dtype=logits.dtype)

            probs = tf.nn.softmax(logits)
            log_probs = tf.nn.log_softmax(logits)
            label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

            # numerator = tf.reduce_sum(inputs.masked_lm_weights * label_log_probs)
            # denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
            numerator = tf.reduce_sum(tf.cast(inputs.masked_lm_weights, label_log_probs.dtype) * label_log_probs)
            denominator = tf.cast(tf.reduce_sum(masked_lm_weights) + 1e-6, numerator.dtype)
            loss = numerator / denominator
            preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

            MLMOutput = collections.namedtuple("MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"])
            return MLMOutput(
                logits=logits,
                probs=probs,
                per_example_loss=label_log_probs,
                loss=loss,
                preds=preds
            )

    def _get_discriminator_output(self, inputs, discriminator, labels):
        """Discriminator binary classifier."""
        with tf.variable_scope("discriminator_predictions"):
            hidden = tf.layers.dense(
                discriminator.get_sequence_output(),
                units=self._bert_config.hidden_size,
                activation=modeling.get_activation(self._bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(self._bert_config.initializer_range)
            )
            logits = tf.squeeze(tf.layers.dense(hidden, units=1), -1)
            logits = tf.cast(logits, tf.float32)
            weights = tf.cast(inputs.input_mask, tf.float32)
            labelsf = tf.cast(labels, tf.float32)
            # weights = tf.cast(inputs.input_mask, logits.dtype)
            # labelsf = tf.cast(labels, logits.dtype)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labelsf) * weights
            per_example_loss = (tf.reduce_sum(losses, axis=-1) / (1e-6 + tf.reduce_sum(weights, axis=-1)))
            loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
            probs = tf.nn.sigmoid(logits)
            preds = tf.cast(tf.round((tf.sign(logits) + 1) / 2), tf.int32)
            DiscOutput = collections.namedtuple("DiscOutput", ["loss", "per_example_loss", "probs", "preds", "labels"])
            return DiscOutput(
                loss=loss,
                per_example_loss=per_example_loss,
                probs=probs,
                preds=preds,
                labels=labels,
            )

    def _get_fake_data(self, inputs, mlm_logits):
        """Sample from the generator to create corrupted input."""
        inputs = pretrain_helpers.unmask(inputs)
        disallow = tf.one_hot(
            inputs.masked_lm_ids,
            depth=self._bert_config.vocab_size,
            dtype=tf.float32
        ) if self._config.disallow_correct else None
        sampled_tokens = tf.stop_gradient(
            pretrain_helpers.sample_from_softmax(mlm_logits / self._config.temperature, disallow=disallow)
        )
        sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
        updated_input_ids, masked = pretrain_helpers.scatter_update(
            inputs.input_ids,
            sampled_tokids,
            inputs.masked_lm_positions
        )
        labels = masked * (1 - tf.cast(tf.equal(updated_input_ids, inputs.input_ids), tf.int32))
        updated_inputs = pretrain_data.get_updated_inputs(inputs, input_ids=updated_input_ids)
        FakedData = collections.namedtuple("FakedData", ["inputs", "is_fake_tokens", "sampled_tokens"])
        return FakedData(inputs=updated_inputs, is_fake_tokens=labels,
                         sampled_tokens=sampled_tokens)

    def _build_transformer(
            self,
            name,
            inputs: pretrain_data.Inputs,
            is_training,
            use_fp16=False,
            bert_config=None,
            **kwargs
    ):
        """Build a transformer encoder network."""
        if bert_config is None:
            bert_config = self._bert_config
        return modeling.BertModel(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=inputs.input_ids,
            input_mask=inputs.input_mask,
            token_type_ids=inputs.segment_ids,
            use_one_hot_embeddings=self._config.use_tpu,
            scope=name,
            use_fp16=use_fp16,
            **kwargs
        )


def get_generator_config(config: PretrainingConfig, bert_config: modeling.BertConfig):
    """Get model config for the generator network."""
    gen_config = modeling.BertConfig.from_dict(bert_config.to_dict())
    gen_config.hidden_size = int(round(
        bert_config.hidden_size * config.generator_hidden_size))
    gen_config.num_hidden_layers = int(round(
        bert_config.num_hidden_layers * config.generator_layers))
    gen_config.intermediate_size = 4 * gen_config.hidden_size
    gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
    return gen_config


def model_fn_builder(config: PretrainingConfig):
    """Build the model for training."""

    def model_fn(features, labels, mode, params):
        """Build the model for training."""
        model = PretrainingModel(
            config=config,
            features=features,
            is_training=mode == tf.estimator.ModeKeys.TRAIN,
            init_checkpoint=config.init_checkpoint
        )
        utils.log("Model is built!")
        to_log = {
            "gen_loss": model.mlm_output.loss,
            "disc_loss": model.disc_output.loss,
            "total_loss": model.total_loss
        }
        if mode == tf.estimator.ModeKeys.TRAIN:

            tf.summary.scalar('gen_loss', model.mlm_output.loss)
            tf.summary.scalar('disc_loss', model.disc_output.loss)
            tf.summary.scalar('total_loss', model.total_loss)

            lr_multiplier = hvd.size() if config.scale_lr else 1
            train_op = optimization.create_optimizer(
                loss=model.total_loss,
                learning_rate=config.learning_rate * lr_multiplier,
                num_train_steps=config.num_train_steps,
                weight_decay_rate=config.weight_decay_rate,
                warmup_steps=config.num_warmup_steps,
                warmup_proportion=0,
                lr_decay_power=config.lr_decay_power,
                layerwise_lr_decay_power=-1,
                n_transformer_layers=None,
                hvd=hvd,
                use_fp16=config.use_fp16,
                num_accumulation_steps=config.num_accumulation_steps,
                allreduce_post_accumulation=config.allreduce_post_accumulation
            )
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=model.total_loss,
                train_op=train_op,
                training_hooks=[
                    training_utils.ETAHook(
                        to_log=to_log,
                        n_steps=config.num_train_steps,
                        iterations_per_loop=config.iterations_per_loop,
                        on_tpu=False,
                        log_every=1,
                        is_training=True
                    )
                ]
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=model.total_loss,
                eval_metrics=model.eval_metrics,
                evaluation_hooks=[
                    training_utils.ETAHook(
                        to_log=to_log,
                        n_steps=config.num_eval_steps,
                        iterations_per_loop=config.iterations_per_loop,
                        on_tpu=False,
                        log_every=1,
                        is_training=False
                    )
                ]
            )
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported")
        return output_spec

    return model_fn


def train_or_eval(config: PretrainingConfig):
    """Run pre-training or evaluate the pre-trained model."""
    if config.do_train == config.do_eval:
        raise ValueError("Exactly one of `do_train` or `do_eval` must be True.")
    if config.debug and config.do_train:
        utils.rmkdir(config.model_dir)
    utils.heading("Config:")
    utils.log_config(config)

    # session config
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())  # one gpu per process
    # session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1  # xla
    # session_config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.NO_MEM_OPT  # xla

    # run config
    # согласно примеру: https://gist.github.com/alsrgv/34a32f30292f4e2c1fa29ec0d65dea26
    # model_dir = config.model_dir if hvd.rank() == 0 else None
    # UPD: если model_dir == None, то Estimator по умолчанию сохраняет чекпоинты в /tmp, что сжирает системный диск

    run_config = tf.estimator.RunConfig(
        model_dir=config.model_dir,
        session_config=session_config,
        save_checkpoints_steps=config.save_checkpoints_steps if hvd.rank() == 0 else None,
        save_summary_steps=100 if hvd.rank() == 0 else 0,
        keep_checkpoint_max=config.keep_checkpoint_max,
        log_step_count_steps=10000
    )

    # model_fn
    model_fn = model_fn_builder(config=config)

    # training hooks
    training_hooks = []

    if hvd.size() > 1:
        training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    # estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    if config.do_train:
        utils.heading("Running training")
        input_fn = pretrain_data.get_input_fn(
            pretrain_tfrecords=config.pretrain_tfrecords,
            max_seq_length=config.max_seq_length,
            batch_size=config.train_batch_size,
            is_training=True,
            hvd=hvd,
            num_cpu_threads=8
        )
        estimator.train(
            input_fn=input_fn,
            hooks=training_hooks,
            max_steps=config.num_train_steps
        )
    if config.do_eval:
        utils.heading("Running evaluation")
        input_fn = pretrain_data.get_input_fn(
            pretrain_tfrecords=config.pretrain_tfrecords,
            max_seq_length=config.max_seq_length,
            batch_size=config.eval_batch_size,
            is_training=False,
            hvd=hvd,
            num_cpu_threads=8
        )
        result = estimator.evaluate(
            input_fn=input_fn,
            steps=config.num_eval_steps
        )
        for key in sorted(result.keys()):
            utils.log("  {:} = {:}".format(key, str(result[key])))
        return result


# def train_one_step(config: PretrainingConfig):
#     """Builds an ELECTRA model an trains it for one step; useful for debugging."""
#     train_input_fn = pretrain_data.get_input_fn(config, True)
#     features = tf.data.make_one_shot_iterator(train_input_fn(dict(
#         batch_size=config.train_batch_size))).get_next()
#     model = PretrainingModel(config, features, True)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         utils.log(sess.run(model.total_loss))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Location of data files")
    parser.add_argument("--model-dir", required=True, help="Location of models")
    parser.add_argument("--model-name", required=True, help="The name of the model being fine-tuned.")
    parser.add_argument("--init-checkpoint", required=False, default=None, help="Path to init checkpoint")
    parser.add_argument("--hparams", default="{}", help="JSON dict of model hyperparameters.")
    args = parser.parse_args()

    if args.hparams.endswith(".json"):
        hparams = utils.load_json(args.hparams)
    else:
        hparams = json.loads(args.hparams)

    if hvd.rank() == 0:
        print("command line args:")
        print(args)
        print("hparams:")
        print(hparams)

    tf.logging.set_verbosity(tf.logging.ERROR)

    config = PretrainingConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        model_name=args.model_name,
        init_checkpoint=args.init_checkpoint,
        **hparams
    )

    if config.use_fp16:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

    train_or_eval(config=config)


if __name__ == "__main__":
    main()
