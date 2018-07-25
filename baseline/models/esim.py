import tensorflow as tf
import numpy as np
from datetime import datetime
from utils import data_utils
from sklearn.metrics import accuracy_score, log_loss

def attention_softmax3d(values):
    """
    Performs a softmax over the attention values.

    :param values: 3d tensor with raw values
    :return: a tensor with the same shape as `values`
    """
    original_shape = tf.shape(values)
    num_units = original_shape[2]
    reshaped_values = tf.reshape(values, tf.stack([-1, num_units]))
    softmaxes = tf.nn.softmax(reshaped_values)
    return tf.reshape(softmaxes, original_shape)

def mask_3d(values, seq_lens, mask_value, dimension=2):
    """
    Given a batch of matrices, each with m x n, mask the values in each
    row after the position indicated in seq_lens

    :param values: tensor with shape (batch_size, m, n)
    :param seq_lens: tensor with shape (batch_size) containing the lengths
        that should be limited
    :param mask_value: scalar value to assign to items after the limited lengths
    :param dimension: over which dimension to mask values
    :return: a tensor with the same shape as `values`
    """
    if dimension == 1:
        values = tf.transpose(values, [0, 2, 1])
    time_step1 = tf.shape(values)[1]
    time_step2 = tf.shape(values)[2]

    ones = tf.ones_like(values, dtype=tf.float32)
    pad_values = mask_value * ones
    mask = tf.sequence_mask(seq_lens, time_step2)

    mask3d = tf.expand_dims(mask, 1)
    mask3d = tf.tile(mask3d, (1, time_step1, 1))

    masked = tf.where(mask3d, values, pad_values)

    if dimension == 1:
        masked = tf.transpose(masked, [0, 2, 1])

    return masked

class ESIM(object):
    """
    Enhanced Sequential Inference Model
    """

    def __init__(self, num_classes, seq_len, word_embeddings, hparams):
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.num_units = hparams["num_units"]
        self.input_dropout = hparams["input_dropout"]
        self.output_dropout = hparams["output_dropout"]
        self.state_dropout = hparams["state_dropout"]
        self.hidden_layers = hparams["hidden_layers"]
        self.hidden_units = hparams["hidden_units"]
        self.hidden_dropout = hparams["hidden_dropout"]
        self.lr = hparams["lr"]
        self.l2_reg_lambda = hparams["l2_reg_lambda"]

        self.batch_size = hparams["batch_size"]
        self.num_epochs = hparams["num_epochs"]
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.sent1 = tf.placeholder(tf.int32, [None, seq_len], 'sent1')
        self.sent2 = tf.placeholder(tf.int32, [None, seq_len], 'sent2')
        self.sent1_len = tf.placeholder(tf.int32, [None], 'sent1_len')
        self.sent2_len = tf.placeholder(tf.int32, [None], 'sent2_len')
        self.labels = tf.placeholder(tf.int32, [None], 'label')
        self.input_dropout_ph = tf.placeholder(tf.float32, [], 'input_dropout')
        self.output_dropout_ph = tf.placeholder(tf.float32, [], 'output_dropout')
        self.state_dropout_ph = tf.placeholder(tf.float32, [], 'state_dropout')
        self.hidden_dropout_ph = tf.placeholder(tf.float32, [], 'hidden_dropout')

        with tf.device('/cpu:0'), tf.name_scope("word_embedding"):
            W = tf.Variable(word_embeddings, trainable=False, dtype=tf.float32, name="W")
            embedded_words1 = tf.nn.embedding_lookup(W, self.sent1)
            embedded_words2 = tf.nn.embedding_lookup(W, self.sent2)

        self.alpha, self.beta = self.attend(embedded_words1, embedded_words2,
                                            self.sent1_len, self.sent2_len)
        self.v1 = self.compare(embedded_words1, self.beta, self.sent1_len)
        self.v2 = self.compare(embedded_words2, self.alpha, self.sent2_len, True)
        self.logits = self.aggregate(self.v1, self.v2)
        self.probas = tf.nn.softmax(self.logits, -1, "probas")
        self.answer = tf.argmax(self.logits, 1, 'answer')

        hits = tf.equal(tf.cast(self.answer, tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(hits, tf.float32), name="accuracy")

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labels)
        weights = [v for v in tf.trainable_variables()
                   if 'weight' in v.name]
        l2_loss = tf.contrib.layers.apply_regularization(
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
            weights_list=weights)
        self.loss = tf.add(tf.reduce_mean(cross_entropy), l2_loss, "loss")

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

    def attend(self, sent1, sent2, sent1_len, sent2_len):
        """
        Compute inter-sentence attention.

        :param sent1: tensor in shape (batch_size, time_steps, embeded_size)
        :param sent2: tensor in shape (batch_size, time_steps, embeded_size)
        :param sent1_len: tensor in shape (batch_size)
        :param sent2_len: tensor in shape (batch_size)
        :return: a tuple of 3-d tensors (alpha, beta)
        """
        with tf.variable_scope('inter-attention') as self.attend_scope:
            repr1 = self._apply_lstm(sent1, sent1_len, self.attend_scope)
            repr2 = self._apply_lstm(sent2, sent2_len, self.attend_scope, True)
            repr2 = tf.transpose(repr2, [0, 2, 1])

            attentions = tf.matmul(repr1, repr2)
            masked = mask_3d(attentions, sent2_len, -np.inf)
            att_sent1 = attention_softmax3d(masked)

            att_transposed = tf.transpose(attentions, [0, 2, 1])
            masked = mask_3d(att_transposed, sent1_len, -np.inf)
            att_sent2 = attention_softmax3d(masked)

            alpha = tf.matmul(att_sent1, sent1, name="alpha")
            beta = tf.matmul(att_sent2, sent2, name="beta")

        return alpha, beta

    def compare(self, sent, soft_alignment, seq_len, reuse_weights=False):
        """
        Compare one sentence to its soft alignment.

        :param sent: embedded sentence in shape (batch_size, time_steps, num_units)
        :param soft_alignment: tensor in shape (batch_size, time_steps, num_units)
        :param seq_len: tensor in shape (batch_size)
        :param reuse_weights: whether to reuse weights inside the same scope
        :return: a tensor in shape (batch, time_steps, num_units)
        """
        with tf.variable_scope('comparision', reuse=reuse_weights) as self.compare_scope:
            inputs = [sent, soft_alignment, sent - soft_alignment,
                      sent * soft_alignment]
            sent_and_alignment = tf.concat(axis=2, values=inputs)

            output = self._apply_lstm(
                sent_and_alignment, seq_len, self.compare_scope, reuse_weights)

        return output

    def aggregate(self, v1, v2):
        """
        Aggregate the representations induced.

        :param v1: tensor in shape (batch, time_steps, num_units)
        :param v2: tensor in shape (batch, time_steps, num_units)
        :return: logits over classes in shape (batch, num_classes)
        """
        inputs = self._create_aggregate_input(v1, v2)
        with tf.variable_scope('aggregation') as self.aggregate_scope:
            initializer = tf.random_normal_initializer(0.0, 0.1)
            with tf.variable_scope('linear'):
                shape = [self.hidden_units, self.num_classes]
                weights = tf.get_variable('weights', shape, initializer=initializer)
                bias = tf.get_variable('bias', [self.num_classes],
                                       initializer=tf.zeros_initializer())

            pre_logits = self._apply_feedforward(inputs, [self.hidden_units] * self.hidden_layers,
                                                 self.aggregate_scope)
            logits = tf.nn.xw_plus_b(pre_logits, weights, bias)

        return logits

    def _create_aggregate_input(self, v1, v2):
        """
        Create the input to the aggregate step.

        :param v1: tensor in shape (batch, time_steps, num_units)
        :param v2: tensor in shape (batch, time_steps, num_units)
        :return: a tensor in shape (batch, num_units * 4)
        """
        v1 = mask_3d(v1, self.sent1_len, 0, 1)
        v2 = mask_3d(v2, self.sent2_len, 0, 1)
        v1_sum = tf.reduce_sum(v1, 1)
        v2_sum = tf.reduce_sum(v2, 1)
        v1_max = tf.reduce_max(v1, 1)
        v2_max = tf.reduce_max(v2, 1)

        return tf.concat(axis=1, values=[v1_sum, v2_sum, v1_max, v2_max])

    def _apply_feedforward(self, inputs, hidden_units, scope, reuse_weights=False):
        """
        Apply feedforward network to the given inputs.

        :param inputs: tensor in shape (batch, input_dim)
        :param hidden_units: list containing the number of hidden units in each layer
        :param scope: tensorflow variable scope
        :param reuse_weights: whether to reuse weights inside the same scope
        :return: a tensor in shape (batch, hidden_units[-1])
        """
        scope = scope or 'feedforword'
        outputs = inputs
        with tf.variable_scope(scope, reuse=reuse_weights):
            for i, hidden_unit in enumerate(hidden_units):
                with tf.variable_scope('layer%d' % i):
                    outputs = tf.layers.dense(outputs, hidden_unit, tf.nn.relu)
                    outputs = tf.nn.dropout(outputs, self.hidden_dropout_ph)
        return outputs

    def _apply_lstm(self, inputs, length, scope=None, reuse_weights=False):
        """
        Apply LSTM to the given inputs.

        :param inputs: tensor in shape (batch, time_steps, embedded_size)
        :param length: tensor in shape (batch)
        :param scope: tensorflow variable scope
        :param reuse_weights: whether to reuse weights inside the same scope
        :return: a tensor in shape (batch, time_steps, 2 * num_units)
        """
        scope_name = scope or 'lstm'
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(scope_name, reuse=reuse_weights) as lstm_scope:
            fw_lstm = tf.nn.rnn_cell.LSTMCell(self.num_units, initializer=initializer)
            bw_lstm = tf.nn.rnn_cell.LSTMCell(self.num_units, initializer=initializer)
            fw_lstm = tf.nn.rnn_cell.DropoutWrapper(fw_lstm,
                                                    self.input_dropout_ph,
                                                    self.output_dropout_ph,
                                                    self.state_dropout_ph)
            bw_lstm = tf.nn.rnn_cell.DropoutWrapper(bw_lstm,
                                                    self.input_dropout_ph,
                                                    self.output_dropout_ph,
                                                    self.state_dropout_ph)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, inputs,
                                                         dtype=tf.float32,
                                                         sequence_length=length,
                                                         scope=lstm_scope)
            output_fw, output_bw = outputs
            concat_outputs = tf.concat(axis=2, values=[output_fw, output_bw])
        return concat_outputs

    def init(self, sess):
        """
        Initialize all tensorflow variables.

        :param sess: tensorflow session
        """
        sess.run(tf.global_variables_initializer())

    def _create_feed_dict(self, sent1, sent2, sent1_len, sent2_len, labels=None,
                          input_dropout=1.0, output_dropout=1.0,
                          state_dropout=1.0, hidden_dropout=1.0):
        """
        Create feed dict for training.
        """
        feed_dict = {
            self.sent1: sent1,
            self.sent2: sent2,
            self.sent1_len: sent1_len,
            self.sent2_len: sent2_len,
            self.input_dropout_ph: input_dropout,
            self.output_dropout_ph: output_dropout,
            self.state_dropout_ph: state_dropout,
            self.hidden_dropout_ph: hidden_dropout
        }
        if labels is not None:
            feed_dict[self.labels] = labels
        return feed_dict

    def fit(self, sess, train, valid=None):
        """
        Train the model.

        :param sess: tensorflow session
        :param train: training dataset
        :param valid: validation dataset
        """
        train_batches = data_utils.batch_iter(train, self.batch_size, self.num_epochs)
        data_size = len(train)
        num_batches_per_epoch = int((data_size - 1) / self.batch_size) + 1
        best_acc = 0.0
        best_loss = 1e10
        best_epoch = 0
        for batch in train_batches:
            sent1, sent2, sent1_len, sent2_len, labels = zip(*batch)
            feeds = self._create_feed_dict(sent1, sent2, sent1_len, sent2_len, labels,
                                           self.input_dropout, self.output_dropout,
                                           self.state_dropout, self.hidden_dropout)
            ops = [self.train_op, self.global_step, self.loss, self.accuracy]
            _, step, loss, acc = sess.run(ops, feed_dict=feeds)
            time_str = datetime.now().isoformat()
            print("{}: step {} loss {:g} acc {:g}".format(time_str, step, loss, acc))
            if (step % num_batches_per_epoch == 0) and (valid is not None):
                print("\nValidation:")
                print("previous best epoch {}, loss {:g}, acc {:g}".format(
                    best_epoch, best_loss, best_acc))
                probas = self.predict(sess, valid[0])
                valid_loss = log_loss(valid[1], probas)
                valid_acc = accuracy_score(valid[1], np.argmax(probas, 1))
                time_str = datetime.now().isoformat()
                print("{}: epoch {} loss {:g} acc {:g}".format(
                    time_str, step // num_batches_per_epoch, valid_loss, valid_acc))
                print("")
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_loss = valid_loss
                    best_epoch = step // num_batches_per_epoch
                if step // num_batches_per_epoch - best_epoch > 3:
                    break
        return best_epoch, best_loss, best_acc

    def predict(self, sess, test):
        batches = data_utils.batch_iter(test, self.batch_size, 1, shuffle=False)
        all_probas = []
        for batch in batches:
            sent1, sent2, sent1_len, sent2_len = zip(*batch)
            feeds = self._create_feed_dict(sent1, sent2, sent1_len, sent2_len)
            probas = sess.run(self.probas, feed_dict=feeds)
            all_probas.append(probas)
        return np.vstack(all_probas)
