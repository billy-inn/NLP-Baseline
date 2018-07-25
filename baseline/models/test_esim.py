from utils import embedding_utils, io_utils, data_utils
import config
import numpy as np
import tensorflow as tf
from models.esim import ESIM

def create_session():
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=8,
        allow_soft_placement=True,
        log_device_placement=False)
    return tf.Session(config=session_conf)

def test_esim():
    sent1, sent2, labels = io_utils.read_snli(config.SNLI_VALID)

    embedding = embedding_utils.Embedding(
        config.EMBEDDING_DATA,
        sent1 + sent2,
        config.SNLI_MAX_LENGTH)

    label_dict = data_utils.create_label_dict(labels)

    sent1_len = np.array(list(map(embedding.len_transform, sent1)))
    sent2_len = np.array(list(map(embedding.len_transform, sent2)))
    sent1 = np.array(list(map(embedding.text_transform, sent1)))
    sent2 = np.array(list(map(embedding.text_transform, sent2)))
    labels = np.array(list(map(lambda x: label_dict[x], labels)))

    train_set = list(zip(sent1, sent2, sent1_len, sent2_len, labels))
    valid_set = list(zip(sent1, sent2, sent1_len, sent2_len))
    valid = (valid_set, labels)

    kwargs = {
        "num_classes": len(label_dict),
        "vocab_size": embedding.vocab_size,
        "embedding_size": embedding.embedding_dim,
        "seq_len": config.SNLI_MAX_LENGTH,
        "word_embeddings": embedding.embedding,
        "hparams": {
            "num_units": 30,
            "input_dropout": 0.9,
            "output_dropout": 0.9,
            "state_dropout": 0.9,
            "hidden_layers": 1,
            "hidden_units": 50,
            "hidden_dropout": 0.9,
            "lr": 0.001,
            "l2_reg_lambda": 0.0001,
            "batch_size": 256,
            "num_epochs": 20,
        }
    }

    model = ESIM(**kwargs)
    sess = create_session()
    model.init(sess)
    model.fit(sess, train_set, valid)
