from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import myCode.dga2_GAN.dga_model
from myCode.dga2_GAN import dga_model
from myCode.dga2_GAN.dga_reader import load_data, DataReader


flags = tf.flags

# data
flags.DEFINE_string('data_dir',    'dga_data',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('train_dir',   'cv2',     'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('load_model',   'cv/gl_epoch004_25.7601.model',    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
flags.DEFINE_integer('rnn_size',        50,                             'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('char_embed_size', 30,                             'dimensionality of character embeddings')
flags.DEFINE_integer('embed_dimension', 32,                             'embedding features dimensions')
flags.DEFINE_string ('kernels',         str([2] * 20 + [3] * 10),            'CNN kernel widths')
flags.DEFINE_string ('kernel_features', str([32] * 30),                      'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')
flags.DEFINE_integer('random_dimension',         32,                    'dimension of random numbers input in generator')

# optimization
flags.DEFINE_float  ('learning_rate_decay', 0.5,  'learning rate decay')
flags.DEFINE_float  ('learning_rate',       0.001,  'starting learning rate')
flags.DEFINE_float  ('decay_when',          1.0,  'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float  ('param_init',          0.5, 'initialize parameters at')
flags.DEFINE_integer('batch_size',          64,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs',          7,   'number of full passes through the training data')
flags.DEFINE_integer('max_epochs_lr',       5,   'number of epochs of training lr model')
flags.DEFINE_integer('max_epochs_gl',       5,   'number of epochs of training generator model')
flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
flags.DEFINE_integer('max_word_length',     70,   'maximum word length')
flags.DEFINE_integer('iteration',     3,   'number of iterations of lr-training before a gl-lr training')

# bookkeeping
flags.DEFINE_integer('seed',           1021, 'random number generator seed')
flags.DEFINE_integer('print_every',    200,    'how often to print current loss')

FLAGS = flags.FLAGS


def run_test(session, m, data, batch_size, num_steps, reader):
    """Runs the model on the given data."""

    costs = 0.0
    iters = 0
    state = session.run(m.initial_state)

    for step, (x, y) in enumerate(reader.dataset_iterator(data, batch_size, num_steps)):
        cost, state = session.run([m.cost, m.final_state], {
            m.input_data: x,
            m.targets: y,
            m.initial_state: state
        })

        costs += cost
        iters += 1

    return costs / iters


def main(_):
    #　tf.device('/gup:0')
    ''' Trains model from data '''

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)

    char_vocab, char_tensors, char_lens, max_word_length = load_data(FLAGS.data_dir, 70)

    train_reader = DataReader(char_tensors['train'], char_lens['train'], FLAGS.batch_size)

    print('initialized all dataset readers')

    with tf.Graph().as_default(), tf.Session() as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build training graph '''
        # initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
        # initializer = tf.random_uniform_initializer(0.0, 2 * FLAGS.param_init)
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("Model", initializer=initializer):
            train_model = myCode.dga2_GAN.dga_model.inference_graph(
                    char_vocab_size=char_vocab.size,
                    char_embed_size=FLAGS.char_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    num_rnn_layers=FLAGS.rnn_layers,
                    rnn_size=FLAGS.rnn_size,
                    max_word_length=max_word_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    dropout=FLAGS.dropout,
                    embed_dimension=FLAGS.embed_dimension)
            train_model.update(myCode.dga2_GAN.dga_model.decoder_graph(train_model.embed_output,
                                                       char_vocab_size=char_vocab.size,
                                                       batch_size=FLAGS.batch_size,
                                                       num_highway_layers=FLAGS.highway_layers,
                                                       num_rnn_layers=FLAGS.rnn_layers,
                                                       rnn_size=FLAGS.rnn_size,
                                                       max_word_length=max_word_length,
                                                       kernels=eval(FLAGS.kernels),
                                                       kernel_features=eval(FLAGS.kernel_features),
                                                       dropout=FLAGS.dropout,
                                                       ))
            train_model.update(myCode.dga2_GAN.dga_model.en_decoder_loss_graph(train_model.input,
                                                               train_model.input_len_g,
                                                               train_model.decoder_output,
                                                               batch_size=FLAGS.batch_size,
                                                               max_word_length=max_word_length
                                                               ))

            train_model.update(myCode.dga2_GAN.dga_model.genearator_layer(batch_size=FLAGS.batch_size,
                                                          input_dimension=FLAGS.random_dimension,
                                                          max_word_length=max_word_length,
                                                          embed_dimension=FLAGS.embed_dimension))
            train_model.update(myCode.dga2_GAN.dga_model.generator_layer_loss(train_model.gl_output,
                                                              batch_size=FLAGS.batch_size,
                                                              max_word_length=max_word_length,
                                                              embed_dimension=FLAGS.embed_dimension
                                                              ))

            train_model.update(myCode.dga2_GAN.dga_model.lr(train_model.gl_output, FLAGS.batch_size, max_word_length, FLAGS.embed_dimension))
            train_model.update(myCode.dga2_GAN.dga_model.lr_loss(train_model.lr_output, batch_size=FLAGS.batch_size))

            # scaling loss by FLAGS.num_unroll_steps effectively scales gradients by the same factor.
            # we need it to reproduce how the original Torch code optimizes. Without this, our gradients will be
            # much smaller (i.e. 35 times smaller) and to get system to learn we'd have to scale learning rate and max_grad_norm appropriately.
            # Thus, scaling gradients so that this trainer is exactly compatible with the original
            train_model.update(dga_model.autoencoder_train_graph(train_model.en_decoder_loss,
                                                                   FLAGS.learning_rate,
                                                                   FLAGS.max_grad_norm))
            train_model.update(myCode.dga2_GAN.dga_model.lr_train_graph(train_model.lr_loss,
                                                          FLAGS.learning_rate,
                                                          FLAGS.max_grad_norm))
            train_model.update(myCode.dga2_GAN.dga_model.generator_train_graph(train_model.gl_loss,
                                                                 FLAGS.learning_rate,
                                                                 FLAGS.max_grad_norm))

        # create saver before creating more graph nodes, so that we do not save any vars defined below
        saver = tf.train.Saver(max_to_keep=50)

        saver.restore(session, FLAGS.load_model)
        #　print('Loaded model from', FLAGS.load_model, 'saved at global step', train_model.global_step.eval())

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=session.graph)

        ''' take learning rate from CLI, not from saved graph '''
        session.run(
            [tf.assign(train_model.learning_rate, FLAGS.learning_rate),
             tf.assign(train_model.lr_learning_rate, FLAGS.learning_rate),
             tf.assign(train_model.gl_learning_rate, FLAGS.learning_rate)]
        )

        rnn_state_g = session.run([train_model.initial_rnn_state_g])

        np_random = np.random.RandomState(FLAGS.seed)
        ''' training lr here '''
        print("***************train logistic regression********************")
        for epoch in range(FLAGS.max_epochs_lr):
            epoch_start_time = time.time()
            avg_lr_loss = 0.0
            avg_gr_loss = 0.0
            count = 0
            for x, y in train_reader.iter():
                count += 1
                start_time = time.time()

                if (count % FLAGS.iteration) != 0:
                    generator_input = np_random.rand(FLAGS.batch_size, FLAGS.random_dimension)

                    gl_output = session.run([
                        train_model.gl_output,
                    ], {
                        train_model.gl_input: generator_input,
                    })

                    rnn_state_g, _, embed_output = session.run([
                        train_model.final_rnn_state_g,
                        train_model.clear_char_embedding_padding,
                        train_model.embed_output,
                    ], {
                        train_model.input: x,
                        train_model.input_len_g: y,
                        train_model.initial_rnn_state_g: rnn_state_g,
                    })

                    # origin_dga = [char_vocab.change(dga).replace(" ", "") for dga in generated_dga]
                    target = np.zeros([FLAGS.batch_size])
                    # generated_dga[0: int(len(generated_dga) / 2)] = x[0: int(len(generated_dga) / 2)]
                    target[0: int(len(target) / 2)] = np.ones([int(len(target) / 2)])
                    gl_output = gl_output[0]
                    gl_output[0: int(len(embed_output) / 2)] = embed_output[0: int(len(embed_output) / 2)]
                    #
                    # for i in range(int(len(generated_dga) / 2), len(generated_dga)):
                    #     dga_len = 0
                    #     dga = generated_dga[i]
                    #     for dga_char in dga:
                    #         if dga_char == ' ':
                    #             break
                    #         dga_len += 1
                    #     y[i] = dga_len

                    lr_loss_d, _, step_lr = session.run([
                        train_model.lr_loss,
                        train_model.train_op_lr,
                        train_model.global_step_lr,
                    ], {
                        train_model.lr_input: gl_output,
                        train_model.lr_target: target
                    })
                    avg_lr_loss += 0.05 * (lr_loss_d - avg_lr_loss)

                else:
                    generator_input = np_random.rand(FLAGS.batch_size, FLAGS.random_dimension)
                    target = np.zeros([FLAGS.batch_size])
                    rl_loss_g, _,  step_lr = session.run([
                        train_model.lr_loss,
                        train_model.train_op_g,
                        train_model.global_step_lr
                    ], {
                        train_model.gl_input: generator_input,
                        train_model.lr_target: target
                    })

                    avg_gr_loss += 0.05 * (rl_loss_g - avg_gr_loss)

                if count % FLAGS.print_every == 0:
                    time_elapsed = time.time() - start_time
                    print('Regression Logistic: %6d: %d [%5d/%5d], loss_lr/loss_g = %6.8f/%6.7f secs/batch = %.4fs' % (step_lr, epoch, count, train_reader.length, avg_lr_loss, avg_gr_loss, time_elapsed))
                    lr_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="lr_loss", simple_value=avg_lr_loss),
                    ])
                    summary_writer.add_summary(lr_summary, step_lr)

                    gr_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="gr_loss", simple_value=avg_gr_loss),
                    ])
                    summary_writer.add_summary(gr_summary, step_lr)

            print('Epoch training time:', time.time()-epoch_start_time)

            save_as = '%s/lr_epoch%03d_%.4f.model' % (FLAGS.train_dir, epoch, avg_lr_loss)
            saver.save(session, save_as)

        # saver.save(session, "final_model")
        print('Saved model')


if __name__ == "__main__":
    tf.app.run()
