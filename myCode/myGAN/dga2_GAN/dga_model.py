from __future__ import print_function
from __future__ import division

import tensorflow as tf


class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
        (属性字典-一种方便的数据结构，类似于python 3.3中的SimpleNamespace可以使用属性来读/写字典内容。)
    '''

    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='SAME') + b


def linear(input_, output_size, scope=None, activation=tf.nn.leaky_relu):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''
    # shape = input_.get_shape().as_list()
    # if len(shape) != 2:
    #     raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    # if not shape[1]:
    #     raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    # input_size = shape[1]
    #
    # # Now the computation.
    # with tf.variable_scope(scope or "SimpleLinear"):
    #     matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
    #     bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
    #
    # return tf.matmul(input_, tf.transpose(matrix)) + bias_term
    return tf.layers.dense(input_, output_size, activation=activation)


def highway(input_, size, num_layers=1,scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    # return linear(input_, size)

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = linear(input_, size, scope='highway_lin_%d' % idx)
            t = linear(input_, size, scope='highway_gate_%d' % idx, activation=tf.nn.sigmoid)
            output = t * g + (1. - t) * input_
            input_ = output
    return output


def tdnn(input_, kernels, kernel_features, scope='TDNN'):
    '''

    :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)
    '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    # max_word_length = input_.get_shape()[1]

    # input_: [batch_size*num_unroll_steps, 1, max_word_length, embed_size]

    layers = []
    with tf.variable_scope(scope):
        kernel_index = 0
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            conv = tf.layers.conv1d(input_, kernel_feature_size, kernel_size, 1, padding='same')
            conv = tf.transpose(conv, [0, 2, 1])
            pool = tf.layers.max_pooling1d(conv, [conv.get_shape()[1]], 1, padding='valid')
            pool2 = tf.transpose(pool, [0, 2, 1])
            layers.append(pool2)

            kernel_index += 1

        if len(kernels) > 1:
            output = tf.concat(layers, 2)
        else:
            output = layers[0]

    return output


# the sum of kernel_features should be a multiple of max_word_length
def inference_graph(char_vocab_size,
                    char_embed_size=20,
                    batch_size=20,
                    num_highway_layers=2,
                    num_rnn_layers=2,
                    rnn_size=50,
                    max_word_length=65,
                    kernels=[2] * 20 + [3] * 10,
                    kernel_features=[32] * 30,
                    dropout=0.0,
                    embed_dimension=32):
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    with tf.variable_scope('Encoder'):

        input_ = tf.placeholder(tf.int32, shape=[batch_size, max_word_length], name="input")
        input_len = tf.placeholder(tf.int32, shape=[batch_size], name="input")

        ''' First, embed characters '''
        with tf.variable_scope('Embedding'):
            char_embedding = tf.get_variable('char_embedding', [char_vocab_size, char_embed_size])

            ''' this op clears embedding vector of first symbol (symbol at position 0, which is by convention the position
            of the padding symbol). It can be used to mimic Torch7 embedding operator that keeps padding mapped to
            zero embedding vector and ignores gradient updates. For that do the following in TF:
            1. after parameter initialization, apply this op to zero out padding embedding vector
            2. after each gradient update, apply this op to keep padding at zero'''
            clear_char_embedding_padding = tf.scatter_update(char_embedding, [0],
                                                             tf.constant(0.0, shape=[1, char_embed_size]))

            # [batch_size x max_word_length, num_unroll_steps, char_embed_size]
            input_embedded = tf.nn.embedding_lookup(char_embedding, input_)

        ''' Second, apply convolutions '''
        input_cnn = tdnn(input_embedded, kernels, kernel_features)

        input_cnn = tf.layers.batch_normalization(input_cnn)

        ''' Maybe apply Highway '''
        if num_highway_layers > 0:
            input_cnn = tf.reshape(input_cnn, [batch_size * max_word_length, -1])
            input_cnn = highway(input_cnn, input_cnn.get_shape()[-1], num_layers=num_highway_layers)
            input_cnn = tf.reshape(input_cnn, [batch_size, max_word_length, -1])
            input_cnn = tf.layers.batch_normalization(input_cnn)

        ''' Finally, do LSTM '''
        with tf.variable_scope('LSTM'):
            def create_rnn_cell():
                cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
                if dropout > 0.0:
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1. - dropout)
                return cell

            if num_rnn_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(num_rnn_layers)], state_is_tuple=True)
            else:
                cell = create_rnn_cell()

            initial_rnn_state = cell.zero_state(batch_size, dtype=tf.float32)

            input_cnn = tf.reshape(input_cnn, [batch_size, max_word_length, -1])
            # input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(input_cnn, max_word_length, 1)]

            outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, input_cnn, sequence_length=input_len,
                                                         initial_state=initial_rnn_state, dtype=tf.float32)

            outputs = tf.layers.batch_normalization(outputs)

            # outputs = tf.concat([tf.expand_dims(output, 1) for output in outputs], 1)
            outputs = tf.reshape(outputs, [batch_size * max_word_length, -1])
            embed_output = linear(outputs, embed_dimension, scope='out_linear')
            embed_output = tf.reshape(embed_output, [batch_size, max_word_length, embed_dimension])

    return adict(
        input=input_,
        input_len_g=input_len,
        clear_char_embedding_padding=clear_char_embedding_padding,
        input_embedded=input_embedded,
        input_cnn=input_cnn,
        initial_rnn_state_g=initial_rnn_state,
        final_rnn_state_g=final_rnn_state,
        rnn_outputs=outputs,
        embed_output=embed_output
    )


def decoder_graph(_input,
                  char_vocab_size,
                  batch_size=20,
                  num_highway_layers=2,
                  num_rnn_layers=2,
                  rnn_size=50,
                  max_word_length=65,
                  kernels=[2] * 20 + [3] * 10,
                  kernel_features=[32] * 30,
                  dropout=0.0,
                  ):

    # rnn_input = [tf.squeeze(x, [1]) for x in tf.split(_input, max_word_length, 1)]
    # input_len = tf.placeholder(tf.int32, shape=[batch_size], name="input")
    _input = tf.layers.batch_normalization(_input)

    rnn_input = [tf.squeeze(x, [1]) for x in tf.split(_input, max_word_length, 1)]

    with tf.variable_scope('Decoder'):
        def create_rnn_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
            if dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1. - dropout)
            return cell

        if num_rnn_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(num_rnn_layers)], state_is_tuple=True)
        else:
            cell = create_rnn_cell()

        initial_rnn_state = cell.zero_state(batch_size, dtype=tf.float32)


        # outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, _input, sequence_length=input_len,
        #                                              initial_state=initial_rnn_state, dtype=tf.float32)
        outputs, final_rnn_state = tf.nn.static_rnn(cell, rnn_input, initial_state=initial_rnn_state, dtype=tf.float32)


        # outputs = [tf.expand_dims(time_unit, 1) for time_unit in outputs]
        outputs = tf.concat([tf.expand_dims(output, 1) for output in outputs], 1)

        outputs = tf.layers.batch_normalization(outputs)
        '''
            highway network
        '''
        if num_highway_layers > 0:
            # rnn_outputs = tf.concat(outputs, 1)
            rnn_outputs = tf.reshape(outputs, [batch_size * max_word_length, -1])
            highway_outputs = highway(rnn_outputs, rnn_outputs.get_shape()[-1], num_layers=num_highway_layers)
            outputs = tf.reshape(highway_outputs, [batch_size, max_word_length, -1])
            outputs = tf.layers.batch_normalization(outputs)

        '''
            cnn network
        '''
        cnn_outputs = tdnn(outputs, kernels, kernel_features)
        # cnn_outputs = tf.layers.batch_normalization(cnn_outputs)

        cnn_outputs = tf.reshape(cnn_outputs, [batch_size * max_word_length, -1])
        embed_out = linear(cnn_outputs, char_vocab_size, scope='out_linear')
        embed_out = tf.reshape(embed_out, [batch_size, max_word_length, -1])

        generated_dga = tf.multinomial(tf.reshape(embed_out, [batch_size * max_word_length, -1]), 1)
        generated_dga = tf.reshape(tf.squeeze(generated_dga), [batch_size, max_word_length])

    return adict(
        decoder_input=_input,
        # input_len_d=input_len,
        decoder_output=embed_out,
        initial_rnn_state_d=initial_rnn_state,
        final_rnn_state_d=final_rnn_state,
        generated_dga=generated_dga
    )


def en_decoder_loss_graph(input_, input_len, embed_out, batch_size=20, max_word_length=65):
    with tf.variable_scope('Loss'):
        input_ = tf.reshape(input_, [batch_size * max_word_length])
        mask = tf.sequence_mask(input_len, max_word_length)
        mask2 = tf.logical_not(mask)
        embed_out = tf.reshape(embed_out, [batch_size * max_word_length, -1])
        loss1 = tf.reduce_mean(tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=embed_out,
                                                                                             labels=input_),
                                              tf.reshape(mask, [-1])), name='loss')
        loss2 = tf.reduce_mean(tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=embed_out,
                                                                                             labels=input_),
                                              tf.reshape(mask2, [-1])), name='loss')

        # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=embed_out, labels=input_), name='loss')
        # loss = loss1 + 0.05 * loss2

        loss = tf.where(tf.greater(loss2, loss1), loss1 + 10 * loss2, loss1 + 0.05 * loss2)
    return adict(
        en_decoder_loss=loss,
        mask1=mask,
        mask2=mask2,
        loss1=loss1,
        loss2=loss2
    )


def autoencoder_train_graph(loss, learning_rate=1.0, max_grad_norm=0.1):
    ''' Builds training graph. '''
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.variable_scope('RMSProp_aed'):
        # SGD learning parameter
        learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')

        # collect all trainable variables
        tvars = [x for x in tf.trainable_variables() if "Model/Encoder" in x.name or "Model/Decoder" in x.name]
        # grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
        # grads = tf.gradients(loss, tvars)
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.01)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return adict(
        learning_rate=learning_rate,
        global_step_autoencoder=global_step,
        # global_norm=max_grad_norm,
        train_op=train_op)


def lr(input_, batch_size=20, max_word_length=65, embed_dimension=32):
    with tf.variable_scope('LR'):
        # input_ = tf.placeholder(tf.float32, shape=[batch_size, max_word_length, embed_dimension], name="input")
        input_re = tf.reshape(input_, [batch_size, -1])
        output = linear(input_re, 2, scope='lr_linear')
    return adict(
        lr_input=input_,
        lr_output=output
    )


def lr_loss(_input, batch_size=20):
    with tf.variable_scope('LR_loss'):
        target = tf.placeholder(tf.int32, shape=[batch_size], name="target")
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_input, labels=target),
                              name='loss')
    return adict(
        lr_target=target,
        lr_loss=loss
    )


def lr_train_graph(loss, learning_rate=0.01, max_grad_norm=5.0):
    ''' Builds training graph. '''
    global_step = tf.Variable(0, name='global_step_lr', trainable=False)

    with tf.variable_scope('Adam_lr'):
        # SGD learning parameter
        learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
        learning_rate_g = tf.Variable(learning_rate, trainable=False, name='learning_rate_g')

        # collect all trainable variables
        tvars = [x for x in tf.trainable_variables() if "Model/LR" in x.name]
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)

        optimizer_a = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer_a.apply_gradients(zip(grads, tvars), global_step=global_step)

        gvars = [x for x in tf.trainable_variables() if "Model/GL" in x.name]
        grads_g, global_norm_g = tf.clip_by_global_norm(tf.gradients(-loss, gvars), max_grad_norm)

        optimizer_ga = tf.train.AdamOptimizer(learning_rate_g)
        train_op_g = optimizer_ga.apply_gradients(zip(grads_g, gvars), global_step=global_step)


    return adict(
        lr_learning_rate=learning_rate,
        lr_learning_rate_g=learning_rate_g,
        global_step_lr=global_step,
        global_norm_lr=global_norm,
        train_op_lr=train_op,
        train_op_g=train_op_g,
    )


def genearator_layer(batch_size=20, input_dimension=32, max_word_length=65, embed_dimension=32):
    with tf.variable_scope('GL'):
        input_ = tf.placeholder(tf.float32, shape=[batch_size, input_dimension], name="input")
        output = linear(input_, max_word_length * embed_dimension, scope='lr_linear')
        output = tf.reshape(output, [batch_size, max_word_length, embed_dimension])
    return adict(
        gl_input=input_,
        gl_output=output
    )


def generator_layer_loss(_input, batch_size=20, max_word_length=65, embed_dimension=32):
    with tf.variable_scope('gl_loss'):
        target = tf.placeholder(tf.float32, shape=[batch_size, max_word_length, embed_dimension], name="target")
        loss = tf.reduce_sum(tf.losses.mean_squared_error(_input, target))
    return adict(
        gl_target=target,
        gl_loss=loss
    )


def generator_train_graph(loss, learning_rate=0.01, max_grad_norm=5.0):
    ''' Builds training graph. '''
    global_step = tf.Variable(0, name='global_step_gl', trainable=False)

    with tf.variable_scope('Adam_gl'):
        # SGD learning parameter
        learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')

        # collect all trainable variables
        tvars = [x for x in tf.trainable_variables() if "Model/GL" in x.name]
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)

        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return adict(
        gl_learning_rate=learning_rate,
        global_step_gl=global_step,
        global_norm_gl=global_norm,
        train_op_gl=train_op)


def model_size():
    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        size += sz
    return size


if __name__ == '__main__':
    with tf.Session() as sess:
        with tf.variable_scope('Model'):
            graph = inference_graph(char_vocab_size=51, dropout=0.5)
            graph.update(decoder_graph(graph.embed_output, graph.input_len_g, char_vocab_size=51))
            graph.update(en_decoder_loss_graph(graph.input, graph.input_len_g, graph.decoder_output))

            graph.update(lr())
            graph.update(lr_loss(graph.lr_output))
            graph.update(genearator_layer())
            graph.update(generator_layer_loss(graph.gl_output))

            graph.update(autoencoder_train_graph(graph.en_decoder_loss))


        print('Model size is:', model_size())

