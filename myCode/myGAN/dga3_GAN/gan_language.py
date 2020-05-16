import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import myCode.dga3_GAN.language_helpers
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot

# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!
DATA_DIR = 'AlexaTop1M_NoSeparate'
if len(DATA_DIR) == 0:
    raise Exception("Please specify path to data directory in gan_language.py!")

BATCH_SIZE = 64 # Batch size
# How many iterations to train for, min value is 1000, Please increase the number of iteration in 1000 units
ITERS = 30000
SEQ_LEN = 32 # Sequence length in characters
DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 10 # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 100000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data). default value is 10000000

lib.print_model_settings(locals().copy())

lines, charmap, inv_charmap = myCode.dga3_GAN.language_helpers.load_dataset(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)


def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, len(charmap)])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random.normal(shape)

def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 5, output)
    return inputs + (0.3*output)

def Generator(n_samples, prev_outputs=None):
    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, SEQ_LEN*DIM, output)
    output = tf.reshape(output, [-1, DIM, SEQ_LEN])
    output = ResBlock('Generator.1', output)
    output = ResBlock('Generator.2', output)
    output = ResBlock('Generator.3', output)
    output = ResBlock('Generator.4', output)
    output = ResBlock('Generator.5', output)
    output = lib.ops.conv1d.Conv1D('Generator.Output', DIM, len(charmap), 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output)
    return output
def Discriminator(inputs):
    output = tf.transpose(inputs, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', len(charmap), DIM, 1, output)
    output = ResBlock('Discriminator.1', output)
    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    output = tf.reshape(output, [-1, SEQ_LEN*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', SEQ_LEN*DIM, 1, output)
    return output
real_inputs_discrete = tf.compat.v1.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
fake_inputs = Generator(BATCH_SIZE)
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

disc_real = Discriminator(real_inputs)
disc_fake = Discriminator(fake_inputs)

disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)


# WGAN lipschitz-penalty
alpha = tf.random.uniform(
    shape=[BATCH_SIZE,1,1],
    minval=0.,
    maxval=1.
)

differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty


gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')


gen_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.999).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.999).minimize(disc_cost, var_list=disc_params)

# Dataset iterator
def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]],
                dtype='int32'
            )

# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.

true_char_ngram_lms = [myCode.dga3_GAN.language_helpers.NgramLanguageModel(i+1, lines[10*BATCH_SIZE:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [myCode.dga3_GAN.language_helpers.NgramLanguageModel(i+1, lines[:10*BATCH_SIZE], tokenize=False) for i in range(4)]
for i in range(4):
    print ( "validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])) )
true_char_ngram_lms = [myCode.dga3_GAN.language_helpers.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]

with tf.Session() as session:
    session.run(tf.global_variables_initializer())


    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))

        return decoded_samples


    gen = inf_train_gen()

    sum_time = 0.
    line_time = 0.
    loading_str = "*"
    # 训练3万轮
    for iteration in range(ITERS):
        # 记录当前时间
        start_time = time.time()
        #第一轮
        if (iteration == 0):
            now_time = time.clock()
            print("[Start]")

            # Train generator（第二轮开始训练生成器）
            if iteration > 0:
                _ = session.run(gen_train_op)

            # Train critic
            for i in range(CRITIC_ITERS):
                _data = gen.__next__()
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_inputs_discrete: _data}
                )

                # print("_disc_cost "+str(_disc_cost))
                # print("_ "+str(_))
                # print("_data "+str(_data))
                # print("gen_cost "+str(gen_cost))
                # print("disc_cost"+str(disc_cost))
                # How many iterations to change line
                change_line = int(ITERS / 1000)

                after_time = time.clock() - now_time
                sum_time += after_time
                eta_time = (ITERS - iteration) * (after_time)

                print(
                    "[{1:10}] [Iteration]: {0:10} [Unit iteration time    ]: {2:10.2f} secs [ETA]: {3:10.2f} secs".format(
                        (iteration + 1), loading_str, after_time, eta_time), end="\r")
                now_time = time.clock()
                if iteration % change_line == (change_line - 1):
                    loading_str += "*"
                    if iteration % (10 * change_line) == (10 * change_line - 1):
                        print("{5:5.0f}{0:7} [Iteration]: {1:10} [{2:23}]: {3:10.2f} secs [SUM]: {4:10.2f}".format(
                            "% Done!", (iteration + 1), (str(10 * change_line) + "x iterations time"),
                            (sum_time - line_time), sum_time, (100 * iteration / ITERS)))
                        loading_str = "*"
                        line_time = sum_time

                lib.plot.plot('time', time.time() - start_time)
                lib.plot.plot('train disc cost', _disc_cost)

                if iteration % (10 * change_line) == (10 * change_line - 1):
                    # print("checkpintB"+str(iteration+1))
                    samples = []
                    for i in range(10):
                        samples.extend(generate_samples())

                    for i in range(4):
                        lm = myCode.dga3_GAN.language_helpers.NgramLanguageModel(i + 1, samples, tokenize=False)
                        lib.plot.plot('js{}'.format(i + 1), lm.js_with(true_char_ngram_lms[i]))

                    with open('output_data/samples_{}.txt'.format(str(iteration + 1).zfill(7)), 'w',
                              encoding='utf8') as f:
                        for s in samples:
                            s = "".join(s)
                            s = myCode.dga3_GAN.language_helpers.checkDNSFrom(s)
                            f.write(str(s) + "\n")

                if iteration % (10 * change_line) == (10 * change_line - 1):
                    # print(iteration)
                    lib.plot.flush()
                    lib.plot.tick()

