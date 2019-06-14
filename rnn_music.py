import time
import os
import functools
import numpy as np
from utils import *
import tensorflow as tf
tf.enable_eager_execution()


file_path = 'data/irish.abc'
txt = open(file_path).read()
# print(f'Length of text {len(txt)}')
# print(txt[:250])

vocab = sorted(set(txt))
# print(f'Unique characters: {len(vocab)}')

########################
# Process Training Data #
#######################

# Map unique characters to indices
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
text_as_int = np.array([char_to_idx[char] for char in txt])

# Map indices to characters
idx_to_char = np.array(vocab)

# Create Training examples and Targets
seq_length = 100
examples_per_epoch = len(txt) // seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

# Define input and target texts for each sequence


def split_to_sequences(seq):
    """
        Splits to input and target sequnces
    """
    return seq[:-1], seq[1:]


dataset = sequences.map(split_to_sequences)

"""
for input_ex, target_example in dataset.take(1):
    for i, (input_idx, target_idx) in \
            enumerate(zip(input_ex[:5], target_example[:5])):
        print(f'Step {i}')
        print(f' Input {input_idx} {idx_to_char[input_idx]}')
        print(f' expected ouput {target_idx} {idx_to_char[target_idx]}')
"""

# Create_training_batches
batch_size = 64
buffer_size = 10000

steps_per_epoch = examples_per_epoch // batch_size
dataset = dataset.shuffle(buffer_size).batch(
    batch_size, drop_remainder=True)

# Define RNN model
vocab_size = len(vocab)
embedding_dims = 256
rnn_units = 1024

if not tf.test.is_gpu_available():
    LSTM = functools.partial(tf.keras.layers.LSTM,
                             recurrent_activation='sigmoid')
else:
    LSTM = tf.keras.layers.CuDNNLSTM
LSTM = functools.partial(LSTM,
                         return_sequences=True,
                         recurrent_initializer='glorot_uniform',
                         stateful=True)

# Build model


def build_model(vocab_size, embedding_dims, rnn_units, batch_size):
    model = tf.keras.Sequential(
        [tf.keras.layers.Embedding(vocab_size, embedding_dims,
                                   batch_input_shape=[batch_size, None]),
         LSTM(rnn_units),
         tf.keras.layers.Dense(vocab_size)
         ]
    )

    return model


model = build_model(vocab_size, embedding_dims,
                    rnn_units, batch_size=batch_size)
model.summary()


"""
for input_ex_batch, target_ex_batch in dataset.take(1):
    example_batch_predictions = model(input_ex_batch)
    # print(example_batch_predictions.shape, '  ~ ',
    #      batch_size, seq_length, vocab_size)

# Predictions from untrained model
sample_indices = tf.random.categorical(
    example_batch_predictions[0], num_samples=1)
sample_indices = tf.squeeze(sample_indices, axis=-1)

# print('Input: ', ''.join(idx_to_char[input_ex_batch[0]]))
# print('Next char predictions: ', ''.join(idx_to_char[sample_indices]))

"""

#
# Train Model
###

# Loss


def compute_loss(labels, logits):
    return tf.keras.backend.sparse_categorical_crossentropy(
        labels,
        logits, from_logits=True)


# example_batch_loss = compute_loss(target_ex_batch, example_batch_predictions)
# print('Scalar Loss: ', example_batch_loss.numpy().mean())

# train
epochs = 10
history = []

optimizer = tf.train.AdamOptimizer()
checkpoint_dir = '.training_checkpoint'
chpnt_prefix = os.path.join(checkpoint_dir, 'chkpoint_{epoch}')

plotter = Plotter(x_label='iterations', y_label='loss')

for epoch in range(epochs):
    # Initialize hidden state at start of epoch
    hidden = model.reset_states()

    # Enumerate training dataset
    # progress_msg = create_progress_text('Loss : {loss:.2}')
    progress_bar = create_progress_bar()

    for input_, target in progress_bar(dataset):
        with tf.GradientTape() as tape:
            pred = model(input_)
            loss = compute_loss(target, pred)

    # Compute gradients and minimize
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    history.append(loss.numpy().mean())
    # progress_msg.update_mapping(loss=history[-1])
    plotter.plot(history)
    # Update model with new weights
    model.save_weights(chpnt_prefix.format(epoch=epoch))

# Restore checkpoint
model = build_model(vocab_size, embedding_dims,
                    rnn_units, batch_size=batch_size)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape(input_shape=[batch_size, None]))

model.summary()


def generate_text(model, start_string, gen_length=1500):
    """
        Generate test using the trained model
    """
    generated_text = []

    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    model.reset_states()
    bar = create_progress_bar()

    for i in bar(range(gen_length)):
        predictions = model(input_eval)

        # Clear batch dimensions
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.multinomial(
            predictions, num_samples=1)[-1, 0].numpy()

        # Pass predictions and previous hidden states as next inputs
        input_eval = tf.expand_dims([predicted_id], 0)

        # Add predicted character to generated text
        generate_text.append(idx_to_char[predicted_id])

    return start_string + ''.join(generated_text)


text = generate_text(model, start_string='X')
play_songs(text)
print(text)
