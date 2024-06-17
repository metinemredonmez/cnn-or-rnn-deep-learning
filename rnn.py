import tensorflow as tf
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Sample text data
text = "Hello world! This is a sample text for RNN."

# Preprocess the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1

# Create input-output pairs for training
seq_length = 5
X, y = [], []
for i in range(len(sequences) - seq_length):
    X.append(sequences[i:i + seq_length])
    y.append(sequences[i + seq_length])

X = tf.keras.preprocessing.sequence.pad_sequences(X)
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Build the RNN model
model = tf.keras.Sequential([
    Embedding(vocab_size, 10, input_length=seq_length),
    SimpleRNN(50, return_sequences=False),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100)
