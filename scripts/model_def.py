
import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.

        Parameters:
        - vocab_size: Size of the vocabulary (number of unique words) used for embedding.
        - output_size: The number of classes we want to predict (e.g., 1 for binary classification).
        - embedding_dim: Dimensionality of the embedding layer (how many features to represent each word).
        - hidden_dim: Number of features in the hidden state of the LSTM (how much information it can hold).
        - n_layers: The number of stacked LSTM layers (how deep the network is).
        - drop_prob: Probability of dropping out units in the dropout layer to prevent overfitting.
        """
        super().__init__()  # Call the initializer of the parent class (nn.Module) to ensure that everything is set up properly.

        self.output_size = output_size  # Store output size
        self.n_layers = n_layers          # Store number of LSTM layers
        self.hidden_dim = hidden_dim      # Store hidden dimension size

        # Embedding layer to convert word indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer with specified parameters
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # Dropout layer to help prevent overfitting
        self.dropout = nn.Dropout(0.3)

        # Fully connected layer to produce output from LSTM outputs
        self.fc = nn.Linear(hidden_dim, output_size)

        # Sigmoid layer for output activation (to squash values between 0 and 1)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.

        Parameters:
        - x: Input batch of word indices.
        - hidden: The hidden state of the LSTM from the previous time step.

        Returns:
        - sig_out: The output probabilities for each input in the batch.
        - hidden: The hidden state for the next time step.
        """
        batch_size = x.size(0)  # Get the batch size from the input

        # Get the embeddings for the input word indices
        embeds = self.embedding(x)
        # Pass the embeddings through the LSTM layer, which processes the sequence of word embeddings and updates the hidden state.
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Reshape lstm_out for the fully connected layer
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # Apply dropout for regularization
        out = self.dropout(lstm_out)
        # Pass the output through the fully connected layer
        out = self.fc(out)
        # Apply the sigmoid activation function
        sig_out = self.sig(out)

        # Reshape the output to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # Get the last output from the sequence

        # Return the last sigmoid output and the hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data  # Get the weight of the model to determine the device (CPU/GPU)

        if (train_on_gpu):  # Check if training on GPU
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),  # Initialize hidden state on GPU
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())  # Initialize cell state on GPU
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),  # Initialize hidden state on CPU
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())  # Initialize cell state on CPU

        return hidden  # Return the initialized hidden state, which will be used for processing the input sequences.
