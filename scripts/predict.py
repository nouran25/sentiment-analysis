
def predict(net, test_review, sequence_length=200):
    ''' Prints out whether a give review is predicted to be
        positive or negative in sentiment, using a trained model.

        params:
        net - A trained net
        test_review - a review made of normal text and punctuation
        sequence_length - the padded length of a review
        '''
    #change the reviews to sequence of integers
    int_rev = preprocess(test_review, stoi)
    #pad the reviews as per the sequence length of the feature
    features = pad_features(int_rev, seq_length=sequence_length)

    #changing the features to PyTorch tensor
    features = torch.from_numpy(features)

    #pass the features to the model to get prediction
    net.eval()
    # Get the batch size from features shape
    batch_size = features.shape[0]
    val_h = net.init_hidden(batch_size) # Initialize hidden state with correct batch size

    val_h = tuple([each.data for each in val_h])

    if(train_on_gpu):
        features = features.cuda()

    output, val_h = net(features, val_h)

    # Get the prediction for the entire review by taking the mean prediction
    pred = torch.round(torch.mean(output))

    #mapping the numeric values to postive or negative
    output = "Positive" if pred.item() == 1 else "Negative"

    # print custom response based on whether test_review is pos/neg
    return output
