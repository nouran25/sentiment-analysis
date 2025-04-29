
def preprocess(review, stoi, reviews_int=None):
    """
    Preprocess the review text by converting it to lowercase,
    splitting it into words, and mapping words to integers using the stoi dictionary.

    Args:
    review (str): The text of the review.
    stoi (dict): A dictionary mapping words to integer values.
    reviews_int (list): A list to store integer representations of the reviews (optional).

    Returns:
    list: A list of integers representing the words in the review.
    """
    # If reviews_int is not provided, initialize it as an empty list
    if reviews_int is None:
        reviews_int = []

    review = review.lower()  # Convert the review to lowercase for consistency
    word_list = review.split()  # Split the review into individual words
    num_list = []  # Initialize an empty list to store the integer representations of words

    for word in word_list:
        if word in stoi.keys():  # Check if the word exists in the stoi dictionary
            num_list.append(stoi[word])  # Append the integer corresponding to the word

    # Append the processed review's integer list to reviews_int
    reviews_int.append(num_list)
    
    return num_list  # Return the list of integers for the current review
