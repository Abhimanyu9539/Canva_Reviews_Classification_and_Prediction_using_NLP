from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def vectorize(tokens_list, y, vect='bow', min_df=5, ng_low=1, ng_high=3,
              test_size=0.2,  rs=42):
    """
        Given list of tokens and the dependent variables, the function will
        vectorize the tokens, split the set into train and test and returns the
        splits along with the vectorizer
        :param token_list: List of processed tokens
        :param y: Dependent variable
        :param vect: Vectorizer ('bow' for count vectors, 'bowb' for binary count
                            vectors, 'ng' for n-grams and 'tf' for tf-idf
        :param min_df: min_df parameter in CountVectorizer
        :param ng_low: Lower value for n-gram
        :param ng_high: Higher value for n-gram
        :param test_size: Size of test split
        :param rs: random seed
        :return: train and test vectors (both X and y), vectorizer
        """

    # Create vectorizer
    if vect == 'bow':
        vectorizer = CountVectorizer(min_df=min_df)
    elif vect == 'bowb':
        vectorizer = CountVectorizer(binary=True, min_df=min_df)
    elif vect == 'ng':
        vectorizer = CountVectorizer(min_df=min_df, ngram_range=(ng_low,ng_high))
    elif vect == 'tf':
        vectorizer = TfidfVectorizer(min_df=min_df)
    else:
        raise Exception("Vect has to be one of 'bow', 'bowb', 'ng', 'tf'")

    # Fit the vectorizer and transform the input data
    X = vectorizer.fit_transform(tokens_list)

    # Split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=test_size,
                                            stratify=y, random_state=rs)
    return  x_train, x_test, y_train, y_test, vectorizer