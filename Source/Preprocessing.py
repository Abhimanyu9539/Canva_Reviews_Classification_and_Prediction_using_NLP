from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer

sw = stopwords.words('english')
tokenizer  = RegexpTokenizer(r'\w+')

def process_text(review, stem='p'):
    """
    Given a text, the function convertes the text into lower case
    removes stop words, removes punctuations, tokenizes the text
    performs stemming and return the processed text

    :param review: raw text
    :param stem: Stemmer :'p' for PoerterSTemmer and
    'l' for lancaster stemmer

    :return: processed text
    """

    # Convert the text to lower case
    review = review.lower()

    # Word tokenize the review
    tokens = word_tokenize(review)

    # Remove stopwords
    tokens = [t for  t in tokens if t not in sw]

    # Remove punctuations
    tokens = [tokenizer.tokenize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 0]
    tokens =["".join(t) for t in tokens]

    # Create stemmer
    if stem =='p':
        stemmer = PorterStemmer()
    elif stem == 'l':
        stemmer = LancasterStemmer()
    else:
        raise Exception("stem has to be either 'p' for Porter or 'l' for Lancaster")

    # Stemming
    tokens = [stemmer.stem(t) for t in tokens]

    # Return clean string
    return " ".join(tokens)