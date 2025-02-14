from collections import ChainMap
from typing import Callable, Dict, Set

import pandas as pd


class FeatureMap:
    name: str

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        pass

    @classmethod
    def prefix_with_name(self, d: Dict) -> Dict[str, float]:
        """just a handy shared util function"""
        return {f"{self.name}/{k}": v for k, v in d.items()}


class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        # Tokenize, lowercase, remove stop words, and keep only unique words
        unique_words = set(word.lower() for word in text.split() if word.lower() not in cls.STOP_WORDS)
        
        # Create features dictionary with count 1.0 for each unique word
        features = {word: 1.0 for word in unique_words}

        # Update global word frequency
        #for word in unique_words:
        #    cls.word_frequency[word] = cls.word_frequency.get(word, 0) + 1
        
        return cls.prefix_with_name(features)


class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        if len(text.split()) < 10:
            k = "short"
            v = 1.0
        else:
            k = "long"
            v = 5.0
        ret = {k: v}
        return self.prefix_with_name(ret)

# Below are some additional features 

class LexicalComplexity(FeatureMap):
    name = "lex_complexity"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        """
        Computes lexical complexity features:
          - Average word length.
          - Type-token ratio (unique words / total words). ie. vocab diversity. 
        
        Args:
            text (str): Input text.
        
        Returns:
            Dict[str, float]: A dictionary of features with keys prefixed by 'lex_complexity/'.
        """
        # Tokenize the text by splitting on whitespace.
        words = [word for word in text.split() if word]
        
        # Avoid division by zero if there are no words.
        if not words:
            avg_word_length = 0.0
            type_token_ratio = 0.0
        else:
            # Calculate average word length.
            avg_word_length = sum(len(word) for word in words) / len(words)
            # Calculate type-token ratio.
            # Lowercase words to treat "The" and "the" as the same token.
            unique_words = set(word.lower() for word in words)
            type_token_ratio = len(unique_words) / len(words)
        
        # Build a feature dictionary.
        features = {
            "avg_word_length": avg_word_length,
            "type_token_ratio": type_token_ratio
        }
        # Prefix each feature key with the feature set name.
        return cls.prefix_with_name(features)

class PunctuationFeatures(FeatureMap):
    name = "punc"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        """
        Computes punctuation features:
            - Count of exclamation marks.
            - Count of question marks.
        You can extend this by counting other punctuation marks if desired.
        """
        # Count key punctuation marks in the raw text.
        features = {
            "exclamation": text.count("!"),
            "question": text.count("?"),
        }
        return cls.prefix_with_name(features)

class SentimentLexiconFeatures(FeatureMap):
    name = "sentilex"

    # Extended lexicon for SST2:
    LEXICON = {
        # Strong Positive
        "excellent": 2.0,
        "amazing": 2.0,
        "wonderful": 2.0,
        "fantastic": 2.0,
        "incredible": 2.0,
        "spectacular": 2.0,
        "brilliant": 2.0,
        "masterful": 2.0,
        "magnificent": 2.0,
        "stunning": 2.0,
        "riveting": 2.0,
        "delightful": 1.5,
        "great": 1.5,
        "good": 1.0,
        "enjoyable": 1.5,
        "fun": 1.5,
        "exciting": 1.5,
        "thrilling": 2.0,
        "heartwarming": 1.5,
        "inspiring": 1.5,
        "hilarious": 1.5,
        
        # Moderate Positive
        "pleasant": 1.0,
        "nice": 1.0,
        "positive": 1.0,
        
        # Moderate Negative
        "poor": -1.0,
        "bad": -1.0,
        "mediocre": -1.0,
        "lame": -1.5,
        "disappointing": -1.5,
        "uninteresting": -1.0,
        "forgettable": -1.0,
        
        # Strong Negative
        "terrible": -2.0,
        "awful": -2.0,
        "horrible": -2.0,
        "dreadful": -2.0,
        "miserable": -2.0,
        "depressing": -1.5,
        "sorrowful": -1.5,
        "tragic": -2.0,
        "unbearable": -2.0,
        "waste": -1.5,
    }


    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        """
        Extract sentiment-based features from the input text using a predefined sentiment lexicon.

        This method tokenizes the input text by converting it to lowercase and splitting it on whitespace.
        For each token that is present in the lexicon, its corresponding sentiment score is retrieved and added
        to a cumulative sentiment score. It also counts the number of sentiment-bearing tokens. The method then
        computes:
          - A cumulative sentiment score (the sum of sentiment scores of all matching tokens),
          - An average sentiment score per SENTIMENT-BEARING token
          - individual features for each sentiment-bearing word with their associated scores.
        
        The resulting features are then prefixed with the feature set name ("sentilex/") using the
        `prefix_with_name` method from the base class.

        Parameters:
            text (str): The input text from which to extract sentiment features.

        Returns:
            Dict[str, float]: A dictionary containing the following keys:
                - 'sentilex/sentiment_score': The cumulative sentiment score for the text.
                - 'sentilex/sentiment_avg': The average sentiment score per sentiment-bearing token (or 0.0 if no such tokens exist).
                - 'sentilex/word_TOKEN': additional keys for each sentiment-bearing token (e.g., 'sentilex/word_excellent') with their respective sentiment scores.
        """
        tokens = text.lower().split()  # or use a more advanced tokenizer
        score = 0.0
        count = 0
        # add an individual feature for each high-signal word
        features = {}
        for token in tokens:
            if token in cls.LEXICON:
                score += cls.LEXICON[token]
                count += 1
                # add an individual feature for this token:
                features[f"word_{token}"] = cls.LEXICON[token]
        # add a cumulative sentiment score feature
        features["sentiment_score"] = score
        # add average sentiment per sentiment-bearing token if count > 0.
        features["sentiment_avg"] = score / count if count > 0 else 0.0
        return cls.prefix_with_name(features)

class GroupDiscriminativeFeatures(FeatureMap):
    name = "discgroup"
    
    # Define a lexicon for each newsgroup.
    # Each key is a newsgroup name and its value is a dictionary mapping discriminative words to weights.
    LEXICONS = {
        "comp.graphics": {
            "graphics": 1.5,
            "render": 1.5,
            "image": 1.5,
            "vector": 1.5
        },
        "comp.os.ms-windows.misc": {
            "windows": 1.5,
            "microsoft": 1.5,
            "os": 1.5,
            "driver": 1.5
        },
        "comp.sys.ibm.pc.hardware": {
            "ibm": 1.5,
            "pc": 1.5,
            "hardware": 1.5,
            "motherboard": 1.5
        },
        "comp.sys.mac.hardware": {
            "mac": 1.5,
            "apple": 1.5,
            "macintosh": 1.5
        },
        "comp.windows.x": {
            "x": 1.5,
            "xorg": 1.5,
            "unix": 1.5
        },
        "rec.autos": {
            "car": 1.5,
            "automobile": 1.5,
            "engine": 1.5
        },
        "rec.motorcycles": {
            "motorcycle": 1.5,
            "bike": 1.5,
            "rider": 1.5
        },
        "rec.sport.baseball": {
            "baseball": 1.0,
            "pitcher": 1.5,
            "hitter": 1.5,
            "slugger": 1.5
        },
        "rec.sport.hockey": {
            "hockey": 1.5,
            "puck": 1.5,
            "ice": 1.5,
            "rink": 1.5
        },
        "sci.crypt": {
            "cryptography": 1.5,
            "encryption": 1.5,
            "decryption": 1.5
        },
        "sci.electronics": {
            "electronics": 1.5,
            "circuit": 1.5,
            "resistor": 1.5
        },
        "sci.med": {
            "medical": 1.0,
            "doctor": 1.5,
            "treatment": 1.5,
            "disease": 1.5
        },
        "sci.space": {
            "space": 1.0,
            "nasa": 1.5,
            "orbit": 1.5,
            "astronomy": 1.5
        },
        "misc.forsale": {
            "forsale": 1.5,
            "buy": 1.5,
            "sell": 1.5,
            "price": 1.5
        },
        "talk.politics.misc": {
            "politics": 1.5,
            "government": 1.5,
            "policy": 1.5
        },
        "talk.politics.guns": {
            "gun": 1.5,
            "weapon": 1.5,
            "firearm": 1.5
        },
        "talk.politics.mideast": {
            "israel": 1.5,
            "palestine": 1.5,
            "mideast": 1.5
        },
        "talk.religion.misc": {
            "religion": 1.5,
            "faith": 1.5,
            "spiritual": 1.5
        },
        "alt.atheism": {
            "atheism": 1.5,
            "god": 1.5  
        },
        "soc.religion.christian": {
            "christian": 1.5,
            "church": 1.5,
            "bible": 1.5
        },
    }
    
    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        """
        Extract discriminative word features for each newsgroup from the input text using group-specific lexicons.
        
        The method converts the input text to lowercase and splits it on whitespace to obtain tokens.
        It then iterates over each newsgroup-specific lexicon defined in LEXICONS. For each group,
        if a token is present in that group's lexicon, a feature is added with a key of the form
        'discgroup/<group>/word_<token>' and the corresponding weight is assigned.
        
        This design allows the model to receive signals that a particular word is strongly indicative
        of a specific newsgroup, even if the lexicon for that group contains only a few words.
        
        Parameters:
            text (str): The input text (document) from which to extract discriminative features.
        
        Returns:
            Dict[str, float]: A dictionary of features with keys prefixed by "discgroup/".
            For example, if the token "computer" appears in the text and is present in the lexicon for
            "comp.graphics", the feature 'discgroup/comp.graphics/word_computer' will be included with its
            associated weight.
        """
        tokens = text.lower().split()  # Basic tokenization; refine as needed.
        features = {}
        for group, lexicon in cls.LEXICONS.items():
            for token in tokens:
                if token in lexicon:
                    features[f"word_{group}_{token}"] = lexicon[token]
        return cls.prefix_with_name(features)



FEATURE_CLASSES_MAP = {c.name: c for c in [BagOfWords, SentenceLength, LexicalComplexity, PunctuationFeatures, SentimentLexiconFeatures, GroupDiscriminativeFeatures]}


def make_featurize(
    feature_types: Set[str],
) -> Callable[[str], Dict[str, float]]:
    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in feature_types]

    def _featurize(text: str):
        f = ChainMap(*[fn(text) for fn in featurize_fns])
        return dict(f)

    return _featurize


__all__ = ["make_featurize"]

if __name__ == "__main__":
    text = "I love this movie"
    print(text)
    print(BagOfWords.featurize(text))
    print(LexicalComplexity.featurize(text))
    print(PunctuationFeatures.featurize(text))
    print(SentimentLexiconFeatures.featurize(text))
    print(GroupDiscriminativeFeatures.featurize(text))
    featurize = make_featurize({"bow", "len", "lex_complexity", "punc", "sentilex", "discgroup"})
    print(featurize(text))
