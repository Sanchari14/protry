import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

def get_synonyms(word, pos=None):
    """
    Gets synonyms for a given word, optionally filtered by part of speech.

    Args:
        word (str): The word to find synonyms for.
        pos (str, optional):  Part of speech  to filter synonyms.
            Valid values are 'n', 'v', 'a', 'r', 's'  (noun, verb, adjective, adverb, satellite adjective).
            Defaults to None (no filtering).

    Returns:
        list: A list of synonyms.
    """
    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name != word.lower():  # Exclude the original word
                synonyms.add(name)
    return list(synonyms)


def filter_synonyms(synonyms, context_words):
    """
    Filters a list of synonyms based on their relevance to a given context.

    Args:
        synonyms (list): A list of synonyms.
        context_words (list): A list of words defining the context.

    Returns:
        list: A filtered list of synonyms.
    """
    filtered_synonyms = []
    for syn in synonyms:
        # Check if any context word is present in the synonym
        if any(context_word in syn for context_word in context_words):
            filtered_synonyms.append(syn)
    return filtered_synonyms



initial_keywords_stressed = ["stressed", "anxious", "worried", "overwhelmed"]
context_words = ["feel", "stress", "anxiety", "worry", "overwhelm", "pressure", "tense", "frustrated", "sad", "depressed"] # Added more context words
all_stressed_keywords = set()

for keyword in initial_keywords_stressed:
    syns = get_synonyms(keyword)
    filtered_syns = filter_synonyms(syns, context_words) # Filter the synonyms
    all_stressed_keywords.update([keyword.lower()])  # Include the original keyword
    all_stressed_keywords.update(filtered_syns)

# Add synonyms of base words
for word in initial_keywords_stressed:
    syns = get_synonyms(word)
    all_stressed_keywords.update(syns)

# Output for intents.txt
for keyword in sorted(all_stressed_keywords):
    print(f"{keyword},feeling_stressed")
