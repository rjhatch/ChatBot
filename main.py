import nltk
import random
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

chat_file = open('brucewillis.txt', 'r', errors='ignore')

raw = chat_file.read()

raw = raw.lower()

#nltk.download('punkt')
#nltk.download('wordnet')

greeting_input = ("hey", "morning", "evening", "afternoon", "what's up?", "howdy", "hello")
greeting_response = ["hey", "good day!", "hello there!", "well met", "greetings", "salutations"]

sentence_list = nltk.sent_tokenize(raw)
word_list = nltk.word_tokenize(raw)

lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]


remove_punctuation = dict((ord(punctuation), None) for punctuation in string.punctuation)


def remove_punctuations(text):
    return lemmatize_words(nltk.word_tokenize(text.lower().translate(remove_punctuation)))


def greeting_reply(text):
    for word in text.split():
        if word.lower() in greeting_input:
            return random.choice(greeting_response)


def create_reply(usr_input):
    response = ''
    sentence_list.append(usr_input)
    word_vectors = TfidfVectorizer(tokenizer=remove_punctuations)
    vectorized_words = word_vectors.fit_transform(sentence_list)

    similarity_values = cosine_similarity(vectorized_words[-1], vectorized_words)
    similar_sentence_value = similarity_values.argsort()[0][-2]
    similar_vectors = similarity_values.flatten()
    similar_vectors.sort()

    matched_vector = similar_vectors[-2]

    print(matched_vector)

    if matched_vector == 0:
        return response + "I regret to inform you that I do not understand you."
    else:
        return response + sentence_list[similar_sentence_value]


chatbot_running = True

print("Greetings, fellow entity. \nYou may call me Entity 07 \nSay \"bye\" to end the conversation."
      "\nAsk me a question about Bruce Willis:")

while chatbot_running:
    user_input = input().lower()
    if user_input == 'bye':
        print("Goodbye!")
        chatbot_running = False
        break
    if greeting_reply(user_input) is not None:
        print("Sentient Being: " + greeting_reply(user_input))
    else:
        print("Entity 07: ", end="")
        print(create_reply(user_input))
        sentence_list.remove(user_input)
