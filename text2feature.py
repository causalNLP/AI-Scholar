import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from nltk import word_tokenize
import json
from tqdm import tqdm

class TextFeatures:
    def __init__(self):
        pass

    def text2emb(self):
        pass


class PersonalInfo(TextFeatures):
    def __init__(self, domain):
        super().__init__()
        self.domain = domain  # one of PublicVars.domains, i.e., ['AI', 'ML', 'NLP', 'CV']

    def name2features(self):
        pass

    def institute2features(self):
        pass

    def guess_age(self):
        pass

    def guess_gender(self):
        pass


class PaperInfo(TextFeatures):
    def __init__(self, paper_obj_preprocessed, paper_pdf_parse = None):
        # scipdf
        #super().__init__()
        self.title = paper_obj_preprocessed["title"]
        self.abstract = paper_obj_preprocessed["abstract"]
        self.citations = paper_obj_preprocessed["outbound_citations"] # outbound citatons
        self.pdf = paper_pdf_parse

    def get_paper_topic(self):
        pass

    def get_paper_text_style(self):
        pass

    def get_semantic_reference_novelty(self, type="abstract", embedding="glove", distance="cos", percentile=100):
        cited_texts = []
        for paper_id in self.citations:
            if paper_id in ID_TO_POSITION.keys():
                cited_texts.append(ALL_PAPERS[ID_TO_POSITION[paper_id]][type])
        if len(cited_texts) > 1:
            distances = self.compute_distances(cited_texts, embedding, distance)
            novelty_score = np.percentile(distances, percentile)
        else:
            novelty_score = None
        num_citations = len(self.citations)
        num_citations_included = len(cited_texts)
        return novelty_score, num_citations, num_citations_included

    def compute_distances(self, text_list, embedding, distance):
        dist_list = []
        representation_list = self.get_representations(text_list, embedding)
        if distance == "cos":
            similarities = cosine_similarity(representation_list) # Todo: check type and shape here
            n = len(representation_list)
            for i in range(n):
                for j in range(i + 1, n):
                    dist_list.append((1 - similarities[i][j]) / 2) # Normalize to [0,1]
        return dist_list

    def get_representations(self, text_list, embedding="glove"):
        representation_list = []
        if embedding == "glove":
            glove_embeddings = load_glove("data/glove.twitter.27B.50d.txt") #Note: miss out on a lot of scientific tokens with this choice
            for text in text_list:
                text_tokenized = tokenize(text)
                word_rep_list = []
                for word in text_tokenized:
                    if word in glove_embeddings.keys():
                        word_rep_list.append(glove_embeddings[word])
                text_rep = list(np.mean(word_rep_list, 0))
                representation_list.append(text_rep)
        return representation_list



class PaperTextStyle:
    def __init__(self):
        pass

    def get_readability(self, text):
        try:
            import textstat
        except:
            import os
            os.system('pip install textstat')
            import textstat
        # reference: https://github.com/shivam5992/textstat#the-flesch-reading-ease-formula
        readability = textstat.flesch_reading_ease(text)  # gives a score of 0 (difficult) - 60 (standard) - 100 (easy)

        return readability

    def get_grammatical_errors(self):
        pass

    def get_formality(self):
        pass

    def get_num_formulas(self, paper_json):
        pass


class TweetInfo(TextFeatures):
    def get_tweet_and_meta(self):
        # tweet (as multimodal data)
        # number of likes
        # number of retweets
        pass

    def get_text_style(self):
        pass

    def get_image_or_video_style(self):
        pass

def get_id_to_pos_map(list_of_dict, id="id_semantic_scholar"):
    '''
    get a dictionary mapping ids to position in the list (e.g. paper_obj_preprocessed)
    :param list_of_dict: list of dictionaries
    :param id: id field in the dictionaries
    :return: a dictionary object
    '''
    output_map = {}
    for i, entry in enumerate(list_of_dict):
        output_map[entry[id]]=i
    return output_map

def load_glove(glove_path):
    embed_dict = {}
    with open(glove_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embed_dict[word] = vector
    return embed_dict

def tokenize(text):
    return word_tokenize(text)


def test_code():
    '''
    Just a temporary method for testing the code
    :return: void
    '''
    with open('data/metadata_0.jsonl','r') as json_files:
        json_list = list(json_files)
    json_with_abstract = []
    test_length = 100000
    for file in tqdm(json_list[:test_length]):
        metadata = json.loads(file)
        if metadata["abstract"] != None and metadata["has_outbound_citations"]:
            json_with_abstract.append(metadata)
    ID_TO_POSITION = get_id_to_pos_map(json_with_abstract, id="paper_id")
    ALL_PAPERS = json_with_abstract
    #for paper in json_with_abstract:

    for i in range(len(json_with_abstract)):
        cited_texts = []
        for paper_id in json_with_abstract[i]["outbound_citations"]:
            if paper_id in ID_TO_POSITION.keys():
                cited_texts.append(ALL_PAPERS[ID_TO_POSITION[paper_id]]["abstract"])
        if len(cited_texts) > 0:
            print(i, len(cited_texts))

    paper = PaperInfo(json_with_abstract[14884])
    paper.get_semantic_reference_novelty()


if __name__ == '__main__':
    test_code()