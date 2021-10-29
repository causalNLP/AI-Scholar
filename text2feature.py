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
    def paper_pdf2json(self):
        # scipdf
        pass

    def get_paper_topic(self):
        pass

    def get_paper_text_style(self):
        pass

    def get_novelty(self):
        pass


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
