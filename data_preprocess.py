class ScholarDatabase:
    def __init__(self):
        pass

    def get_citation_network(self, only_significant_citation=False):
        if only_significant_citation:
            # use Semantic Scholar's tool to detect significant citations
            pass
        pass

    def get_coauthorship_network(self):
        pass

    def get_twitter_follower_network(self):
        pass


class PaperDatabase:
    def get_all_papers_by_author(self, author_name):
        pass


class Paper:
    def get_authors(self):
        pass

    def get_year(self):
        pass

    def get_pdf(self):
        pass

    def pdf2json(self):
        pass

    def get_total_citations(self):
        pass

    def get_yearly_citations(self):
        pass

    def get_citers(self):
        pass


class TwitterDatabase:
    def get_all_tweets_by_author(self, author_name):
        pass


class Tweet:
    def get_content(self):
        pass

    def get_paper(self):
        pass  # return a Paper object

    def get_likers(self):
        pass

    def get_retweeters(self):
        pass
