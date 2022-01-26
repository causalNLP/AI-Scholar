import networkx as nx
import json
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Network(object):
    def __init__(self, iterable, id):
        self.load_network(iterable, id)

    def load_network(self, iterable, id):
        self.network = nx.DiGraph()
        for unit in tqdm(iterable):
            self.network.add_node(unit[id])

    def get_network(self):
        return self.network

    def network2features(self, network):
        pass

    def get_in_degree_centrality(self):
        return nx.in_degree_centrality(self.network)

    def get_out_degree_centrality(self):
        return nx.out_degree_centrality(self.network)

    def get_betweenness_centrality(self):
        return nx.algorithms.centrality.betweenness_centrality(self.network, normalized=True)

    def get_closeness_centrality(self, incoming = True):
        '''
        :return: the Wasserman and Faust generalized closeness centrality for isolated components
        '''
        if incoming:
            return nx.algorithms.centrality.closeness_centrality(self.network)
        else:
            return nx.algorithms.centrality.closeness_centrality(self.network.reverse())

    def get_clustering_coefficient(self):
        '''
        :return: the clustering coefficient for all nodes as fraction of possible triangles in its neighborhood
        range [0,1]
        '''
        return nx.algorithms.cluster.clustering(self.network)

class CoauthorshipNetwork(Network):
    def __init__(self, paper_obj_preprocessed, id):
        #self.network = nx.Graph()
        self.load_network(paper_obj_preprocessed, id)
        #for paper in paper_obj_preprocessed:
        #    authors = paper["authors"]
        #    self.network.add_nodes_from(authors)
        #    new_coauthor_edges = [(i,j) for i in authors for j in authors[authors.index(i)+1:]]
        #    self.network.add_edges_from(new_coauthor_edges)

    def load_network(self, iterable, id):
        self.network = nx.Graph()
        # want to create nodes by id and add all other fields as attributes
        for paper in tqdm(iterable):
            authors = paper["authors"]
            if authors == None:
                continue
            else:
                authors = [elem for elem in authors if elem[id] != None]
                ids = [elem[id] for elem in authors]
                self.network.add_nodes_from(zip(ids,authors))
                new_coauthor_edges = [(i,j) for i in ids for j in ids[ids.index(i)+1:]]
                self.network.add_edges_from(new_coauthor_edges)

    def get_degree_centrality(self):
        return nx.degree_centrality(self.network)

class CitationNetwork(Network):
    def __init__(self, paper_obj_preprocessed):
        Network.__init__(self, paper_obj_preprocessed, "paperId")
        for paper in paper_obj_preprocessed:
            self_id = paper["paperId"]
            cited_by_ids = [citing_paper["paperId"] for citing_paper in paper["cited_by"]]
            edges = [(cited_by_id, self_id) for cited_by_id in cited_by_ids]
            self.network.add_edges_from(edges)


class TwitterFollowerNetwork(Network):
    pass


class RetweetNetwork(Network):
    pass

if __name__ == '__main__':
    ex_object = [{"paperId": 1, "cited_by": [{"paperId":2},{"paperId":3}], "authors":[{"authorId": 1, "name": "andreas"},{"authorId": 2, "name": "daphna"},{"authorId": 3, "name": "zhijing"},{"authorId": 4, "name": "mrinmaya"}]},
                 {"paperId":2, "cited_by":[{"paperId":3}], "authors":[{"authorId": 3, "name": "zhijing"},{"authorId": 4, "name": "mrinmaya"},{"authorId": 5, "name": "yvonne"}]},
                 {"paperId":3, "cited_by":[], "authors":[{"authorId": 1, "name": "andreas"},{"authorId": 2, "name": "daphna"}]}]
    citation_graph = CitationNetwork(ex_object)
    print("Citation in degrees:", citation_graph.get_in_degree_centrality())
    coauthor_graph = CoauthorshipNetwork(ex_object, "authorId")
    print("Authors:", coauthor_graph.get_network().nodes)
    print("Betweenness: ", coauthor_graph.get_betweenness_centrality())


    with open('data/preliminary_nlp_papers.jsonl','r') as json_files:
        json_list = list(json_files)
    paper_dictionary = []
    for file in json_list:
        paper_dictionary.append(json.loads(file))

    coauthor_graph_nlp = CoauthorshipNetwork(paper_dictionary, "authorId")
    G = coauthor_graph_nlp.get_network()
    centralities = coauthor_graph_nlp.get_degree_centrality()
    num_nodes = len(G.nodes)
    centralities_values = np.array(list(centralities.values()))
    sns.histplot(centralities_values*(num_nodes-1),
                 stat="density", bins = 15).set_title("Histogram over co-auhorship degrees")
    top_10_deg = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    for authorId, degree in top_10_deg:
        print(G.nodes[authorId]["name"], "has", degree, "co-authors")

    subG1 = G.subgraph(G.nodes[:10])
    nx.draw_random(subG1, labels = nx.get_node_attributes(subG1, "name"))
    subG2 = G.subgraph(list(G.nodes)[-10:])
    nx.draw_random(subG2, labels=nx.get_node_attributes(subG2, "name"))



    print(len(paper_dictionary))