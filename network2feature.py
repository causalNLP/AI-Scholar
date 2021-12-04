import networkx as nx

class Network(object):
    def __init__(self, iterable, id):
        self.load_network(iterable, id)

    def load_network(self, iterable, id):
        self.network = nx.DiGraph()
        for unit in iterable:
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
    def __init__(self, paper_obj_preprocessed):
        self.network = nx.Graph()
        for paper in paper_obj_preprocessed:
            authors = paper["authors"]
            self.network.add_nodes_from(authors)
            new_coauthor_edges = [(i,j) for i in authors for j in authors[authors.index(i)+1:]]
            self.network.add_edges_from(new_coauthor_edges)
        self.network.add_nodes_from(authors)


class CitationNetwork(Network):
    def __init__(self, paper_obj_preprocessed):
        Network.__init__(self, paper_obj_preprocessed, "id_semantic_scholar")
        for paper in paper_obj_preprocessed:
            self_id = paper["id_semantic_scholar"]
            cited_by_ids = [citing_paper["id_semantic_scholar"] for citing_paper in paper["cited_by"]]
            edges = [(cited_by_id, self_id) for cited_by_id in cited_by_ids]
            self.network.add_edges_from(edges)


class TwitterFollowerNetwork(Network):
    pass


class RetweetNetwork(Network):
    pass

if __name__ == '__main__':
    ex_object = [{"id_semantic_scholar": 1, "cited_by": [{"id_semantic_scholar":2},{"id_semantic_scholar":3}], "authors":["andreas","daphna","zhijing","mrinmaya"]},
                 {"id_semantic_scholar":2, "cited_by":[{"id_semantic_scholar":3}], "authors":["zhijing","mrinmaya","yvonne"]},
                 {"id_semantic_scholar":3, "cited_by":[], "authors":["andreas","daphna"]}]
    citation_graph = CitationNetwork(ex_object)
    print("Citation in degrees:", citation_graph.get_in_degree_centrality())
    coauthor_graph = CoauthorshipNetwork(ex_object)
    print("Authors:", coauthor_graph.get_network().nodes)
    print("Betweenness: ", coauthor_graph.get_betweenness_centrality())