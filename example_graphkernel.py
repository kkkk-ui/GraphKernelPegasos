from grakel import Graph
import networkx as nx
import GraphKernelFunc as kf


# グラフ1: 三角形
G1_nx = nx.Graph()
G1_nx.add_nodes_from([(0, {"label": 0}), (1, {"label": 1}), (2, {"label": 0})])
G1_nx.add_edges_from([(0, 1), (1, 2), (2, 0)])

# グラフ2: 直線構造
G2_nx = nx.Graph()
G2_nx.add_nodes_from([(0, {"label": 1}), (1, {"label": 1}), (2, {"label": 0})])
G2_nx.add_edges_from([(0, 1), (1, 2)])

# GraKeLの形式に変換（GraKeL Graph = (edge list, node_labels dict)）
def nx_to_grakel(nx_graph):
    node_labels = nx.get_node_attributes(nx_graph, "label")
    return Graph(nx_graph.edges(), node_labels=node_labels)

G1 = nx_to_grakel(G1_nx)
G2 = nx_to_grakel(G2_nx)

# GraKeL用のグラフリスト
graph = [G1, G2]

myKernel = kf.GraghkernelFunc(2, kernelParam=1)
kernelFunc = myKernel
gm = kernelFunc.createMatrix(graph, 3)
gm_n = kf.GraghkernelFunc.normalize_gram_matrix(gm)

print("fit_transform()")
print(gm_n)

g = kf.GraghkernelFunc.k_vec_wl(graph[0], graph, 3)

print("transform()")
print(g)


