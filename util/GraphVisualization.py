import networkx as nx
import matplotlib.pyplot as plt


# Defining a Class
class GraphVisualization:

    def __init__(self):
        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self, pos_node=None):
        # G = nx.complete_graph(3, create_using=nx.DiGraph)
        # G.add_edge(0, 0)
        # pos = nx.circular_layout(G)  spring_layout
        #
        plt.figure(1,figsize=(40,30))
        G = nx.DiGraph()
        G.add_edges_from(self.visual)
        if pos_node is None:
            pos = nx.circular_layout(G)
        else:
            pos = pos_node
        # pos = self.hierarchy_pos(G, 0)

        # As of version 2.6, self-loops are drawn by default with the same styling as
        # other edges
        # nx.draw(G, pos, with_labels=True)

        # Add self-loops to the remaining nodes
        # edgelist = [(1, 1), (2, 2)]
        # G.add_edges_from(edgelist)

        # Draw the newly added self-loops with different formatting
        # nx.draw_networkx_edges(G, pos, edgelist=edgelist, arrowstyle="<|-", style="dashed")

        edge_color = [0.5] * len(self.visual)

        nx.draw(G, pos, with_labels=True, font_color="whitesmoke", node_size=500)
        # nodes = nx.draw_networkx_nodes(G, pos, node_size=300, node_color="indigo")
        edges = nx.draw_networkx_edges(
            G,
            pos,
            node_size=300,
            arrowstyle="->",
            arrowsize=40,
            width=3,
            edge_color=edge_color,
            edge_cmap=plt.cm.Greys,
            edge_vmin=0,
            edge_vmax=1
        )

        # g = nx.DiGraph()
        # g.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (2, 7), (3, 8), (3, 9),
        #                   (4, 10), (5, 11), (5, 12), (6, 13)])
        # p = nx.drawing.nx_pydot.to_pydot(g)
        plt.axis("off")
        plt.savefig('./result_snps/snps_graph.png', dpi=600)
        plt.show()

        # G = nx.Graph()
        # G.add_edges_from(self.visual)
        # pos = nx.circular_layout(G)
        # nx.draw(G, with_labels=True, font_color="whitesmoke")
        # nx.draw_networkx_edges(G, pos, arrowstyle="<|-", style="dashed")
        # plt.axis("off")
        # plt.show()

    def hierarchy_pos(self, G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        '''If there is a cycle that is reachable from root, then result will not be a hierarchy.

           G: the graph
           root: the root node of current branch
           width: horizontal space allocated for this branch - avoids overlap with other branches
           vert_gap: gap between levels of hierarchy
           vert_loc: vertical location of root
           xcenter: horizontal location of root
        '''
        pos = {}
        def h_recur(G, root, width=1., vert_gap=0.2, vert_loc=0.0, xcenter=0.5,
                    pos=None, parent=None, parsed=[]):
            if (root not in parsed):
                parsed.append(root)
                # if pos == None:
                #     pos = {root: (xcenter, vert_loc)}
                # else:
                pos[root] = (xcenter, vert_loc)
                neighbors = G.neighbors(root)
                neighbors = list(neighbors)
                if parent != None:
                    neighbors.remove(parent)

                num_neig = len(list(neighbors))
                if num_neig > 0:
                    dx = width / num_neig
                    nextx = xcenter - width / 2 - dx / 2
                    for neighbor in list(neighbors):
                        nextx += dx
                        pos = h_recur(G, neighbor, width=dx, vert_gap=vert_gap,
                                      vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos,
                                      parent=root, parsed=parsed)
            return pos

        return h_recur(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=pos)

    def hierarchy_pos2(self, G, root, levels=None, width=1., height=1.):
        '''If there is a cycle that is reachable from root, then this will see infinite recursion.
           G: the graph
           root: the root node
           levels: a dictionary
                   key: level number (starting from 0)
                   value: number of nodes in this level
           width: horizontal space allocated for drawing
           height: vertical space allocated for drawing'''
        TOTAL = "total"
        CURRENT = "current"

        def make_levels(levels, node=root, currentLevel=0, parent=None):
            """Compute the number of nodes for each level
            """
            if not currentLevel in levels:
                levels[currentLevel] = {TOTAL: 0, CURRENT: 0}
            levels[currentLevel][TOTAL] += 1
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                if not neighbor == parent:
                    levels = make_levels(levels, neighbor, currentLevel + 1, node)
            return levels

        def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
            dx = 1 / levels[currentLevel][TOTAL]
            left = dx / 2
            pos[node] = ((left + dx * levels[currentLevel][CURRENT]) * width, vert_loc)
            levels[currentLevel][CURRENT] += 1
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                if not neighbor == parent:
                    pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc - vert_gap)
            return pos

        if levels is None:
            levels = make_levels({})
        else:
            levels = {l: {TOTAL: levels[l], CURRENT: 0} for l in levels}
        vert_gap = height / (max([l for l in levels]) + 1)
        return make_pos({})

if __name__ == "__main__":
    # Driver code
    G = GraphVisualization()
    G.addEdge(1, 0)
    G.addEdge(2, 0)
    G.addEdge(1, 2)
    G.addEdge(3, 1)
    G.addEdge(4, 1)
    G.addEdge(5, 1)
    G.addEdge(5, 2)
    G.visualize()