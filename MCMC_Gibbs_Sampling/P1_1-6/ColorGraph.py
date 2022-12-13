import numpy as np

class ColorGraph:
    def __init__(self, V, V_color, colors, E) -> None:
        self.V = V
        self.V_color = V_color
        self.E = E
        self.colors = colors

    def __poss_colors(self, v):
        poss_colors = self.colors.copy()

        for vertex in self.E[v]:
            neighbor_color = self.V_color[vertex]
            if neighbor_color in poss_colors:
                poss_colors.remove(neighbor_color)
                if len(poss_colors) == 0:
                    return poss_colors
        return poss_colors

    def next(self):
        v = np.random.choice(self.V)
        poss_colors = self.__poss_colors(v)
        self.V_color[v] = np.random.choice(poss_colors)
        return self.V_color

def get_colorings(color_graph, N):
    colorings = {}
    for _ in range(N):
        coloring = tuple(color_graph.next())
        colorings[coloring] = colorings.get(coloring, 0) + 1
    
    return len(colorings.keys()), [amount / N for amount in colorings.values()]
