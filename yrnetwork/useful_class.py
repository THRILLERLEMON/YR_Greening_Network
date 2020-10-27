from matplotlib.artist import Artist
from igraph import BoundingBox, Graph, palettes
import numpy as np


class GraphArtist(Artist):
    """Matplotlib artist class that draws igraph graphs.
    Only Cairo-based backends are supported.
    """
    def __init__(self, graph, bbox, palette=None, *args, **kwds):
        """Constructs a graph artist that draws the given graph within
        the given bounding box.

        `graph` must be an instance of `igraph.Graph`.
        `bbox` must either be an instance of `igraph.drawing.BoundingBox`
        or a 4-tuple (`left`, `top`, `width`, `height`). The tuple
        will be passed on to the constructor of `BoundingBox`.
        `palette` is an igraph palette that is used to transform
        numeric color IDs to RGB values. If `None`, a default grayscale
        palette is used from igraph.

        All the remaining positional and keyword arguments are passed
        on intact to `igraph.Graph.__plot__`.
        """
        Artist.__init__(self)

        if not isinstance(graph, Graph):
            raise TypeError("expected igraph.Graph, got %r" % type(graph))

        self.graph = graph
        self.palette = palette or palettes["gray"]
        self.bbox = BoundingBox(bbox)
        self.args = args
        self.kwds = kwds

    def draw(self, renderer):
        from matplotlib.backends.backend_cairo import RendererCairo
        if not isinstance(renderer, RendererCairo):
            raise TypeError(
                "graph plotting is supported only on Cairo backends")
        self.graph.__plot__(renderer.gc.ctx, self.bbox, self.palette,
                            *self.args, **self.kwds)


def z_score_normalization(x):
    x = np.array(x)
    x = (x - x.mean()) / (x.std())
    return x


def normalization(x):
    x = np.array(x)
    x = (x - x.min()) / (x.max() - x.min())
    return x


def test():
    import math

    # Make Matplotlib use a Cairo backend
    import matplotlib
    matplotlib.use("cairo.pdf")
    import matplotlib.pyplot as pyplot

    # Create the figure
    fig = pyplot.figure()

    # Create a basic plot
    axes = fig.add_subplot(111)
    xs = range(200)
    ys = [math.sin(x / 10.) for x in xs]
    axes.plot(xs, ys)

    # Draw the graph over the plot
    # Two points to note here:
    # 1) we add the graph to the axes, not to the figure. This is because
    #    the axes are always drawn on top of everything in a matplotlib
    #    figure, and we want the graph to be on top of the axes.
    # 2) we set the z-order of the graph to infinity to ensure that it is
    #    drawn above all the curves drawn by the axes object itself.
    graph = Graph.GRG(100, 0.2)
    graph_artist = GraphArtist(graph, (10, 10, 150, 150), layout="kk")
    graph_artist.set_zorder(float('inf'))
    axes.artists.append(graph_artist)

    # Save the figure
    fig.savefig("test.pdf")

    print("Plot saved to test.pdf")


if __name__ == "__main__":
    test()