import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

pos = {"A": (0.0, 1.0), "B": (2.0, 1.7), "C": (3.3, 0.3), "D": (1.2, -0.5), "E": (4.9, 1.3)}
# (from, to, curvature): two edges between the same vertices are drawn as
# two arcs, curving to either side
undirected = [("A", "B", 0), ("A", "D", 0), ("B", "D", 0.28), ("B", "D", -0.28),
              ("B", "C", 0), ("D", "C", 0), ("C", "E", 0)]
directed = [("A", "B", 0.28), ("A", "B", -0.28), ("A", "D", 0), ("B", "D", 0),
            ("B", "C", 0.28), ("C", "B", 0.28), ("D", "C", 0), ("C", "E", 0)]

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, edges, is_directed, title in zip(
    axs, (undirected, directed), (False, True), ("Undirected graph", "Directed graph")
):
    style = "-|>" if is_directed else "-"
    for a, b, rad in edges:
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle=style, mutation_scale=18,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=17, shrinkB=17,
                                     color="0.15", lw=1.8))
    x, y = pos["E"]  # a loop: an edge from a vertex to itself
    ax.add_patch(FancyArrowPatch((x - 0.2, y + 0.17), (x + 0.2, y + 0.17), arrowstyle=style,
                                 mutation_scale=18, connectionstyle="arc3,rad=-1.8",
                                 color="0.15", lw=1.8))
    for name, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.2, fc="#1f77b4", ec="none", zorder=3))
        ax.text(x, y, name, ha="center", va="center", color="white", fontsize=11, zorder=4)
    ax.set_xlim(-0.6, 5.7)
    ax.set_ylim(-1.0, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
fig.tight_layout()