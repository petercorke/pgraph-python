import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

pos = {"a": (0.0, 0.0), "b": (2.0, 0.0), "c": (4.0, 0.0)}
# (name, from, to, curvature, label offset): e1 and e2 are parallel, drawn
# as two arcs curving to either side
edges = [("e1", "a", "b", 0.3, (0, 0.5)), ("e2", "a", "b", -0.3, (0, -0.5)),
         ("e3", "b", "c", 0, (0, 0.18))]

fig, axs = plt.subplots(1, 2, figsize=(9, 2.3))
for ax, is_directed, title in zip(axs, (False, True), ("UGraph", "DGraph")):
    style = "-|>" if is_directed else "-"
    for name, p, q, rad, (dx, dy) in edges:
        ax.add_patch(FancyArrowPatch(pos[p], pos[q], arrowstyle=style, mutation_scale=16,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=15, shrinkB=15,
                                     color="0.15", lw=1.6))
        mx = (pos[p][0] + pos[q][0]) / 2 + dx
        my = (pos[p][1] + pos[q][1]) / 2 + dy
        ax.text(mx, my, name, ha="center", va="center", fontsize=10, color="0.15")
    x, y = pos["c"]  # e4: a loop, an edge from c to itself
    ax.add_patch(FancyArrowPatch((x - 0.15, y + 0.15), (x + 0.15, y + 0.15), arrowstyle=style,
                                 mutation_scale=16, connectionstyle="arc3,rad=-1.8",
                                 color="0.15", lw=1.6))
    ax.text(x, y + 0.62, "e4", ha="center", va="center", fontsize=10, color="0.15")
    for name, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.2, fc="#1f77b4", ec="none", zorder=3))
        ax.text(x, y, name, ha="center", va="center", color="white", fontsize=11, zorder=4)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-0.8, 0.95)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
fig.tight_layout()