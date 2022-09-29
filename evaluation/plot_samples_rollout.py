import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

FONT_SIZE = 14
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (8,1.5)
ROLLOUT_LENGTH = 8
COLORS = ['black', 'goldenrod', 'deepskyblue']
LABELS = ['GT', 'MF', 'CBMF']
CLASSES_NAMES = [r'$F^1$', r'$F^2$', r'$F^3$', r'$F^4$', r'$F^5$']
NUM_CLASSES = len(CLASSES_NAMES)
GRAPH_NAME = "K2"
GRAPH_FORMAT = 'pdf'

# piano playing examples
# example 1
if GRAPH_NAME == "P1":
    gt_cells = [(1,0),(1,1),(1,2),(0,3),(0,4),(0,5),(0,6),(0,7)]
    mf_cells = [(1,0),(1,1),(1,2),(0,4),(0,5),(0,6),(0,7)]
    cbmf_cells = [(1,0),(1,1),(1,2),(0,3),(0,4),(0,5),(0,6),(0,7)]
elif GRAPH_NAME == "P2":
    gt_cells = [(0,2),(0,3),(0,4),(0,5)]
    mf_cells = [(0,3),(0,4)]
    cbmf_cells = [(0,2),(0,3),(0,4),(0,5)]
elif GRAPH_NAME == "K1":
    gt_cells = [(2,1),(2,2),(2,3),(2,4),(2,5),(2,6)]
    mf_cells = [(2,2),(2,3),(2,4),(2,5)]
    cbmf_cells = [(2,1),(2,2),(2,3),(2,4),(2,5),(2,6)]
else: #K2
    gt_cells = [(4,1),(4,2),(4,3),(4,4),(4,5)]
    mf_cells = [(4,1),(3,3),(3,4)]
    cbmf_cells = [(4,1),(4,2),(4,3),(4,4),(4,5),(4,6)]


all_cells = [gt_cells, mf_cells, cbmf_cells]

# create customized grid to differ between fingers
for i in range(1, NUM_CLASSES):
    plt.plot([0,8], [i]*2, color='lightgrey', zorder=0)
for i in range(1, ROLLOUT_LENGTH):
    plt.plot([i]*2, [0,5], color='lightgrey', zorder=0)

# prepare graph
step_size = 1
eps = 1e-1
x = 0
gca = plt.gca()

# draw all cells
for i, method_cells in enumerate(all_cells):
    for cell in method_cells:
        cell_origin = (cell[1], cell[0]+((2-i)*0.33))
        gca.add_patch(Rectangle(cell_origin, 1, 0.33, fill=True, color=COLORS[i], zorder=5))

# prepare customized legend
legend_lines = [Line2D([0], [0], color=y, label=x, linewidth=3, linestyle='-') for x,y in zip(LABELS, COLORS)]
metrics_legend = plt.legend(handles=legend_lines, loc='upper right', fontsize=FONT_SIZE-4)
#metrics_legend = plt.legend(handles=legend_lines, loc='lower right', fontsize=FONT_SIZE-4)
plt.gca().add_artist(metrics_legend)

# prepare axes
plt.xticks(color='w', fontsize=FONT_SIZE)
plt.yticks(np.arange(0.5, NUM_CLASSES+0.5, step=1), CLASSES_NAMES, fontsize=FONT_SIZE)
plt.xlim([0.0, 8.0])
plt.ylim([0.0, 5.0])

# save plot
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join("sample_rollout_{}.{}".format(GRAPH_NAME, GRAPH_FORMAT)))