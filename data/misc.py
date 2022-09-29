import numpy as np
import cv2

# convert flow map ([res,res,2]) to RGB image ([res,res,3]) according to FlowNet optical flow map
# code was taken from https://raw.githubusercontent.com/philferriere/tfoptflow/master/tfoptflow/optflow.py
def convert_flow_to_rgb(flow_np):

    # extract flow magnitude and angle per pixel
    flow_magnitude, flow_angle = cv2.cartToPolar(flow_np[..., 0].astype(np.float32), flow_np[..., 1].astype(np.float32))

    # convert nan values to 0 if there is any
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.0

    # Create HSV and normalize magnitude and angle
    hsv = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255

    # convert to RGB
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return img

# extract flow map using the Gunnar-Farneback algorithm, expect to get two grayscaled numpy images
# result will be an optical flow map of the shape [res,res,2]
def extract_flow_map(img_prev, img):

    return cv2.calcOpticalFlowFarneback(img_prev, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)


# class to hold markers swaps as graph and compute required swaps chain
class MarkersGraph(object):

    def __init__(self):
        super(MarkersGraph, self).__init__()

        self.edges = []
        self.nodes_to_remove = []

    def add_node_to_remove(self, node_name):

        # collect nodes that received glitches and needs to be removed
        self.nodes_to_remove.append(node_name)

    def add_edge(self, node1, node2):

        # collect edges that needs to be swapped
        self.edges.append([node1, node2])

    def compute_swaps(self):

        # iterate over all edges in markers graph
        swaps_chain = []
        for i, edge in enumerate(self.edges):
            
            # take swap into account only if haven't taken yet
            if edge in swaps_chain or edge[::-1] in swaps_chain:
                continue

            # add swap action
            if edge[0] != edge[1]:
                swaps_chain.append(edge)

            # iterate over the rest of the edges to replace nodes after swaps
            for j in range(i+1, len(self.edges)):

                # replace source node with the other one of the edge if found
                if self.edges[j][0] == edge[0]:
                    self.edges[j][0] = edge[1]
                elif self.edges[j][0] == edge[1]:
                    self.edges[j][0] = edge[0]
                
                # replace target node with the other one of the edge if found
                # if self.edges[j][1] == edge[0]:
                #     self.edges[j][1] = edge[1]
                # elif self.edges[j][1] == edge[1]:
                #     self.edges[j][1] = edge[0]

        return swaps_chain