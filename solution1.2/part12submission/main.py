import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path
import scipy.io
import random
import math
import time
import sys
import statistics
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import networkx as nx


# Function to visualize the structure of Image_Homographie as a tree
def visualize_structure(root_node):
    # Create a directed graph
    graph = nx.DiGraph()

    # Recursive function to add nodes and edges
    def add_nodes_edges(node):
        graph.add_node(node, label=str(node))
        for child in node.childs:
            graph.add_edge(node, child)
            add_nodes_edges(child)

    # Add nodes and edges starting from the root node
    add_nodes_edges(root_node)

    # Generate a tree-like layout manually
    def tree_layout(graph, root, level=0, pos=None, x=0, width=1, depth_spacing=1):
        if pos is None:
            pos = {}
        pos[root] = (x, -level * depth_spacing)
        children = list(graph.successors(root))
        if children:
            dx = width / len(children)
            next_x = x - width / 2 + dx / 2
            for child in children:
                pos = tree_layout(graph, child, level + 1, pos, next_x, dx, depth_spacing)
                next_x += dx
        return pos

    root = root_node
    pos = tree_layout(graph, root)

    # Draw the graph
    labels = nx.get_node_attributes(graph, 'label')

    plt.figure(figsize=(12, 10))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=2000, node_color="lightblue", font_size=10,
            font_weight="bold", arrowsize=20)
    plt.title("Visualization of Image_Homographie Structure as a Tree")
    plt.show()


arguments = sys.argv[1:]
print(arguments)

ref_dir = arguments[0]
N_cameras = len(arguments) - 1
if N_cameras % 2 == 1:
    print('No output path found')
else:
    N_cameras = int(N_cameras / 2)
Cameras = [[{}, {}, {}] for n in
           range(N_cameras)]  # images, Yolos(if available), keypoints #apperntly all of this must be dictionaries
for i in range(N_cameras):
    dir = arguments[i * 2 + 1]
    output_dir = arguments[i * 2 + 2]
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    files = os.listdir(dir)
    files.sort()
    images = Cameras[i][0]
    yolo = Cameras[i][1]
    kp = Cameras[i][2]
    for file in files:
        if file[-4:] == '.jpg':
            images[int(file[4:8])] = np.array(Image.open(dir + '/' + file))
        if file[-4:] == '.mat':
            if file[:4] == 'yolo':
                yolo[int(file[5:9])] = scipy.io.loadmat(dir + '/' + file)
            elif file[:2] == 'kp':
                kp[int(file[3:7])] = scipy.io.loadmat(dir + '/' + file)

ref_image = np.array(Image.open(ref_dir + '/img_ref.jpg'))
ref_kp = scipy.io.loadmat(ref_dir + '/kp_ref.mat')


class Image_Homographie():
    def __init__(self, H=None, parent=None, c=None, i=None, depth=0):
        self.H = H
        self.childs = []
        self.parent = parent
        self.c = c
        self.i = i
        self.depth = depth

    def create_child(self, H, c, i):
        child = Image_Homographie(H, self, c, i, self.depth + 1)
        self.childs.insert(0, child)

    def __repr__(self):
        return f'{self.c} {self.i} {self.depth} childs:{len(self.childs)}'


class breadth_first():
    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.fifo = [parent_node]
        self.opened = []

    def next(self):
        if len(self.fifo):
            node = self.fifo.pop(0)
            self.opened.append(node)
            self.fifo = self.fifo + node.childs
            return node
        return None

    def update(self, opened):
        for i in opened:
            self.fifo = self.fifo + i.childs
            if i in self.fifo: self.fifo.remove(i)
        self.opened = opened


def getMatches(kpFrame, kpRef):
    desc_ref, desc_input, kp_ref, kp_input = kpRef['desc'], kpFrame['desc'], kpRef['kp'], kpFrame['kp']

    kdtree = KDTree(desc_input)
    distances, indices = kdtree.query(desc_ref, k=1)

    matches = [
        (
            kp_ref[i][0],  # First value from kp_ref
            kp_ref[i][1],  # Second value from kp_ref
            kp_input[indices[i]][0],  # First value from kp_input
            kp_input[indices[i]][1]  # Second value from kp_input
        )
        for i in range(len(kp_ref))
    ]

    formatted_matches = [
        np.array(kp_ref[i].tolist() + kp_input[indices[i]].tolist())
        for i in range(len(kp_ref))
    ]

    return formatted_matches[:150]


def distance(v0, vref, m):
    d = 0
    for i in range(len(v0)):
        d += (v0[i] - vref[i]) ** 2
        if d > m: return d, False
    return d, True


def getMatches2(kpFrame, kpRef):
    _matches = []
    matches = []
    indexes = random.choices(range(len(kpFrame['desc'])), k=300)
    desc = kpFrame['desc']
    desc_ref = kpRef['desc']
    kpFrame = kpFrame['kp']
    kpref = kpRef['kp']
    for i in indexes:
        d = desc[i]
        index = -1
        n = 100
        for j, _d in enumerate(desc_ref):
            q, flag = distance(d, _d, n)
            if flag and q < n: n = q;index = j;
        _matches.append([n, index, i])
        # _matches.append(n)
    '''
    m=np.array(_matches)[:,0]
    print('\tStats Before:')
    print('\t\tMean:',statistics.mean(m))
    print('\t\tSTdev:',statistics.stdev(m))
    print('\t\tVariance:',statistics.variance(m))
    print('\t\tMode:',statistics.mode(m))
    print('\t\tMedian:',statistics.median(m))
    q1,q2,q3=statistics.quantiles(m, method='inclusive', n=4)
    print('\t\tQ1:',q1,'\n\t\tQ2:',q2,'\n\t\tQ3:',q3)
    '''
    _matches.sort()
    _matches = _matches[:150]
    '''
    m=np.array(_matches)[:,0]
    print('\tStats After:')
    print('\t\tMean:',statistics.mean(m))
    print('\t\tSTdev:',statistics.stdev(m))
    print('\t\tVariance:',statistics.variance(m))
    print('\t\tMode:',statistics.mode(m))
    print('\t\tMedian:',statistics.median(m))
    q1,q2,q3=statistics.quantiles(m, method='inclusive', n=4)
    print('\t\tQ1:',q1,'\n\t\tQ2:',q2,'\n\t\tQ3:',q3)
    '''
    for j in _matches:
        matches.append(np.array(kpref[j[1]].tolist() + kpFrame[j[2]].tolist()))  # , dtype=np.uint16
    return matches


def compute_homography_with_RANSAC(camera_index, image_index, ref_camera=None, ref_index=None):
    print(f'Homography {camera_index} {image_index} with {ref_camera} {ref_index}')
    kp = Cameras[camera_index][2][image_index]
    kp_ref = None
    if ref_camera == None:
        kp_ref = ref_kp
    else:
        kp_ref = Cameras[ref_camera][2][ref_index]
    matches = getMatches(kp, kp_ref)
    n_needed = 4
    P = 0.99
    p = 0.15  # maybe 0.15? Try it... it should be faster
    iterations = int(math.log(1 - P) / math.log(1 - p ** n_needed))  # 72#min(50, int(len(matches)/4))
    minimal_inliners = int(0.15 * len(matches)) + 4
    h, w = Cameras[camera_index][0][image_index].shape[:2]
    max_error = max(2, int(w * h * 0.0000005))  # Erro de 0.00005%
    print('\tMax error:', max_error)
    best = [-1, 0]  # 0: nº de inliners; 1: index of inliners
    A = []
    i = 0
    while i < iterations and minimal_inliners > best[0]:
        pairs = random.choices(matches, k=n_needed)
        A = []
        for x, y, xx, yy in pairs:
            # xx=-float(xx) #Avoid Overflow error
            # yy=-float(yy) #Avoid Overflow error
            A.append([x, y, 1, 0, 0, 0, -xx * x, -xx * y, -xx])
            A.append([0, 0, 0, x, y, 1, -yy * x, -yy * y, -yy])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        inliners = []
        for j, p in enumerate(matches):
            x, y, xx, yy = p
            w = H @ np.array([x, y, 1])
            if w[2] == 0: continue
            w /= w[2]
            _xx, _yy = w[:2]
            erro = math.sqrt((xx - _xx) ** 2 + (yy - _yy) ** 2)
            # print(j, xx, yy, _xx, _yy, erro)
            if erro < max_error:
                inliners.append(j)
        if len(inliners) > best[0]:
            best[0] = len(inliners)
            best[1] = inliners.copy()
        i += 1
    if i == iterations: return False
    print('\tIterations: %d' % (i))
    print('\tInliners: %.3f %% %d' % (best[0] / len(matches) * 100, best[0]))
    inliners = [matches[i] for i in best[1]]
    A = []
    for x, y, xx, yy in inliners:
        A.append([x, y, 1, 0, 0, 0, -xx * x, -xx * y, -xx])
        A.append([0, 0, 0, x, y, 1, -yy * x, -yy * y, -yy])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[-1][-1]
    # print(H)
    return H


main_node = Image_Homographie(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def compute_homographies():
    homographies = [{k: None for k in Cameras[c][0].keys()} for c in range(len(Cameras))]
    redo = []
    for c in range(len(Cameras)):
        h = []
        for i in Cameras[c][0].keys():
            start = time.time()
            print(f'Homographie with Cameras[{c}]images[{i}]')
            s = breadth_first(main_node)
            n = s.next()
            result = False
            while n != None and type(result) == bool:
                result = compute_homography_with_RANSAC(c, i, n.c, n.i)
                if type(result) == bool:  # and result==False:
                    print('\tFail')
                    n = s.next()
                else:
                    n.create_child(result @ n.H, c, i)

            if n == None:
                redo.append([c, i, s.opened.copy(), 1000000])
            elif n.parent != None:
                redo.append([c, i, s.opened.copy(), n.depth + 1])
                homographies[c][i] = n.childs[-1].H
            else:
                homographies[c][i] = n.childs[-1].H
            delta = time.time() - start
            print('Processing Time in %.3f s' % (delta))
        visualize_structure(main_node)

    print('Redo')
    redo.reverse()
    for j in redo:
        c, i, opened, depth = j
        print(f'Homographie with Cameras[{c}]images[{i}]')
        s.update(opened)
        n = s.next()
        result = False
        while n != None and (n.depth + 1 >= depth or (n.c == c and n.i == i)):
            print(f'\t{n.c} {n.i} {n.depth}')
            n = s.next()
        while n != None and type(result) == bool and n.depth + 1 < depth:
            result = compute_homography_with_RANSAC(c, i, n.c, n.i)
            if type(result) == bool and result == False:
                n = s.next()
                if n != None and n.c == c and n.i == i:
                    n = s.next()
            else:
                n.create_child(result @ n.H, c, i)
        if type(result) != bool:
            homographies[c][i] = n.childs[-1].H
        else:
            print('\tEnable to do')
    visualize_structure(main_node)
    return homographies


h = compute_homographies()

###Yolo
output_yolo = [[] for c in range(N_cameras)]
for c in range(N_cameras):
    yolos = Cameras[c][1]
    k = list(yolos.keys())
    output = {}
    for i in k:
        _yolo = yolos[i]
        o = {'bbox': [], 'id': _yolo['id'], 'class': _yolo['class']}
        H = np.linalg.inv(h[c][i])
        for pair in _yolo['xyxy']:
            x1, y1, x2, y2 = pair
            x1 = int(x1);
            x2 = int(x2);
            y1 = int(y1);
            y2 = int(y2);
            vertices = [[x1, y1, 1], [x1, y2, 1], [x2, y1, 1], [x2, y2, 1]]
            max_y = 0
            min_y = 2000000
            min_x = 2000000
            max_x = 0
            for v in vertices:
                w = H @ np.array(v)
                w /= w[2]
                x = int(w[0])
                y = int(w[1])
                if x > max_x: max_x = x;
                if x < min_x: min_x = x;
                if y > max_y: max_y = y;
                if y < min_y: min_y = y;
            o['bbox'].append([min_x, min_y, max_x, max_y])
        output[y] = o
    output_yolo[c] = output

# Saving everything
for c in range(N_cameras):
    H = {'H': None}
    A = [[[], [], []], [[], [], []], [[], [], []]]
    keys = list(Cameras[c][0].keys())
    for i in keys:
        A[0][0].append(h[c][i][0][0])
        A[0][1].append(h[c][i][0][1])
        A[0][2].append(h[c][i][0][2])
        A[1][0].append(h[c][i][1][0])
        A[1][1].append(h[c][i][1][1])
        A[1][2].append(h[c][i][1][2])
        A[2][0].append(h[c][i][2][0])
        A[2][1].append(h[c][i][2][1])
        A[2][2].append(h[c][i][2][2])
    H['H'] = np.array(A)
    scipy.io.savemat(arguments[(c + 1) * 2] + '/homographies.mat', H)
    keys = list(output_yolo[c].keys())
    for k in keys:
        scipy.io.savemat(arguments[(c + 1) * 2] + f'/yolooutput_{k:05d}.mat', output_yolo[c][k])
