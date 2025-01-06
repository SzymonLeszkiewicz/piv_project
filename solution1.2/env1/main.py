import argparse
import os
import numpy as np
from PIL import Image
import scipy.io
import random
import math
import time
import sys
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from math import ceil
from scipy.io import savemat


# Argument parsing function
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Program to calculate homographies and process data from multiple cameras"
    )
    parser.add_argument(
        "reference_dir",
        type=str,
        help="Path to the directory containing the reference image and its keypoints"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=str,
        help=(
            "List of input directories containing images and corresponding output directories. "
            "Each input directory must be paired with an output directory."
        ),
        required=True,
    )
    args = parser.parse_args()

    # Validate: Ensure the number of input and output directories is even
    if len(args.inputs) % 2 != 0:
        parser.error("The number of input and output directories must be even.")

    # Pair input and output directories
    cameras = []
    for i in range(0, len(args.inputs), 2):
        input_dir = args.inputs[i]
        output_dir = args.inputs[i + 1]
        cameras.append((input_dir, output_dir))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Create output directories if they don't exist

    return args.reference_dir, cameras


def load_data(ref_dir, cameras):
    # Example: Demonstrating the usage of parsed arguments
    global images
    ref_image_path = os.path.join(ref_dir, "img_ref.jpg")
    ref_kp_path = os.path.join(ref_dir, "kp_ref.mat")

    if not os.path.exists(ref_image_path) or not os.path.exists(ref_kp_path):
        print("Error: Reference image or keypoints file not found in the reference directory.")
        sys.exit(1)

    ref_image = np.array(Image.open(ref_image_path))
    ref_kp = scipy.io.loadmat(ref_kp_path)['kp']
    ref_desc = scipy.io.loadmat(ref_kp_path)['desc']

    # Load images and data from each camera input directory
    for input_dir, output_dir in cameras:
        if not os.path.exists(input_dir):
            print(f"Error: Input directory {input_dir} does not exist.")
            continue

        files = os.listdir(input_dir)
        images = {}
        yolos = {}
        matches_data = {}

        for file in files:
            file_path = os.path.join(input_dir, file)
            if file.endswith('.jpg'):
                image_index = int(file[-8:8])  # Assuming file format is like 'img_XXXX.jpg'
                images[image_index] = np.array(Image.open(file_path))
            elif file.endswith('.mat'):
                if file.startswith('yolo'):
                    yolo_index = int(file[5:9])
                    yolos[yolo_index] = scipy.io.loadmat(file_path)
                elif file.startswith('kp'):
                    kp_index = int(file[3:7])
                    kp_data = scipy.io.loadmat(file_path)

                    matches_data[kp_index] = {
                        'keypoints': kp_data['kp'],
                        'descriptors': kp_data['desc']
                    }

        print(
            f"Loaded {len(images)} images, {len(yolos)} YOLO files, and {len(matches_data)} keypoint files from {input_dir}.")
        yield ref_image, ref_kp, ref_desc, images, yolos, matches_data


def match_keypoints_optimized(desc_ref, desc_input, kp_ref, kp_input):
    kdtree = KDTree(desc_input)
    distances, indices = kdtree.query(desc_ref, k=1)

    matches = [
        (
            kp_ref[i][0],  # Pierwsza wartość z kp_ref
            kp_ref[i][1],  # Druga wartość z kp_ref
            kp_input[indices[i]][0],  # Pierwsza wartość z kp_input
            kp_input[indices[i]][1]  # Druga wartość z kp_input
        )
        for i in range(len(kp_ref))
    ]
    return matches


def display_matches(ref_image, kp_ref, input_image, kp_input, matches):
    """
    Wyświetla dopasowania punktów kluczowych między obrazem referencyjnym a wejściowym.

    Args:
        ref_image (PIL.Image): Obraz referencyjny.
        kp_ref (np.ndarray): Kluczowe punkty obrazu referencyjnego.
        input_image (PIL.Image): Obraz wejściowy.
        kp_input (np.ndarray): Kluczowe punkty obrazu wejściowego.
        matches (list): Lista dopasowań punktów [(x_ref, y_ref), (x_input, y_input)].
    """
    plt.figure(figsize=(15, 8))

    # Rysowanie obrazów obok siebie
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(ref_image, cmap='gray')
    ax1.scatter(kp_ref[:, 0], kp_ref[:, 1], c='blue', s=5, label="Keypoints Ref")
    ax1.set_title("Reference Image")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(input_image, cmap='gray')
    ax2.scatter(kp_input[:, 0], kp_input[:, 1], c='red', s=5, label="Keypoints Input")
    ax2.set_title("Input Image")

    plt.suptitle("Matched Keypoints")
    plt.show()


def sample_homography(sample):
    """
    Oblicza macierz homografii na podstawie podanych punktów.

    Args:
        sample (np.ndarray): Macierz punktów [(x1, y1, x2, y2)].

    Returns:
        np.ndarray: Macierz homografii 3x3.
    """
    A = []
    for x, y, xx, yy in sample:
        A.append([x, y, 1, 0, 0, 0, -xx * x, -xx * y, -xx])
        A.append([0, 0, 0, x, y, 1, -yy * x, -yy * y, -yy])
    A = np.array(A)

    # Obliczanie macierzy homografii za pomocą SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]  # Normalizacja
    return H


def compute_homography_with_RANSAC(matches, n_needed=4, P=0.99, p=0.1, max_error=3):
    """
        Main function to compute homography using RANSAC.

        Args:
            matches (list): List of matches [(x1, y1, x2, y2)].
            n_needed (int): Minimum number of points to compute homography.
            P (float): Desired probability of finding a correct homography.
            p (float): Probability that a random point is an inlier.
            max_error (float): Maximum distance error to consider a point as an inlier.

        Returns:
            np.ndarray: Final 3x3 homography matrix.
            list: Best inliers (indices).
    """
    iterations = ceil(math.log(1 - P) / math.log(1 - p ** n_needed))
    best_inliers = []
    best_homography = None

    match_points = np.array(matches)
    x, y, xx, yy = match_points[:, 0], match_points[:, 1], match_points[:, 2], match_points[:, 3]
    ones = np.ones_like(x)

    minimal_inliners = int(0.1 * len(matches)) + 4
    best = [-1, 0]  # 0: nº de inliners; 1: index of inliners
    i = 0
    while i < iterations and minimal_inliners > best[0]:
        sample_indices = random.sample(range(len(matches)), n_needed)
        sample = match_points[sample_indices]

        H = sample_homography(sample)

        projected = np.dot(H, np.vstack((x, y, ones)))
        projected /= projected[2]

        projected_x, projected_y = projected[0], projected[1]
        errors = np.sqrt((xx - projected_x) ** 2 + (yy - projected_y) ** 2)

        inliers = np.where(errors < max_error)[0]

        if len(inliers) > best[0]:
            best[0] = len(inliers)
            best[1] = inliers.copy()

        i += 1

    if i == iterations: return False

    # inlier_points = match_points[best_inliers]
    inlier_points = match_points[best[1]]
    final_homography = sample_homography(inlier_points)

    return final_homography  # , best[1]


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

    def __repr__(self):
        return f"Parent: {self.parent_node}\nFifo: {self.fifo}\nOpened: {self.opened}"


class Image_Homographie():
    def __init__(self, H=None, parent=None, c=None, i=None, depth=0):
        self.H = H
        self.childs = []
        self.parent = parent
        self.camera = c  # camera
        self.image = i  # image
        self.depth = depth  # depth of the tree

    def create_child(self, H, c, i):
        child = Image_Homographie(H, self, c, i, self.depth + 1)
        self.childs.insert(0, child)

    def __repr__(self):
        return f'image {self.i} from camera {self.c} with depth {self.depth}, childs {len(self.childs)}'


def save_homographies(homographies, output_file):
    """
    Funkcja zapisująca homografie do pliku .mat.

    Args:
        homographies (list): Lista macierzy homografii (każda macierz ma wymiar 3x3).
        output_file (str): Nazwa pliku wyjściowego (np. "homographies.mat").
    """
    # Tworzymy tablicę NumPy o wymiarach (3, 3, Nv), gdzie Nv to liczba klatek
    H_array = np.stack(homographies, axis=2)  # Łączymy macierze w tablicę 3D

    # Tworzymy słownik z kluczem "H"
    H_dict = {"H": H_array}

    # Zapisujemy plik .mat
    savemat(output_file, H_dict)
    print(f"Homographies saved to: {output_file}")


if __name__ == "__main__":
    ref_dir, cameras = parse_arguments()
    print("Reference directory:", ref_dir)
    print("Camera pairs (input and output):", cameras)

    main_node = Image_Homographie(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    homographies = [{k: None for k in Cameras[c][0].keys()} for c in range(len(Cameras))]
    redo = []

    for camera, (ref_image, ref_kp, ref_desc, images, yolos, matches_data) in enumerate(load_data(ref_dir, cameras)):
        print("\nCamera: ", camera + 1)

        for image_index, image in images.items():

            kp_input = matches_data[image_index]['keypoints']
            desc_input = matches_data[image_index]['descriptors']

            hommographies_graph = breadth_first(main_node)
            n = hommographies_graph.next()
            result = False

            while n != None and type(result) == bool:

                if n.camera == None:
                    matches = match_keypoints_optimized(ref_desc, desc_input, ref_kp, kp_input)
                else:
                    descriptors, keypoints = matches_data[n.i]['descriptors'], matches_data[n.i]['keypoints']
                    matches = match_keypoints_optimized(desc_input, descriptors, kp_input, keypoints)

                result = compute_homography_with_RANSAC(matches)

                if type(result) == bool:
                    n = hommographies_graph.next()
                else:
                    n.create_child(result @ n.H, camera, image_index)

            if n == None:
                redo.append([camera, image_index, hommographies_graph.opened.copy(), 1000000])
            elif n.parent != None:
                redo.append([camera, image_index, hommographies_graph.opened.copy(), n.depth + 1])
                homographies[camera][image_index] = n.childs[-1].H
            else:
                homographies[camera][image_index] = n.childs[-1].H
            delta = time.time() - start
            print('Processing Time in %.3f s' % (delta))

    print('Redo')
    redo.reverse()
    for j in redo:

        camera, i, opened, depth = j
        print(f'Homographie with Cameras[{camera}]images[{i}]')
        hommographies_graph = breadth_first(main_node)
        hommographies_graph.update(opened)
        n = hommographies_graph.next()
        result = False

        while n != None and (n.depth + 1 >= depth or (n.c == camera and n.i == i)):
            print(f'\t{n.c} {n.i} {n.depth}')
            n = hommographies_graph.next()

        while n != None and type(result) == bool and n.depth + 1 < depth:
            result = compute_homography_with_RANSAC(camera, i, n.c, n.i)
            if type(result) == bool and result == False:
                n = hommographies_graph.next()
                if n != None and n.c == camera and n.i == i:
                    n = hommographies_graph.next()
            else:
                n.create_child(result @ n.H, camera, i)
        if type(result) != bool:
            homographies[camera][i] = n.childs[-1].H
        else:
            print('\tEnable to do')
    print(homographies)

    # ref_dir, cameras = parse_arguments()
    # print("Reference directory:", ref_dir)
    # print("Camera pairs (input and output):", cameras)
    #
    # for i, (ref_image, ref_kp, ref_desc, images, yolos, matches_data) in enumerate(load_data(ref_dir, cameras)):
    #     print("Camera: ", i + 1)
    #
    #     homographies = []
    #     for image_index, image in images.items():
    #         kp_input = matches_data[image_index]['keypoints']
    #         desc_input = matches_data[image_index]['descriptors']
    #
    #         matches = match_keypoints_optimized(ref_desc, desc_input, ref_kp, kp_input)
    #
    #         ransac_result = compute_homography_with_RANSAC(matches)
    #
    #         if ransac_result:
    #             homography, inliers = ransac_result
    #             homographies.append(homography)
    #             print(f"Found homography with {len(inliers)} inliers - frame {image_index}.")
    #         else:
    #             print(f"Homography not found for frame {image_index}.")
    #
    #     output_file = f"{cameras[i][1]}/homographies.mat"
    #     save_homographies(homographies, output_file)
    #     break
