import sys
import os
import dlib
import glob
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    # return the list of (x, y)-coordinates
    return coords

def detect_landmark(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    # print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #     k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        # Draw the face landmarks on the screen.
        # apparently all landmarks displayed in reverse (over both axes) in our images, so flip them!
        # return shape_to_np(shape)
        return -1 * shape_to_np(shape)

    return np.array([])

def file_landmarks(f):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    landmarks = detect_landmark(img)
    return landmarks

def rescale_landmarks(landmarks):
    landmarks = landmarks.astype(float)
    for i in [0, 1]:
        landmarks[:, i] = landmarks[:, i]
        difference = landmarks[:, i].max() - landmarks[:, i].min()
        landmarks[:, i] = landmarks[:, i] - landmarks[:, i].mean()
        landmarks[:, i] = landmarks[:, i] / difference
    return landmarks

def flipper_upper(landmarks):
    landmarks *= -1
    return landmarks

def plot_landmarks(data):
    """Visualize predicted landmarks overlayed on ground truth"""
    labels = ["ground truth", "model"]
    for i in range(len(data)):
        data[i] = rescale_landmarks(data[i])
        # data[i] = flipper_upper(data[i])
        plt.scatter(data[i][:, 0], data[i][:, 1], label=labels[i])
        for j in range(len(data[i])):
            plt.text(data[i][j, 0], data[i][j, 1], j)
    plt.legend()
    plt.show()

def files_landmarks():
    # TODO: convert to memoized function
    pickled_file = Path("landmarks.pkl")
    if not pickled_file.exists():
        faces_folder_path = 'pics'
        files = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
        data = list(map(file_landmarks, tqdm(files)))
        with open(pickled_file, 'wb') as f:
            pickle.dump(data, f)
    else:
        file_stream = open(pickled_file, 'rb')
        data = pickle.load(file_stream)
        print(f'Loaded data \'{pickled_file}\'.')
    return data

def main(path):
    im = dlib.load_rgb_image(path)
    landmarks = -detect_landmark(im)
    plt.figure()
    plt.imshow(im)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=8, c='r')
    plt.show()
    return

if __name__ == "__main__":
    # Take a single picture of yourself or pick random one from the web.
    # Extract ground truth landmarks using Dlib (http://dlib.net/face_landmark_detection.py.html).
    # Keep face closer to the frontal and neutral for now.
    if len(sys.argv) < 2:
        data = files_landmarks()
        print(data)
        plot_landmarks(data)
        plt.savefig('results/landmarks.png')
    else:
        main(sys.argv[1])
