"""
Transfer the expression of a video sequence with face (source video) to a target avatar.
"""
import os
import glob
import matplotlib.pyplot as plt
import cv2
import torch
from landmarks import detect_landmark
from latent_param_estimation import load_morphace, Model

def video_frames(cap):
    """yield a cv.VideoCapture's frames in an iterator. inspiration: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html"""
    while(cap.isOpened()):
        ret, frame = cap.read()
        yield frame
        if not ret:
            break
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

def estimate_model(frames, identity, expression, alpha=None):
    landmarks_pics = torch.stack(list(map(lambda f: torch.tensor(detect_landmark(f)), frames)))
    model = Model(landmarks_pics, identity, expression, alpha)
    return train_model(model)

if __name__ == "__main__":

    (texture, identity, expression, triangles) = load_morphace()
    # Capture 1 second video sequence with face changing expression from a neutral face.
    cap = cv2.VideoCapture('pics/fugu.mp4')
    frames = list(video_frames(cap))

    # Transfer expression into a previously obtained avatar:

    # - Estimate source identity parameters α using M frames;
    m = 10
    m_frames = frames[:m]
    model = estimate_model(m_frames, identity, expression)
    alpha = model.alpha

    # - Fix α for the full video sequence;
    # - ∀j = 1..J (for J frames) estimate δj, ωj, tj using energy minimization.
    #   Initializing parameters by parameters from previous frame may prevent the model to converge into local optima.
    model = estimate_model(frames, identity, expression, alpha)
    deltas = model.delta

    for i, delta in enumerate(deltas):
        # - Apply each δj on the target avatar
        G = reconstruct_face(identity, expression, alpha, delta)
        # TODO: use target avatar based on texturing part
        mesh = Mesh(G, texture.mean, triangles)

        # Report transferred textured results.
        img = mesh_to_png(mesh)
        plt.imshow(img)
        plt.savefig(f'results/video_{i}.png')

    # # Report transferred textured results.
    #     # (ground_truth, landmarks) = tpl
    #     plot_landmarks(tpl)
    #     plt.savefig(f'results/video_{i}.png')
