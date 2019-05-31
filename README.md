# CV2_Assignment3

## Usage

- Download the [Basel Face dataset](https://faces.dmi.unibas.ch/bfm/bfm2017/restricted/model2017-1_face12_nomouth.h5) after applying for access [here](https://faces.dmi.unibas.ch/bfm/bfm2017.html) and put it in the repo directory.
```bash
# install dlib:
sudo apt-get install cmake libx11-dev
git clone https://github.com/davisking/dlib.git
cd dlib/
python setup.py install --no DLIB_USE_CUDA

# back in the repo:
pip install -r requirements.txt
python mesh_to_png.py
python morphable_model.py
python pinhole_camera.py
python ./face_landmark_detection.py ./shape_predictor_68_face_landmarks.dat ./pics
```
