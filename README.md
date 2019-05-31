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

## Notes
- Section 3, Equation 2: [\hat{x}, \hat{y}, \hat{z}, \hat{d}] are homogeneous coordinates obtained after projection. You can remove homogeneous coordinate by dividing by \hat{d} and get u, v and depth respectively. You can check SfM lecture for more details about camera projections.
- For projection matrix you can set principal point to be in the center of an image {W/2, H/2} and fovy to be 0.5.
- There is a way to see directly 3D geometry. You can use provided fio.save_obj from fio.py to export geometry to an obj file. Obj file can be viewed using meshlab (http://www.meshlab.net/ ) or blender (https://www.blender.org/).
- Landmarks are a subset of vertices from the morphable model (indexes are defined by the annotation file provided), that's why you are inferring landmarks.
- You may be wondering why do we always have to set triangles. Triangles connect define a set of faces (set of 3 vertices) which should be connected together to form a surface. Together with vertices they forms a mesh. You can check here for more information https://en.wikipedia.org/wiki/Polygon_mesh
- Section 4, Equation 4: p_{kid}_j should be p_{k}_j
- Section 5: You are required only to texture using single frame. It will work the best if human face is frontal and neutral. Everything else will be extra.
- Section 2: Remember that you are loading variance from BFM weights and you need \sigma in the equation
- Section 3, Q2: We are asking you to plot landmarks on 2D plane, not on a 3D point cloud. You can simply create an image WxH and visualize 2D points after applying projection.
- Section 3: Your camera origin is at (0, 0, 0), camera view direction is (0, 0, -1). Consequently, 3D model is initially behind the camera, therefore remember to shift an object using z translation from section 4.
- Section 2: Geometry from equation (1) is just a point cloud, you can directly visualize using matplotlib 3D scatterplot, for example, for debugging purpose.
- Section 4: Your energy optimization problem is non-convex. If your initial estimation is too far away from local optima, your solution may not be optimal. Code from (Section 3, Q2) can be used to see how far you are from the original face.
- Section 5: Bilinear interpolation is required because your 2D projections are real numbers, which means that each 2D projection is situated between 4 discrete pixels.
- H and W from the viewport matrix are not estimated from the point cloud. They are input image height and width.
- To convert homogeneous coordinate back to obtain u,v coordinates you just need to divide by a homogeneous coordinate, no division by depth is required.
- No batch norm should be applied in the assignment. We don't model scaling parameter in the assignment. If it doesn't work without batch norm, most likely you have some issue with implementation.
- Test case to debug your projection matrix test.png and an expected overlayed point cloud projection debug0000.png . Rotation parameters are {0, 0, 0}, translation parameters are {0, 0, -400}.
