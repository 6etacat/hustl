import numpy as np
import matlab.engine

print("starting matlab")
eng = matlab.engine.start_matlab()
print("matlab engine started")
eng.eval("addpath('/home/michael/projects/hustl/hustl/matlab');")
eng.eval("addpath('/home/michael/projects/hustl/hustl/matlab/mesh');")
eng.eval("addpath('/home/michael/projects/hustl/hustl/matlab/RANSAC');")
print('path added')

nFrames = 3000

TracksPerFrame = 3000           # number of trajectories in a frame, 200 - 2000 is OK

MeshSize = 8                  # The mesh size of bundled camera path, 6 - 12 is OK
Smoothness = 3                 # Adjust how stable the output is, 0.5 - 3 is OK
Span = 5                      # Omega_t the window span of smoothing camera path, usually set it equal to framerate
Cropping = 0.8                   # adjust how similar the result to the original video, usually set to 1
Rigidity = 4                   # adjust the rigidity of the output mesh, consider set it larger if distortion is too significant, [1 - 4]
iteration = 50                 # number of iterations when optimizing the camera path[10 - 20]

OutputPadding = 800            # the padding around the video, should be large enough. 

inputDir = "'part_res/'"
outputDir = "'data/result_arch/'"

print("Track points")
track = eng.eval(f"GetTracks({inputDir}, {MeshSize}, {TracksPerFrame}, {nFrames});")
print("\nfinished tracking points")
eng.workspace["track"] = track
print("assigned to workspace")

print("Calculate path homography")
path = eng.eval(f"getPath({MeshSize}, track);")
print("\nfinished calculating path")
eng.workspace["path"] = path
print("assigned to workspace")

print("Calculate Bundled Paths")
bundled = eng.eval(f"Bundled({inputDir}, path, {Span}, {Smoothness}, {Cropping}, {Rigidity});")
print("finished calculating path")
eng.workspace["bundled"] = bundled
print("assigned to workspace")
print("Optimize Bundled Path")
# eng.eval(f"bundled.optPath({iteration});")
eng.optPath(bundled, iteration, nargout=0)
print("finished optimizing Bundled Path")

eng.eval(f"bundled.render({outputDir}, {OutputPadding});", nargout=0)
