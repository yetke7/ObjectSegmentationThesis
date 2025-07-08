example_depth_image_original = imread("/home/yusuf/DeepPicking/IPA_data/IPAGearShaft_oneDrop/p_depth/cycle_0000/002_depth_uint16.png/")
size(example_depth_image_original)
%rbg2 = imresize(example_depth_image,1)

focalLength      = [500, 500];
principalPoint   = [100, 100];
imageSize        = size(example_depth_image_original);
intrinsics       = cameraIntrinsics(focalLength,principalPoint,imageSize);
depthScaleFactor = 5e3;
maxCameraDepth   = 5;

ptCloud = pcfromdepth(example_depth_image_original,depthScaleFactor, intrinsics,DepthRange=[0 Inf]);

ptCloud()
pcshow(ptCloud)
pcwrite(ptCloud,"IPAGearShaft_oneDrop_p_depth_cycle_0000_002",PLYFormat="binary");

