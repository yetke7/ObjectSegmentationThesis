function out = helperSimAnnotMATReader(filename,datasetRoot)
% Read annotations for simulated bin picking dataset
% Expected folder structure under `datasetRoot`:
%       depth/      (depth images folder)
%       GT/         (ground truth MAT files)
%       image/      (color images)

    data = load(filename);
    groundTruthMaT = data.groundTruthMaT;
    clear data;
    
    % Read RGB image.
    tmpPath = strsplit(groundTruthMaT.RGBImagePath, '\');
    basePath = {tmpPath{4:end}};
    imRGBPath = fullfile(datasetRoot, basePath{:});
    im = imread(imRGBPath);
    im = rgb2gray(im);
    if(size(im,3)==1)
        im = repmat(im, [1 1 3]);
    end
    
    % Read depth image.
    aa = strsplit(groundTruthMaT.DepthImagePath, '\');
    bb = {aa{4:end}};
    imDepthPath = fullfile(datasetRoot,bb{:}); % handle windows paths
    imD = load(imDepthPath); imD = imD.depth;
    
    % For "undefined" value in instance labels, assign to the first class.
    undefinedSelect = isundefined(groundTruthMaT.instLabels);
    classNames = categories(groundTruthMaT.instLabels);
    groundTruthMaT.instLabels(undefinedSelect) = classNames{1};
    if isrow(groundTruthMaT.instLabels)
        groundTruthMaT.instLabels = groundTruthMaT.instLabels';
    end
    
    % Wrap the camera parameter matrix into the cameraIntrinsics object.
    K = groundTruthMaT.IntrinsicsMatrix;
    intrinsics = cameraIntrinsics([K(1,1) K(2,2)], [K(1,3)  K(2,3)], [size(im,1) size(im, 2)]);
    
    % Process rotation matrix annotations.
    rotationCell = num2cell(groundTruthMaT.rotationMatrix,[1 2]);
    rotationCell = squeeze(rotationCell);
    
    % Process translation annotations.
    translation = squeeze(groundTruthMaT.translation)';
    translationCell = num2cell(translation, 2);
    
    % Process poses into rigidtform3d vector - transpose R to obtain the
    % correct pose.
    poseCell = cellfun( @(R,t)(rigidtform3d(R',t)), rotationCell, translationCell, ...
        UniformOutput=false);
    pose = vertcat(poseCell{:});
    
    % Remove heavily occluded objects.
    if length(groundTruthMaT.occPercentage) == length(groundTruthMaT.instLabels)
        visibility = groundTruthMaT.occPercentage;
        visibleInstSelector = visibility > 0.5;
    else
        visibleInstSelector = true([length(groundTruthMaT.instLabels),1]);
    end
    
    out{1} = im;
    out{2} = imD;                                                    % HxWx1 double array depth-map
    out{3} = groundTruthMaT.instBBoxes(visibleInstSelector,:);       % Nx4 double bounding boxes
    out{4} = groundTruthMaT.instLabels(visibleInstSelector);         % Nx1 categorical object labels
    out{5} = logical(groundTruthMaT.instMasks(:,:,visibleInstSelector));      % HxWxN logical mask arrays
    out{6} = pose(visibleInstSelector);                              % Nx1 rigidtform3d vector of object poses
    out{7} = intrinsics;                                             % cameraIntrinsics object

end

function image = helperVisualizePosePrediction(...
    poses, labels, scores, boxes, modelClassNames, modelPointClouds, poseColors, imageOrig, intrinsics)
    image = imageOrig;
    numPreds = size(boxes,1);  
    detPosedPtClouds = cell(1,numPreds);
    for detIndex = 1:numPreds
        
        detClass = string(labels(detIndex));
        detTform = poses(detIndex);
        
        % Retrieve the 3-D object point cloud of the predicted object class.
        ptCloud = modelPointClouds(modelClassNames == detClass);
        
        % Transform the 3-D object point cloud using the predicted pose.
        ptCloudDet = pctransform(ptCloud, detTform);
        detPosedPtClouds{detIndex} = ptCloudDet;
    
        % Subsample the point cloud for cleaner visualization.
        ptCloudDet = pcdownsample(ptCloudDet,"random",0.05);
    
        % Project the transformed point cloud onto the image using the camera
        % intrinsic parameters and identity transform for camera pose and position.
        projectedPoints = world2img(ptCloudDet.Location,rigidtform3d,intrinsics);
        
        % Overlay the 2-D projected points over the image.helperVisualizeChamferDistance
        image = insertMarker(image,[projectedPoints(:,1), projectedPoints(:,2)],...
            "circle",Size=1,Color=poseColors(modelClassNames==detClass));
    end

    % Insert the annotations for the predicted bounding boxes, classes, and 
    % confidence scores into the image using the insertObjectAnnotation function.
    LabelScoreStr = compose("%s-%.2f",labels,scores); 
    image = insertObjectAnnotation(image,"rectangle",boxes,LabelScoreStr);   
end

function [distADDS,predIndices,gtIndices] = helperEvaluatePosePrediction(...
    modelPointClouds, modelClassNames,boxes,labels,pose, gBox,gLabel,gPose)

% Compare predicted and ground truth pose for a single image containing multiple 
% object instances using the one-sided Chamfer distance.

    function pointCloudADDS = pointCloudChamferDistance(ptCloudGT,ptCloudDet,numSubsampledPoints)
    % Return the one-sided Chamfer distance between two point clouds, which
    % computes the closest point in point cloud B for each point in point cloud A,
    % and averages over these minimum distances.

        % Sub-sample the point clouds                                                                                                                               
        if nargin == 2
            numSubsampledPoints = 1000;
        end
        
        rng("default"); % Ensure reproducibility in the point-cloud sub-sampling step.
        
        if numSubsampledPoints < ptCloudDet.Count
            subSampleFactor = numSubsampledPoints / ptCloudDet.Count;
            ptCloudDet = pcdownsample(ptCloudDet,"random",subSampleFactor);
            subSampleFactor = numSubsampledPoints / ptCloudGT.Count;   
            ptCloudGT = pcdownsample(ptCloudGT,"random",subSampleFactor);
        end
        
        % For each point in GT ptCloud, find the distance to closest point in predicted ptCloud.
        distPtCloud = pdist2(ptCloudGT.Location, ptCloudDet.Location,...
                "euclidean", "smallest",1);

        % Average over all points in GT ptCloud.
        pointCloudADDS = mean(distPtCloud); 
        
    end

    maxADDSThreshold = 0.1;

    % Associate predicted bboxes with ground truth annotations based on
    % bounding box overlaps as an initial step.
    minOverlap = 0.1;
    overlapRatio = bboxOverlapRatio(boxes,gBox);
    [predMatchScores, predGTIndices]  = max(overlapRatio, [], 2); % (numPreds x 1)
    [gtMatchScores, ~]  = max(overlapRatio, [], 1); % (1 x numGT)
    matchedPreds = predMatchScores > minOverlap; 
    matchedGTs = gtMatchScores > minOverlap;

    numPreds = size(boxes,1);

    distADDS = cell(numPreds,1);
    predIndices = cell(numPreds,1);
    gtIndices = cell(numPreds,1);

    for detIndex=1:numPreds
        detClass = string(labels(detIndex));

        % Account for predictions unmatched with GT (false positives).
        if ~matchedPreds(detIndex)
            % If the predicted bounding box does not overlap any
            % ground truth bounding box, then maximum penalty is applied 
            % and the point cloud matching steps are skipped.
            distADDS{detIndex} = maxADDSThreshold;
            predIndices{detIndex} = detIndex;
            gtIndices{detIndex} = 0;
        else
            % Match GT labels to Predicted objects by their bounding 
            % box overlap ratio (box Intersection-over-Union).
            gtIndex = predGTIndices(detIndex);
            detClassname = string(detClass);
            gClassname = string(gLabel(gtIndex));

            if detClassname ~= gClassname
                % If predicted object category is incorrec, set
                % to maximum allowed distance (highly penalized).
                distADDS{detIndex} = maxADDSThreshold;
            else
                % Predicted rotation and translation.                     
                detTform = pose(detIndex);

                % Ground truth pose.
                gTform = gPose(gtIndex);

                % Get the point cloud of the object.
                ptCloud = modelPointClouds(modelClassNames == string(gClassname));

                % Apply the ground truth pose transformation.
                ptCloudTformGT = pctransform(ptCloud, gTform);

                % Apply the predicted pose transformation
                ptCloudDet = pctransform(ptCloud, detTform);

                pointCloudADDSObj = pointCloudChamferDistance(...
                    ptCloudTformGT,ptCloudDet);

                distADDS{detIndex} = pointCloudADDSObj;
            end
            predIndices{detIndex} = detIndex;
            gtIndices{detIndex} = gtIndex;
        end                                      
    end

    distADDS = cat(1, distADDS{:});

    % Account for unmatched GT objects (false negatives).
    numUnmatchedGT = numel(matchedGTs) - nnz(matchedGTs);
    if numUnmatchedGT > 0
        % Set to max distance for unmatched GTs.
        falseNegativesADDS = maxADDSThreshold * ones(numUnmatchedGT,1); 
        fnPred = zeros(numUnmatchedGT,1);
        fnGT = find(~matchedGTs);
        distADDS = cat(1, distADDS, falseNegativesADDS);
        predIndices = cat(1, predIndices, fnPred);
        gtIndices = cat(1, gtIndices, num2cell(fnGT'));
    end

    predIndices = cat(1, predIndices{:});
    gtIndices = cat(1, gtIndices{:});

end

function [scenePtCloud,roiScenePtCloud] = helperPostProcessScene(imDepth,intrinsics,boxes,maxDepth,maxBinDistance,binOrientation)

    % Convert the depth image into an organized point cloud using camera
    % intrinsics.
    scenePtCloud = pcfromdepth(imDepth,1.0,intrinsics);
    
    % Remove outliers, or points that are too far away to be in the bin.
    selectionROI = [...
            scenePtCloud.XLimits(1) scenePtCloud.XLimits(2) ...
            scenePtCloud.YLimits(1) scenePtCloud.YLimits(2) ...
            scenePtCloud.ZLimits(1) maxDepth];
    selectedIndices = findPointsInROI(scenePtCloud, selectionROI);
    cleanScenePtCloud = select(scenePtCloud,selectedIndices);
    
    % Fit a plane to the bin surface.
    [~,~,outlierIndices] = pcfitplane(...
        cleanScenePtCloud,maxBinDistance,binOrientation);
    
    % Re-map indices back to the original scene point cloud. Use this
    % when cropping out object detections from the scene point cloud.
    origPtCloudSelection = selectedIndices(outlierIndices);
    
    % Crop predicted ROIs from the scene point cloud.
    numPreds = size(boxes,1);
    roiScenePtCloud = cell(1,numPreds);
    for detIndex=1:numPreds
        box2D = boxes(detIndex,:);
        
        % Get linear indices into the organized point cloud corresponding to the 
        % predicted 2-D bounding box of an object.
        boxIndices = (box2D(2):box2D(2)+box2D(4))' + (size(scenePtCloud.Location,1)*(box2D(1)-1:box2D(1)+box2D(3)-1));
        boxIndices = uint32(boxIndices(:));
        
        % Remove points that are outliers from earlier pre-processing steps
        % (either belonging to the bin surface or too far away).
        keptIndices = intersect(origPtCloudSelection,boxIndices);
        roiScenePtCloud{detIndex} = select(scenePtCloud,keptIndices);
    end

end

function fig = helperVisualizeChamferDistance(...
    labels, predIndices, gtIndices, modelPointClouds, ...
    modelClassNames, gtClasses, poses, gtPoses, distances)
    fig = figure;

    for idx = 1:numel(predIndices)
        detIndex = predIndices(idx);

        if detIndex == 0
            % The ground truth bounding box does not match any predicted
            % bounding boxes (false negative)
            ptCloudTformDet = pointCloud(single.empty(0, 3));
        else
            detClass = string(labels(detIndex));
            gtIndex = gtIndices(idx);
    
            % Obtain the point cloud of the predicted object.
            ptCloudDet = modelPointClouds(modelClassNames == detClass);
    
            % Predicted 6-DoF pose with ICP refinement.
            detTform = poses(detIndex);
    
            % Apply the predicted pose transformation to the predicted object point
            % cloud.
            ptCloudTformDet = pctransform(ptCloudDet, detTform);
        end

        if gtIndex == 0
            % The predicted bounding box does not match any
            % ground truth bounding box (false positive).
            ptCloudTformGT = pointCloud(single.empty(0,3));
        else
            % Obtain the point cloud of the ground truth object.
            ptCloudGT = modelPointClouds(modelClassNames == string(gtClasses(gtIndex)));

            % Apply the ground truth pose transformation.
            ptCloudTformGT = pctransform(ptCloudGT,gtPoses(gtIndex));
        end

        subplot(2,4,gtIndex);
        pcshowpair(ptCloudTformDet,ptCloudTformGT);
        title(sprintf("d = %.4f",distances(idx)))
    end
end


function occGrid = voxelizePointCloud(points, voxelSize)
    % points: Nx3 (double) matrix
    % voxelSize: scalar (e.g. 0.005 meters)

    if isempty(points)
        occGrid = false(1,1,1);
        return
    end

    % Translate points to positive index space
    minPt = floor(min(points,[],1) / voxelSize);
    coords = floor(points / voxelSize) - minPt + 1;

    % Get grid size
    gridDims = max(coords,[],1);

    % Convert 3D coords to linear indices
    linIdx = sub2ind(gridDims, coords(:,1), coords(:,2), coords(:,3));

    % Create binary occupancy grid
    occGrid = false(gridDims);
    occGrid(linIdx) = true;
end




rng("default");

datasetDir = fullfile(datasetUnzipFolder,"pvcparts100");
gtLocation = fullfile(datasetDir,"GT");

dsRandom = fileDatastore(gtLocation, ...
    ReadFcn=@(x)helperSimAnnotMATReader(x,datasetDir));

randIndices = randperm(length(dsRandom.Files));
numTrainRandom = round(0.7*length(dsRandom.Files)); 
dsTrain = subset(dsRandom,randIndices(1:numTrainRandom));
dsVal = subset(dsRandom,randIndices(numTrainRandom+1:end));

data = preview(dsVal);
imRGB = data{1};
imDepth = data{2};
gtMasks = data{5};
gtClasses = data{4};
gtBoxes = data{3};
gtPoses = data{6};
intrinsics = data{7};

trainClassNames = categories(gtClasses)

imRGB = imread("/tmp/dataset/pvcparts100/image/00001.png");
load("/tmp/dataset/pvcparts100/depth/00001.mat", "depth");
load("/tmp/dataset/pvcparts100/GT/00001.mat");

% Convert depth to imDepth
imDepth = depth;

% Unpack groundTruthMaT
gtMasks = groundTruthMaT.instMasks;
gtBoxes = groundTruthMaT.instBBoxes;
gtClasses = groundTruthMaT.instLabels;
gtPoses = [];

% Wrap rotation and translation into rigidtform3d
for i = 1:size(groundTruthMaT.rotationMatrix,3)
    R = groundTruthMaT.rotationMatrix(:,:,i)';
    t = groundTruthMaT.translation(:,i)';
    gtPoses = [gtPoses; rigidtform3d(R, t)];
end

% Get camera intrinsics
K = groundTruthMaT.IntrinsicsMatrix;
intrinsics = cameraIntrinsics([K(1,1) K(2,2)], [K(1,3) K(2,3)], [size(imRGB,1), size(imRGB,2)]);

% Optional: limit to visible objects only
if isfield(groundTruthMaT, "occPercentage")
    visible = groundTruthMaT.occPercentage > 0.5;
    gtMasks = gtMasks(:,:,visible);
    gtBoxes = gtBoxes(visible,:);
    gtClasses = gtClasses(visible);
    gtPoses = gtPoses(visible);
end

modelFiles = ["/tmp/dataset/pvcparts100/pcdmodels/I_shape.pcd", "/tmp/dataset/pvcparts100/pcdmodels/L_shape.pcd", "/tmp/dataset/pvcparts100/pcdmodels/T_shape.pcd", "/tmp/dataset/pvcparts100/pcdmodels/X_shape.pcd"];
modelClassNames = ["I_shape", "L_shape", "T_shape", "X_shape"];
modelPointClouds = pointCloud.empty();

for i = 1:length(modelFiles)
    % Read mesh and sample to point cloud
    ptCloud = pcread(modelFiles(i)); % If already a point cloud
    % Alternatively, sample from mesh using 'stlread' and 'pcdownsample' if needed
    modelPointClouds(i) = ptCloud;
end

save("pointCloudModels.mat", "modelClassNames", "modelPointClouds");
load("pointCloudModels.mat","modelClassNames","modelPointClouds")


figure
tiledlayout(2,2,'TileSpacing','Compact')
for i = 1:length(modelPointClouds)
    nexttile
    ax = pcshow(modelPointClouds(i));
    title(modelClassNames(i), Interpreter='none')
end
sgtitle("Reference Objects", Color='w');



figure
imshow(imRGB);
title("RGB Image")

figure
imshow(depth);
title("Depth Image")

imRGBAnnot = insertObjectMask(imRGB,gtMasks,Opacity=0.5);
imRGBAnnot = insertObjectAnnotation(imRGBAnnot,"rectangle",gtBoxes,gtClasses);


figure
imshow(imRGBAnnot); 
title("Ground Truth Annotations")

poseColors = ["blue","green","magenta","cyan"];
numPoseColors = length(poseColors);
numObj = size(gtBoxes,1);  
imGTPose = imRGB;

gtPredsImg = helperVisualizePosePrediction(gtPoses, ...
    gtClasses, ...
    ones(size(gtClasses)), ...
    gtBoxes, ...
    modelClassNames, ...
    modelPointClouds, ...
    poseColors, ...
    imRGB, ...
    intrinsics);

figure
imshow(gtPredsImg);
title("Ground Truth Point Clouds")

pretrainedNet = posemaskrcnn("resnet50-pvc-parts");

[poses,labels,scores,boxes,masks] = predictPose(pretrainedNet, ...
                imRGB,imDepth,intrinsics,Threshold=0.5, ...
                ExecutionEnvironment="auto");

% === Evaluate Segmentation Quality (2D Masks) ===
try
    % Combine instance masks into single binary masks
    gtSegMask = any(gtMasks, 3);         % HxW logical
    predSegMask = any(masks, 3);         % HxW logical

    % Resize if shapes mismatch
    if ~isequal(size(gtSegMask), size(predSegMask))
        predSegMask = imresize(predSegMask, size(gtSegMask), 'nearest');
    end

    % Flatten
    gtFlat = gtSegMask(:);
    predFlat = predSegMask(:);

    % Safety: ensure logical
    gtFlat = logical(gtFlat);
    predFlat = logical(predFlat);

    % Confusion matrix terms
    TP = sum(gtFlat & predFlat);
    TN = sum(~gtFlat & ~predFlat);
    FP = sum(~gtFlat & predFlat);
    FN = sum(gtFlat & ~predFlat);

    % Metrics
    accuracy  = (TP + TN) / (TP + TN + FP + FN + eps);
    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);
    f1        = 2 * (precision * recall) / (precision + recall + eps);
    iou       = TP / (TP + FP + FN + eps);

    % Print results
    fprintf('\n=== Segmentation Evaluation (2D RGB Mask) ===\n');
    fprintf('Accuracy : %.4f\n', accuracy);
    fprintf('Precision: %.4f\n', precision);
    fprintf('Recall   : %.4f\n', recall);
    fprintf('F1 Score : %.4f\n', f1);
    fprintf('IoU      : %.4f\n', iou);
catch ME
    fprintf('[ERROR] Evaluation failed: %s\n', ME.message);
end



pretrainedPredsImg = helperVisualizePosePrediction(poses, ...
    labels, ...
    scores, ...
    boxes, ...
    modelClassNames, ...
    modelPointClouds, ...
    poseColors, ...
    imRGB, ...
    intrinsics);

figure;
imshow(pretrainedPredsImg);
title("Pose Mask R-CNN Prediction Results")

maxDepth = 0.5;
maxBinDistance = 0.0015;
binOrientation = [0 0 -1];

[scenePointCloud,roiScenePtCloud] = helperPostProcessScene(depth,intrinsics,boxes,maxDepth, ...
    maxBinDistance,binOrientation);

figure;
pcshow(scenePointCloud,VerticalAxisDir="Down")
title("Scene Point Cloud from Depth Image")

figure;
pcshow(roiScenePtCloud{end}, VerticalAxisDir="Down")
title("Object Point Cloud from Depth Image")

downsampleFactor = 0.25;
numPreds = size(boxes,1); 
registeredPoses = cell(numPreds,1);
for detIndex = 1:numPreds
    detClass = string(labels(detIndex));

    % Define predicted rotation and translation.
    detTform = poses(detIndex);
    detScore = scores(detIndex);
    
    % Retrieve the 3-D object point cloud of the predicted object class.
    ptCloud = modelPointClouds(modelClassNames == detClass);
    
    % Transform the 3-D object point cloud using the predicted pose.
    ptCloudDet = pctransform(ptCloud, detTform);
    
    % Downsample the object point cloud transformed by the predicted pose.
    ptCloudDet = pcdownsample(ptCloudDet,"random",downsampleFactor);

    % Downsample point cloud obtained from the postprocessed scene depth image.
    ptCloudDepth =  pcdownsample(roiScenePtCloud{detIndex},"random",downsampleFactor);
    
    % Run the ICP point cloud registration algorithm with default
    % parameters.
    [tform,movingReg] = pcregistericp(ptCloudDet,ptCloudDepth);
    registeredPoses{detIndex} = tform;
end

figure
subplot(1,2,1)
pcshowpair(ptCloudDet,ptCloudDepth)
title("Predicted Pose and Depth Image Point Clouds")

subplot(1,2,2)
pcshowpair(movingReg,ptCloudDepth)
title("ICP Registration Result")

refinedPoses = cell(1,numPreds);

for detIndex = 1:numPreds 

    detClass = string(labels(detIndex));

    % Define predicted rotation and translation.
    detTform = poses(detIndex);
    detScore = scores(detIndex);

    % Rotation and translation from registration of depth point clouds.
    icpTform = registeredPoses{detIndex};

    % Combine the two transforms to return the final pose.
    combinedTform = rigidtform3d(icpTform.A*detTform.A);
    refinedPoses{detIndex} = combinedTform;

end
refinedPoses = cat(1,refinedPoses{:});

imPoseRefined = helperVisualizePosePrediction(refinedPoses, ...
    labels, ...
    scores, ...
    boxes, ...
    modelClassNames, ...
    modelPointClouds, ...
    poseColors, ...
    imRGB, ...
    intrinsics);


% === Voxel-based Evaluation of ICP-refined Point Clouds ===
gridSize = 0.005; % 5mm voxel grid
TP = 0; FP = 0; FN = 0;

fprintf("\n=== Point Cloud Voxel Comparison ===\n");

for i = 1:numel(refinedPoses)
    detClass = string(labels(i));
    predPose = refinedPoses(i);
    gtIdx = find(gtClasses == detClass, 1); % Match GT by class

    if isempty(gtIdx)
        fprintf("No GT match for class %s\n", detClass);
        continue
    end

    gtPose = gtPoses(gtIdx);

    % Get model point cloud
    model = modelPointClouds(modelClassNames == detClass);

    % Apply transformations
    pcPred = pctransform(model, predPose);
    pcGT   = pctransform(model, gtPose);

    % Voxelize
    occPred = voxelizePointCloud(pcPred.Location, gridSize);
    occGT   = voxelizePointCloud(pcGT.Location, gridSize);

    % Convert to logical occupancy
    predMask = occPred > 0;
    gtMask   = occGT > 0;

    % Pad arrays to same size
    sz = max([size(predMask); size(gtMask)], [], 1);
    predMask(sz(1),sz(2),sz(3)) = 0;
    gtMask(sz(1),sz(2),sz(3))   = 0;

    % Compute metrics
    TP = TP + nnz(predMask & gtMask);
    FP = FP + nnz(predMask & ~gtMask);
    FN = FN + nnz(~predMask & gtMask);
end

% Final metric calculations
precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
accuracy  = TP / (TP + FP + FN + eps);
f1        = 2 * precision * recall / (precision + recall + eps);
iou       = TP / (TP + FP + FN + eps);  % <<< IoU added here

% Display results
fprintf("Precision: %.4f\n", precision);
fprintf("Recall   : %.4f\n", recall);
fprintf("Accuracy : %.4f\n", accuracy);
fprintf("F1 Score : %.4f\n", f1);
fprintf("IoU      : %.4f\n", iou);



figure
imshow(imPoseRefined);
title("Pose Mask R-CNN + ICP")

[distPtCloud,predIndices,gtIndices] = helperEvaluatePosePrediction( ...
    modelPointClouds,modelClassNames,boxes,labels,refinedPoses,gtBoxes,gtClasses,gtPoses);


helperVisualizeChamferDistance(...
    labels, predIndices, gtIndices, modelPointClouds, ...
    modelClassNames, gtClasses, refinedPoses, gtPoses, distPtCloud);

avgChamferDistance = mean(distPtCloud(:));
sgtitle(["Pose Mask R-CNN Chamfer Distances" "Mean " + num2str(avgChamferDistance)], Color='w');

