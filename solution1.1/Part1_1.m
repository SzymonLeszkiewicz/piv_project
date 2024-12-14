% Step 1: Load Keypoints from kp_gmaps.mat
data = load('kp_gmaps.mat'); % Load the .mat file
keypoints = data.kp_gmaps; % Access the keypoints matrix

% Step 2: Extract video and map points
videoPoints = keypoints(:, 1:2); % Points in the first video frame
mapPoints = keypoints(:, 3:4);   % Corresponding points in the Google Maps image

% Step 3: Compute the homography matrix
H = computeHomography(videoPoints, mapPoints);

% Step 4: Save the transformation matrix
save('output_00001.mat', 'H'); % Save the transformation matrix
disp('Homography matrix saved as output_00001.mat');

function H = computeHomography(pts1, pts2)
    % Ensure the inputs are Nx2 matrices
    assert(size(pts1, 2) == 2 && size(pts2, 2) == 2, ...
        'Input points must be Nx2 matrices');

    % Build the system of equations
    N = size(pts1, 1); % Number of point correspondences
    A = zeros(2 * N, 9); % Matrix to hold equations
    for i = 1:N
        x1 = pts1(i, 1);
        y1 = pts1(i, 2);
        x2 = pts2(i, 1);
        y2 = pts2(i, 2);
        A(2 * i - 1, :) = [-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2];
        A(2 * i, :)     = [0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2];
    end

    % Solve for H using SVD
    [~, ~, V] = svd(A);
    H = reshape(V(:, end), [3, 3])'; % Last column of V reshaped into 3x3
end
% Process each frame
videoFiles = dir('img_*.jpg');
for i = 1:length(videoFiles)
    frame = imread(videoFiles(i).name);

    % Warp the frame using the homography
    tform = projective2d(H');
    warpedFrame = imwarp(frame, tform, 'OutputView', imref2d(size(frame)));

    % Save transformed frame
    outputFile = sprintf('output_%05d.jpg', i);
    imwrite(warpedFrame, outputFile);

    % Save transformation matrix for this frame
    save(sprintf('output_%05d.mat', i), 'H');
end
% Process each YOLO file
yoloFiles = dir('yolo_*.mat');
for i = 1:length(yoloFiles)
    load(yoloFiles(i).name, 'xyxy', 'id'); % Load bounding boxes and IDs

    % Transform bounding box corners using H
    numBoxes = size(xyxy, 1);
    bbox = zeros(size(xyxy));
    for j = 1:numBoxes
        % Bottom-left corner
        pt1 = H * [xyxy(j, 1), xyxy(j, 2), 1]';
        pt1 = pt1 ./ pt1(3);

        % Top-right corner
        pt2 = H * [xyxy(j, 3), xyxy(j, 4), 1]';
        pt2 = pt2 ./ pt2(3);

        % Update bounding box
        bbox(j, :) = [pt1(1), pt1(2), pt2(1), pt2(2)];
    end

    % Save transformed bounding boxes
    save(sprintf('yolooutput_%05d.mat', i), 'bbox', 'id');
end
