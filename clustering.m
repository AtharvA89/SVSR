clc;
clear;
close all;

% Read image
he = imread('img.jpeg');

figure;
imshow(he);
title('Original Image');

% Number of clusters
numcolors = 3;

% K-means segmentation
L = imsegkmeans(he, numcolors);

% Overlay segmented image
B = labeloverlay(he, L);

figure;
imshow(B);
title('K-means Segmented Image');

% Convert RGB to LAB
lab_he = rgb2lab(he);

% Use only a*b channels
ab = lab_he(:,:,2:3);

ab = im2single(ab);

% Segment again
pixel_labels = imsegkmeans(ab, numcolors, NumAttempts=3);

% Overlay final result
B2 = labeloverlay(he, pixel_labels);

figure;
imshow(B2);
title('Labeled image a*b*');