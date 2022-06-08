clear all;close all;clc
addpath('Funcs Gabor Fit');
bfilename = 'examplebasis_ZQP';
load_filename = [bfilename '.mat'];
load(load_filename);
gaborFit(bases(:,1:16));

%% draw basis
drawBasis(bases,20,10);