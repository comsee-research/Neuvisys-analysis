clear all;close all;clc
addpath('Funcs Gabor Fit');
bfilename = 'weights';
load_filename = [bfilename '.mat'];
load(load_filename);

[mu_table, sigma_table, theta_table, phase_table, lambda_table, error_table] = gaborFit_onepatch(data);
save("mat/mu.mat", "mu_table");
save("mat/sigma.mat", "sigma_table");
save("mat/lambda.mat", "lambda_table");
save("mat/phase.mat", "phase_table");
save("mat/theta.mat", "theta_table");
save("mat/error.mat", "error_table");

%% draw basis
%drawBasis(data(:, 1:256), 20, 10);