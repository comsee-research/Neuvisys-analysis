clear all;close all;clc

addpath('Funcs Gabor Fit');
folder = '/home/alphat/neuvisys-dv/configuration/network/gabors/data/right/';
load_filename = [folder 'weights_right.mat'];
load(load_filename);

[mu_table, sigma_table, theta_table, phase_table, lambda_table, error_table] = gaborFit_onepatch(data, folder);
save([folder 'mu.mat'], 'mu_table');
save([folder 'sigma.mat'], 'sigma_table');
save([folder 'lambda.mat'], 'lambda_table');
save([folder 'phase.mat'], 'phase_table');
save([folder 'theta.mat'], 'theta_table');
save([folder 'error.mat'], 'error_table');

%% draw basis
%drawBasis(data(:, 1:256), 20, 10);