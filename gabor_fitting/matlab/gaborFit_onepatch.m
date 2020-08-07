function [mu_table, sigma_table, theta_table, phase_table, lambda_table, error_table] = gaborFit_onepatch(basis)

disp("computing gabor basis");
num_basis = size(basis, 2);

[xi, yi] = meshgrid(1:10, 1:10);
TrueBasis = zeros(100, num_basis);
EstBasis = zeros(100, num_basis);
%parameters to estimate
mu_table = zeros(2, num_basis);
sigma_table = zeros(2, num_basis);
theta_table = zeros(1, num_basis);
lambda_table = zeros(1, num_basis);
phase_table = zeros(1, num_basis);
error_table = zeros(1, num_basis);
%for i=1:num_basis
for i=1:num_basis
    disp(['Iteration:' num2str(i)]);
    b = zeros(10, 10);
    b1 = basis(1:100, i);
    b(:, :) = reshape(b1(1:100), [10, 10]);
   
    %
    res = autoGaborSurf_Onepatch(xi, yi, b);
    %
    est_b = reshape(res.G, [100 1]);
   

    EstBasis(:, i) = est_b;
    TrueBasis(:, i) = b1;
    % save results
    mu_table(:, i) = [res.x0; res.y0];
    sigma_table(:, i) = [res.sigmax; res.sigmay];
    theta_table(:, i) = [res.theta];
    lambda_table(i) = res.lambda;
    phase_table(:, i) = [res.phase];
    error_table(i) = res.sse;    

end
figure(1);
%drawBasis_v2(TrueBasis, EstBasis, 20, 10);
save("mat/TrueBasis.mat", "TrueBasis");
save("mat/EstBasis.mat", "EstBasis");
end
