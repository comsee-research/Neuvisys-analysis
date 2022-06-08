function [theta_table,phase_table,lambda_table] = gaborFit(basis)

num_basis = size(basis,2);
[xi,yi] = meshgrid(1:10,1:10);
TrueBasis = zeros(200,num_basis);
EstBasis = zeros(200,num_basis);
mu_table = zeros(2,num_basis);
sigma_table = zeros(2,num_basis);
theta_table = zeros(2,num_basis);
lambda_table = zeros(1,num_basis);
phase_table = zeros(2,num_basis);
error_table = zeros(1,num_basis);
for i=1:num_basis
    %disp(['Iteration:' num2str(i)]);
    b = zeros(10,10,2);
    b1 = basis(:,i);
    b(:,:,1) = reshape(b1(1:100),[10,10]);
    b(:,:,2) = reshape(b1(101:200),[10,10]);    
    %
    res = autoGaborSurf(xi,yi,b);
    %
    est_b1 = reshape(res.G1,[100 1]);
    est_b2 = reshape(res.G2,[100 1]);    
    est_b = [est_b1;est_b2];
    EstBasis(:,i) = est_b;
    TrueBasis(:,i) = b1;
    % save results
    mu_table(:,i) = [res.x0;res.y0];
    sigma_table(:,i) = [res.sigmax;res.sigmay];
    theta_table(:,i) = [res.theta1;res.theta2];
    lambda_table(i) = res.lambda;
    phase_table(:,i) = [res.phase1;res.phase2];
    error_table(i) = res.sse;
end
% figure(1);
% drawBasis(TrueBasis,20,10);
% figure(2);
% drawBasis(EstBasis,20,10);
end