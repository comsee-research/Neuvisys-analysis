function [TrueBasis,EstBasis,mu_table,sigma_table,theta_table,lambda_table,phase_table,error_table] = gaborFit_patch1by1(basis1,basis2)

num_basis = size(basis1,2);
basisdim = size(basis1,1);
onepatchdim = 100;
patchsize = sqrt(onepatchdim);
patchnum = basisdim/onepatchdim;
[xi,yi] = meshgrid(1:patchsize,1:patchsize);
TrueBasis = zeros(basisdim*2,num_basis);
EstBasis = zeros(basisdim*2,num_basis);
mu_table = zeros(2,patchnum*2,num_basis);
sigma_table = zeros(2,patchnum*2,num_basis);
theta_table = zeros(patchnum*2,num_basis);
lambda_table = zeros(patchnum*2,num_basis);
phase_table = zeros(patchnum*2,num_basis);
error_table = zeros(patchnum*2,num_basis);
for basisi=1:num_basis
    disp(['Basis:' num2str(basisi)]);
    
    for patchi=1:patchnum
        %%% The first basis functions in the subspace
        b = basis1(((patchi-1)*onepatchdim+1):(patchi*onepatchdim),basisi);

        res = autoGaborSurf_Onepatch(xi,yi,b);

        est_b = reshape(res.G,[onepatchdim 1]);
        EstBasis(((patchi-1)*onepatchdim+1):(patchi*onepatchdim),basisi) = est_b;
        TrueBasis(((patchi-1)*onepatchdim+1):(patchi*onepatchdim),basisi) = b;
        % save results
        mu_table(:,patchi,basisi) = [res.x0;res.y0];
        sigma_table(:,patchi,basisi) = [res.sigmax;res.sigmay];
        theta_table(patchi,basisi) = res.theta;
        lambda_table(patchi,basisi) = res.lambda;
        phase_table(patchi,basisi) = res.phase;
        error_table(patchi,basisi) = res.sse;
        
        %%% The second basis functions in the subspace
        b = basis2(((patchi-1)*onepatchdim+1):(patchi*onepatchdim),basisi);

        res = autoGaborSurf_Onepatch(xi,yi,b);

        est_b = reshape(res.G,[onepatchdim 1]);
        EstBasis(((patchi-1)*onepatchdim+basisdim+1):(patchi*onepatchdim+basisdim),basisi) = est_b;
        TrueBasis(((patchi-1)*onepatchdim+basisdim+1):(patchi*onepatchdim+basisdim),basisi) = b;
        % save results
        mu_table(:,patchnum+patchi,basisi) = [res.x0;res.y0];
        sigma_table(:,patchnum+patchi,basisi) = [res.sigmax;res.sigmay];
        theta_table(patchnum+patchi,basisi) = res.theta;
        lambda_table(patchnum+patchi,basisi) = res.lambda;
        phase_table(patchnum+patchi,basisi) = res.phase;
        error_table(patchnum+patchi,basisi) = res.sse;
    end
end
% figure(1);
% drawBasis(TrueBasis,20,10);
% figure(2);
% drawBasis(EstBasis,20,10);
end