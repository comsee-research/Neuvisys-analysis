function histFit(rfilename,bfilename)
load_rfilename = [rfilename '.mat'];
load(load_rfilename);
load_bfilename = [bfilename '.mat'];
load(load_bfilename);
% index = [1:5,11,53:58,139:144,232:237];
% sub_basis = basis(:,index);
% drawBasis(sub_basis,20,10);
% EstBasis, phase_table, lambda_table, theta_table, sigma_table
tuningCurve = checkTunCurve2(basis,10);
% computed disparity
disparity = drawHistDisparity(phase_table(1,:),phase_table(2,:),lambda_table,theta_table);

drawHistOrientDisparity(disparity,theta_table);

% reordering
[theta_table,idx] = sort(theta_table,'ascend');
disparity = disparity(:,idx);
tuningCurve = tuningCurve(:,idx);
error_table = error_table(idx);
%
drawErrorOrient(theta_table,tuningCurve,disparity);
%
estBasis = EstBasis(:,idx);
trueBasis = basis(:,idx);
%
tuningCurve = [tuningCurve zeros(21,24)];
disparity = [disparity zeros(2,24)];
orientaion = [theta_table, zeros(1,24)];
errors = [error_table,zeros(1,24)];
exci_table = reshape(disparity(1,1:324),[18 18]);
inhi_table = reshape(disparity(2,1:324),[18 18]);
orient_table = reshape(orientaion,[18 18]);
err_table = reshape(errors,[18 18]);
drawTunCurve(tuningCurve,exci_table',inhi_table',orient_table',err_table');
% drawErrorAndLocality(error_table,sigma_table,lambda_table);
drawBasis(trueBasis,20,10);
drawBasis(estBasis,20,10);
drawHistOrient(theta_table);
end