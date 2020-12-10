function drawErrorAndLocality(errors,sigmas,lambdas)
[error_table,idx] = sort(errors);
sigma_table = sigmas(:,idx);
lambda_table = lambdas(:,idx);
locality_table = abs(sigma_table(1,:))./lambda_table;
%
figure;
X = [error_table;locality_table];
hist3(X',{0:0.1:0.7 0:0.2:4});
zlim([0 40]);
xlabel('fit error');
ylabel('locality');
zlabel('number of basis');
set(gcf,'renderer','opengl');
set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
end