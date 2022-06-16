function tunningCurve = checkTunCurve2(basis,range)
winsize = 10;
Basis_num = size(basis,2);
rds = gen_rds2(winsize,range);
nTestSet = size(rds,3);
%
tunningCurve = zeros(2*range+1,Basis_num);
for j = 1:nTestSet
    tunningCurve = tunningCurve + abs(rds(:,:,j)'*basis);
end
tunningCurve = tunningCurve/nTestSet;
% top = sum((tunningCurve - tunningCurve(end:-1:1,:)).^2);
% bot = sum((tunningCurve + tunningCurve(end:-1:1,:)).^2);
% asymIndex = mean(top./bot);

disp('Finish checking tunning curve!');
end