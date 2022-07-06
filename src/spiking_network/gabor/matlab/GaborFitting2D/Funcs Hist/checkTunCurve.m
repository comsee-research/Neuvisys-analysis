function tunningCurve = checkTunCurve(basis,range)
rng(0);
disp('Start checking tunning curve...');
nTestSet = 500;
winHeight = sqrt(size(basis,1)/2);
winWidth = sqrt(size(basis,1)/2);
Basis_num = size(basis,2);
blocks = rand(winHeight*winWidth,nTestSet)-0.5;
blocks = bsxfun(@rdivide,blocks,sqrt(sum(blocks.^2,1))+eps);
pairblocks = zeros(size(blocks,1)*2,length(range),size(blocks,2));
for i = range
    pairblocks(1:end/2,i-range(1)+1,:) = permute(blocks,[1,3,2]);
    pairblocks(end/2+1:end,i-range(1)+1,:) = permute(circshift(blocks,[winHeight*i,0]),[1,3,2]);
end
%
tunningCurve = zeros(length(range),Basis_num);
for j = 1:nTestSet
    tunningCurve = tunningCurve + abs(pairblocks(:,:,j)'*basis);
end
tunningCurve = tunningCurve/nTestSet;
% top = sum((tunningCurve - tunningCurve(end:-1:1,:)).^2);
% bot = sum((tunningCurve + tunningCurve(end:-1:1,:)).^2);
% asymIndex = mean(top./bot);

disp('Finish checking tunning curve!');
end