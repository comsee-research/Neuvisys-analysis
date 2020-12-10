function drawHistLambda(lambdas)
lb = 2;
ub = 20;
figure;
lambdas(lambdas>ub) = 0;
hist(lambdas,[0:1:ub]);
axis([lb ub 0 100]);
title('Histogram over \lambda');
end