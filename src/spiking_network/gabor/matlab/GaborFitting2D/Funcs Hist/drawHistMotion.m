function drawHistMotion(prev_phase,curr_phase,lambdas,thetas)
ub = 10;
thetas_rad = round(180*thetas/pi);
thetas_rad = mod(thetas_rad,180);
delta_phase = calcAngle(prev_phase,curr_phase);
diffvel = lambdas.*delta_phase/(2*pi);
%
phase_diff_r = [];
thetas_r = [];
samples = length(lambdas);
for i=1:samples
    if(diffvel(i)<=ub)
        phase_diff_r = [phase_diff_r diffvel(i)];
        thetas_r = [thetas_r thetas_rad(i)];
    end
end
% diffvel(diffvel>ub) = 0;
X = [phase_diff_r;thetas_r];
figure;
% hist3(X',[10 4]);
hist3(X',{0:1:10 5:10:175})
set(gcf,'renderer','opengl');
set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
zlim([0 50]);
title('Histogram over \theta and \phi');
end