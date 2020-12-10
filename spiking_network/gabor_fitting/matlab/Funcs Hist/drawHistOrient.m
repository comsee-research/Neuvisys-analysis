function thetas_rad = drawHistOrient(thetas)
thetas_rad = round(180*thetas/pi);
thetas_rad = mod(thetas_rad,180);
%
% idx = find((thetas_rad<=120) & (thetas_rad>=60));
% idx = find(((thetas_rad<=10) & (thetas_rad>=0)) | ((thetas_rad<=179) & (thetas_rad>=170)));
%
figure;
histn(thetas_rad,0,15,180);
%hist(thetas_rad,0:30:180);
%hist(thetas_rad,5:10:175);
% set(gca,'XTick',-7.5:15:187.5)
xlim([-10 190]);
ylim([0 100]);
xlabel('orientation of bases');
ylabel('number of bases');
% hist(thetas_rad,0:45:180);
% axis([0 170 0 100]);
title('Histogram over \theta');
figureHandle = gcf;
set(findall(figureHandle,'type','text'),'fontSize',16,'fontWeight','bold')
end