function drawHistOrientDisparity(disparity,thetas)
thetas_rad = round(180*thetas/pi);
thetas_rad = mod(thetas_rad,180);
X = [thetas_rad;disparity(1,:)]';
figure;
hist3(X,{0:30:180,-10:2:10});
set(gca,'YTickLabel',[-2:0.4:2]);
zlim([0 40]);
xlabel('bases orientation (deg)');
ylabel('velocity (deg/time step)');
zlabel('number of bases');
set(gcf,'renderer','opengl');
set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
end