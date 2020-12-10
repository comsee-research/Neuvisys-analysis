function drawHistLocality(mu)
figure;
% 
hist3(mu','edges',{-5:1:15 -5:1:15});
xlabel('x coord'); ylabel('y coord');
set(gcf,'renderer','opengl');
set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
title('Histogram over \mu');
end