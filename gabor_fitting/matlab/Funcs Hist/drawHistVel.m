function vels = drawHistVel(phase1,phase2,lambdas, thetas)
delta_phase = angle(exp(j*(phase2 - phase1)));
vels = delta_phase.*lambdas./(2*pi);
figure;
bin_c = 0.2*[-10:2:10];
histn(vels,-10,2,10);
xlim([-11 11]);
ylim([0 220]);
set(gca,'XTickLabel',bin_c);
xlabel({'velocity along the direction perpendicular';'to basis orientation deg/(time step)'})
% xlabel('velocity projected on horizontal direction deg/(time step)');
ylabel('number of bases');
title('Histogram over prefered velocity');
figureHandle = gcf;
set(findall(figureHandle,'type','text'),'fontSize',16,'fontWeight','bold')
end