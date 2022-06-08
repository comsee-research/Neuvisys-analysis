function disparity = drawHistDisparity(left_phase,right_phase,lambdas,thetas)
ub = 10;
delta_phase = angle(exp(j*(right_phase - left_phase)));
% sin_thetas = sin(thetas);
% cos_thetas = cos(thetas);
% idx = find((thetas>=pi/4) & (thetas<=3*pi/4));
% pro_angles = cos_thetas;
% pro_angles(idx) = sin_thetas(idx);
% caculate delta phase
% d_exci = delta_phase.*lambdas./(2*pi*pro_angles);
% d_inhi = (delta_phase-sign(delta_phase)*pi).*lambdas./(2*pi*pro_angles);
%
% d_exci = delta_phase.*lambdas./(2*pi*cos(thetas));
% d_inhi = (delta_phase-sign(delta_phase)*pi).*lambdas./(2*pi*cos(thetas));
d_exci = delta_phase.*lambdas./(2*pi);
d_inhi = (delta_phase-sign(delta_phase)*pi).*lambdas./(2*pi);
disparity = [d_exci;d_inhi];
% disparity(:,idx) = 0;

% delta_phase = zeros(1,length(left_phase));
% temp_delta_phase = angle(exp(j*(right_phase-left_phase)));
% for i=1:length(delta_phase)
%     if(abs(temp_delta_phase(i))>pi/2)
%         if(temp_delta_phase(i)>0)
%             delta_phase(i) = temp_delta_phase(i) - pi;
%         else
%             delta_phase(i) = pi + temp_delta_phase(i);
%         end
%     else
%         delta_phase(i) = temp_delta_phase(i);
%     end
% end
% % convert phase shift to pixel shift
% disparity = lambdas.*delta_phase/(2*pi);
% % convert pixel shift to disparity
% disparity = disparity.*cos(thetas);
% kick out disparity out of range
% disparity_r = [];
% samples = length(lambdas);
% for i=1:samples
%     if(abs(disparity(i))<=ub)
%         disparity_r = [disparity_r disparity(i)];
%     end
% end
% % show histogram
% figure;
% hist(disparity_r,-ub:1:ub);
% axis([-ub ub 0 300]);
% title('Histogram over Disparity');
end