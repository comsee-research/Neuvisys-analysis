function [results,exit_flag] = autoGaborSurf_Onepatch(xi,yi,z)
%function [results] = autoGaborSurf(xi,yi,zi1)
%
% Fit a Gabor (a sinusoid windowed by a Gaussian) to a surface.
% The Gabor is defined as follows:
% zi1 = a*exp(-(xip.^2/2/sigmax^2+yip.^2/2/sigmay^2))*cos(2*pi*xip/lambda + phase) + b
%
% Where:
% xip = (xi-x0)*cos(theta) + (yi-y0)*sin(theta);
% yip =-(xi-x0)*sin(theta) + (yi-y0)*cos(theta);
theta = pi/2;
lambda = 5;
sigmax = 5;
sigmay = 5;
phase = 0;
ps = [];
error = 100;
% explore different inital value of mu
for muy=0:5:10
    for mux=0:5:10        
        %Now use lsqcurvefit to finish
        p0 = [mux,muy,lambda,sigmax,sigmay,theta,phase,1,0];
        opts = optimset('Jacobian','on','Display','off');
        long_zi = z(:);
        lb = [-Inf,-Inf,2,0,0,0,0,0,-Inf]';
        ub = [Inf,Inf,20,50,50,pi,2*pi,Inf,Inf]';       
        [ps_temp,E,~,exit_flag] = lsqcurvefit(@(x,xdata) evalGaborOnepatch(x,xdata),p0,[xi(:),yi(:)],long_zi,[lb],[ub], opts);
        % sotre the ps that has the smallest error
        if(E<error)
            error = E;
            ps = ps_temp;
        end        
    end
end
% save results
results.x0 = ps(1); results.y0 = ps(2);
results.lambda= ps(3);
results.sigmax = ps(4); results.sigmay = ps(5);
results.theta = ps(6); 
results.phase = ps(7);
results.a = ps(8); results.b = ps(9);
H = [cos(ps(6)), -sin(ps(6)); sin(ps(6)), cos(ps(6))];
results.K = H * diag([ps(4)^2,ps(5)^2]) * H';
results.sse = error;
% compute the estimated basis
results.G = evalGaborSep(ps,xi,yi);
end

