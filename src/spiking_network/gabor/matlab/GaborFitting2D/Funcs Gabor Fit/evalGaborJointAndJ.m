function [f,J] = evalGaborJointAndJ(ps,xdata)
    yi = xdata(:,2);
    xi = xdata(:,1);
    ps1 = [ps(1:5),ps(6:9)];
    ps2 = [ps(1:5),ps(10:13)];
    %
    g1 = evalGaborSep(ps1,xi,yi);
    g2 = evalGaborSep(ps2,xi,yi);    
    f = [g1;g2];
    J = zeros(2*size(xdata,1),13);    
    %Use finite differencing for the regular parameters
    for ii = 1:5
        delta = zeros(1,9);
        delta(ii) = 1e-5;
        est_g1 = evalGaborSep(ps1+delta,xi,yi);
        est_g2 = evalGaborSep(ps2+delta,xi,yi);        
        est_f = [est_g1;est_g2];
        J(:,ii) = (est_f-f)/1e-5;
    end
    for ii = 6:9
        delta = zeros(1,9);
        delta(ii) = 1e-5;
        est_g1 = evalGaborSep(ps1+delta,xi,yi);
        est_g2 = evalGaborSep(ps2,xi,yi);        
        est_f = [est_g1;est_g2];
        J(:,ii) = (est_f-f)/1e-5;
    end
    for ii = 10:13
        delta = zeros(1,9);
        delta(ii-4) = 1e-5;
        est_g1 = evalGaborSep(ps1,xi,yi);
        est_g2 = evalGaborSep(ps2+delta,xi,yi);        
        est_f = [est_g1;est_g2];
        J(:,ii) = (est_f-f)/1e-5;
    end
end

