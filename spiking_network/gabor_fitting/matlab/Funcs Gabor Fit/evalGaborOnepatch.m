function [z, J]= evalGaborOnepatch(ps,xdata)
    yi = xdata(:,2);
    xi = xdata(:,1);
    z = evalGaborSep(ps,xi,yi);
    
    J = zeros(size(xdata,1),length(ps));    
    %Use finite differencing for the regular parameters
    for ii = 1:length(ps)
        delta = zeros(1,length(ps));
        delta(ii) = 1e-5;
        est_z = evalGaborSep(ps+delta,xi,yi);
        J(:,ii) = (est_z-z)/1e-5;
    end
end

