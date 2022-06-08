function out = sinewavegrating2D(ps,patchsize)
    [xi,yi] = meshgrid(1:patchsize,1:patchsize);
    x0 = round(patchsize/2);
    y0 = round(patchsize/2);
    lambda = ps(1);
    theta = ps(2);
    phase = ps(3);
    
    xip =  (xi-x0)*cos(theta) + (yi-y0)*sin(theta);
    yip = -(xi-x0)*sin(theta) + (yi-y0)*cos(theta);
    
    out = cos(2*pi*xip/lambda+phase);
end