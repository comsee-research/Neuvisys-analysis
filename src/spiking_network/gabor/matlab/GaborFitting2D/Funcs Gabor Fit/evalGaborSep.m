function g = evalGaborSep(ps,xi,yi)        
    x0 = ps(1);
    y0 = ps(2);
    lambda = ps(3);
    sigmax = ps(4);
    sigmay = ps(5);
    theta = ps(6);
    phase = ps(7);
    a = ps(8);
    b = ps(9);
    
    xip =  (xi-x0)*cos(theta) + (yi-y0)*sin(theta);
    yip = -(xi-x0)*sin(theta) + (yi-y0)*cos(theta);
    
    g_in = exp(-xip.^2/2/sigmax^2 -yip.^2/2/sigmay^2).*cos(2*pi*xip/lambda+phase);
    g = a*g_in + b;
end