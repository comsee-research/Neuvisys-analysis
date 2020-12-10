function g = gabor_2d(xi,yi,ps)
x0 = ps(1);
y0 = ps(2);
theta = ps(3);
lambda = ps(4);
sigmax = ps(5);
sigmay = ps(6);
phase = ps(7);

xip =  (xi-x0)*cos(theta) + (yi-y0)*sin(theta);
yip = -(xi-x0)*sin(theta) + (yi-y0)*cos(theta);

g = exp(-xip.^2/2/sigmax^2 -yip.^2/2/sigmay^2).*cos(2*pi*xip/lambda+phase);
end