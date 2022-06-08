% Gabor Filter as taken from Wiki page
% f : Frequency of the sinusoidal factor
% Theta: Orientation of the normal to the parallel stripes of the Gabor
% Psi : Phase offset
% Sigma: Std of the Gaussain envelope
% Gamma: Spatial aspect ratio


%function gb=gabor([sigma,theta,lambda,psi,gamma,c_x,c_y], patchSize/2)
function gb=gaborFreq(Param,xmax)

sigma = Param(1);
theta = Param(2);
f = Param(3);
gamma = Param(4);
psi = Param(5);
c_x = Param(6);
c_y = Param(7);
a   = Param(8);
offset = Param(9);


sigma_x = sigma;
sigma_y = sigma/gamma;
%sigma_y = gamma;


%produces an 8x8 mesh
ymax = xmax;

xmin = -xmax+1; ymin = -ymax+1;
[x,y] = meshgrid(xmin+c_x:xmax+c_x,ymin+c_y:ymax+c_y);

% Rotation
x_theta=x*cos(theta)+y*sin(theta);
y_theta=-x*sin(theta)+y*cos(theta);

%gb= a * exp(-.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)) .* cos((2*pi*x_theta/lambda)+psi) + offset;
gb= a * exp(-.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)) .* cos((2*pi*x_theta*f)+psi) + offset;