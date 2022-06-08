function [predicted_basis] = gabor_4_field(Param, eyes)

    window = 4;

    sigma  = Param(1);
    theta  = Param(2);
    lambda = Param(3); 
    gamma  = Param(4); 
    psi1   = Param(5); 
    psi2   = Param(6);     
    c_x    = Param(7); 
    c_y    = Param(8);
    a      = Param(9);
    offset = Param(10);


    if eyes == 1 % both eyes
        b1 = gabor([sigma,theta,lambda,gamma,psi1,c_x,c_y,a,offset],window);
        b2 = gabor([sigma,theta,lambda,gamma,psi2,c_x,c_y,a,offset],window);    
        predicted_basis = [b1(:);b2(:)];
    else % single eye
        b3 = gabor([sigma,theta,lambda,gamma,psi1,c_x,c_y,a,offset],window);
        predicted_basis = [b3(:)];
    end
end

