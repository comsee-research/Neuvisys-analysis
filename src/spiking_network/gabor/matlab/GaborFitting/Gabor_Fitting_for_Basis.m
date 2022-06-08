% Fit gabor function defined in gabor.m to the basis functions. Use "eyes"
% to dinstinguish between both, left or right patch. The fit is initialized
% with random parameters and executed "nSamples" times. The parameters with the
% smallest residual norm are saved into .mat file.

function [Fitted_Gabor_Filter_Parameters, Error] = Gabor_Fitting_for_Basis(basis, eyes, fitFreq)

nBasis = size(basis, 2); % Number of basis functions
nSamples = 150; % Number of random gabors/inits

% The function  expects a Basis vector where
% Rows    = 64*2
% Columns = 288
Parameter_Set = [];
Resnorm_Set = [];

Parameter_Set(1:nBasis,1:10) = 0.0;
Resnorm_Set(1:nBasis,1) = 0.0;

parfor Index = 1:nBasis
% for Index = 1:nBasis
    %disp(Index);

    rs_norm_high = 1000;
    Best_Parameter_Set = [];

    if eyes == 1
        Selected_Basis = basis(:,Index);
    elseif eyes == 2
        Selected_Basis = basis(1:end/2, Index);
    elseif eyes == 3
        Selected_Basis = basis(end/2+1:end, Index);
    end

    if fitFreq == 0
%         low = [0,-pi/2,2.0,0.01,-pi/2,-pi/2,-4,-4,-Inf,-Inf]; % should be more accurate
        low = [0,-pi/2,0.0,0.01,-pi/2,-pi/2,-4,-4,-Inf,-Inf];
        up = [4,pi/2,16.0,20,pi/2,pi/2,4,4,Inf,Inf];
    else
        low = [0,-pi/2,0.0,0.01,-pi/2,-pi/2,-4,-4,-Inf,-Inf];
        up = [4,pi/2,0.5,20,pi/2,pi/2,4,4,Inf,Inf]; % replace wavelength by frequency
    end

    for random_start = 1:nSamples
%     parfor random_start = 1:nSamples

        if fitFreq == 0
            start = [rand*100,-1+rand*1,rand*16,rand*20,-pi/2 + pi*rand,-pi/2 + pi*rand,10*rand,10*rand,10*rand,10*rand];
        else
            start = [rand*100,-1+rand*1,rand*0.5,rand*20,-pi/2 + pi*rand,-pi/2 + pi*rand,10*rand,10*rand,10*rand,10*rand];
        end
        
        %start = [rand*1,-1+rand*1,rand*16,rand*20,-pi/2 + pi*rand,-pi/2 + pi*rand,1*rand,1*rand,10*rand,10*rand];

        %options = optimset('Jacobian','on');
        options = optimset('MaxIter',10000,'MaxFunEvals',10000,'Display','off');
        
        if fitFreq == 0
            [param,resnorm] = lsqcurvefit(@gabor_4_field, start, eyes, Selected_Basis, low, up, options);
        else
            [param,resnorm] = lsqcurvefit(@gabor_4_field_freq, start, eyes, Selected_Basis, low, up, options);
        end
        
        if resnorm < rs_norm_high
            Best_Parameter_Set = param;
            rs_norm_high = resnorm;
        end
    end

    %disp([rs_norm_high])

    Parameter_Set(Index,:) = Best_Parameter_Set;
    Resnorm_Set(Index,:) = rs_norm_high;
    %Parameter_Set = [Parameter_Set;Best_Parameter_Set];
    %Resnorm_Set = [Resnorm_Set;rs_norm_high];
end

Fitted_Gabor_Filter_Parameters = Parameter_Set;
Error = Resnorm_Set;

% save(['Parameters_Coarse_3D0',int2str(i)],...
%     'Parameter_Set');
% save(['Error_Coarse_3D0',int2str(i)],...
%     'Resnorm_Set');
