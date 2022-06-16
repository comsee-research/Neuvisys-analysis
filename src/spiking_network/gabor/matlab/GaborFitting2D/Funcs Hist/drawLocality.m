function locality = drawLocality(sigma,lambda)
figure;
locality = abs(sigma(1,:))./lambda;
plot(1:length(lambda),locality,'b.');
% plot(lambda,abs(sigma(1,:)),'b.');
xlabel('basis index');
ylabel('\sigma_x/\lambda');
title('\sigma_x/\lambda');
end