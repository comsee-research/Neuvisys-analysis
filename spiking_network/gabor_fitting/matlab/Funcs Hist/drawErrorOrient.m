function drawErrorOrient(thetas,true_tc,est_disparity)
[~,true_max] = max(true_tc);
[~,true_min] = min(true_tc);
true_max = true_max-11;
true_min = true_min-11;
est_max = est_disparity(1,:);
est_min = est_disparity(2,:);
ls_error = sqrt((true_max-est_max).^2 + (true_min-est_min).^2);
figure;
plot(thetas,ls_error);
axis([0 pi 0 200]);
xlabel('basis orientation');
ylabel('error');
end