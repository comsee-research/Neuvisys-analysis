function rds = gen_rds2(winsize,range)
rng(0);
batch_size = 500;
imsize = 200;
image = rand(imsize);
posx = randi([1+range imsize-winsize-range],1,batch_size);
posy = randi([1+range imsize-winsize-range],1,batch_size);
rds = zeros(2*winsize^2,2*range+1,batch_size);
for i=-range:range
    impair = zeros(2*winsize^2,batch_size);
    for j=1:batch_size
        % generate hori shift image
        %         sub_iml = image(posx(j):posx(j)+winsize-1,posy(j):posy(j)+winsize-1);
        %         sub_imr = image(posx(j):posx(j)+winsize-1,posy(j)+i:posy(j)+i+winsize-1);
        % generate hori shift image
        sub_iml = image(posx(j):posx(j)+winsize-1,posy(j):posy(j)+winsize-1);
        sub_imr = image(posx(j)+i:posx(j)+i+winsize-1,posy(j):posy(j)+winsize-1);
        impair(:,j) = reshape([sub_iml,sub_imr],[2*winsize^2 1]);
    end
    impair = impair - ones(size(impair,1),1)*mean(impair);
    impair = bsxfun(@rdivide,impair,sqrt(sum(impair.^2)+eps));
    rds(:,i+range+1,:) = impair;
end
end