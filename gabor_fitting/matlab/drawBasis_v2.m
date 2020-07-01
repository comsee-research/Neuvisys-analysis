function h=drawBasis_v2(A,B, szr,szc)
M=size(A,2);
m=sqrt(M);
%n=m;
n=M/m;
figure;
buf=1;
array=-ones(buf+m*(szr+buf),buf+n*(szc+buf));
k=1;
for i=1:m
    for j=1:n
        clim=max(abs(A(:,k)));
        prev = reshape(A(1:100,k),10,10)/clim;
        curr = reshape(B(1:100,k),10,10)/clim;
        array(buf+(i-1)*(szr+buf)+[1:szr],buf+(j-1)*(szc+buf)+[1:szc]) = [prev;curr];        
        k=k+1;
    end
end
if exist('h','var')
    set(h,'CData',array);
else
    h=imagesc(array,'EraseMode','none',[-1 1]);
    axis image off
end
drawnow
