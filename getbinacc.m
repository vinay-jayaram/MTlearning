% Accuracy for a block trial. MT_getacc(X,y,w,alpha).
function [acc] = getbinacc(X,y,w,alpha)
ntrials=length(y);
if isempty(alpha)
    acc=(sum(sign(X*w)==sign(y))/ntrials);
else
    trialacc=zeros(ntrials,1);
    for i = 1:ntrials
        trialacc(i)=(sign(alpha'*squeeze(X(:,:,i))*w)==y(i));
    end
    acc=sum(trialacc)/ntrials;
end
end