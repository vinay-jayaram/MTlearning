% Accuracy for a block trial. getbinacc(X,y,w,alpha).
% X: 2-D or 3-D matrix, last dimension trials
% y: labels equal to number of trials
% w: weight vector
% alpha: channel weight vector in the case of FD; give [] if none
function [acc] = getbinacc(X,y,w,alpha)
ntrials=length(y);
if isempty(alpha)
    acc=(sum(sign(w'*X)'==sign(y))/ntrials);
else
    trialacc=zeros(ntrials,1);
    for i = 1:ntrials
        trialacc(i)=(sign(alpha'*squeeze(X(:,:,i))*w)==y(i));
    end
    acc=sum(trialacc)/ntrials;
end
end