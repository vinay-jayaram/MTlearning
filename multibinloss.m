function [sacc] = multibinloss(obj,X,y,varargin)
if ndims(X{1})==2
    T='';
elseif ndims(X{1})==3
    T='FD';
end


% correct for cases when there is only one test trial and so ndims is not
% caught properly
if isfield(obj,'alpha') && length(y)==1 && strcmp(T,'')
    T='FD';
    for i = 1:length(X)
        X{i}=reshape(X{i},size(X{i},1),size(X{i},2),1);
    end
end

sacc=zeros(length(X),1);
for i = 1:length(X)
    switch T
        case 'FD'
            sacc(i)=getbinacc(X{i},y{i},obj.weight.mat(:,i),obj.alpha.mat(:,i));
        case ''
            sacc(i)=getbinacc(X{i},y{i},obj.mat(:,i),[]);
        otherwise
            error('Invalid type for calculation')
    end
end
if sum(isnan(sacc))>0
    error('NaN accuracy values found');
end
end

