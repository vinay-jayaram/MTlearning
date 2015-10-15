function [L] = multibinloss(obj,X,y,varargin)
T = invarargin(varargin,'type');

sacc=zeros(1,length(X));
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
L=mean(sacc);
end

