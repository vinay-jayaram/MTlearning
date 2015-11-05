function [priors, accs] = MTdataset(data,labels,pparams,varargin)
% Function that given a dataset and labels: does LOO cross-validation
% where priors are computed on all but one and cross-validated accuracy is
% computed within the one. 
n_cv=invarargin(varargin,'cv');
if isempty(n_cv)
    n_cv=5;
end
sten=ndims(data{1});
cln(1:(ndims(data{1})-1)) = {':'};
sub=length(data);
priors=cell(1,sub);
accs=zeros(1,sub);
for i = 1:sub
    fprintf('Testing on dataset %d \n',i);
    testX=data{i};
    trainX=data(setdiff(1:sub,i));
    testy=labels{i};
    trainy=labels(setdiff(1:sub,i));
    priors{i}=multitask2015(trainX,trainy,pparams{:});
    temp=zeros(1,n_cv);
    indices=1:size(data{i},sten);
    for j = 1:n_cv
        fprintf('CV loop %d\n',j);
        testind=indices(randperm(length(indices),floor(size(data{i},sten)/n_cv)));
        indices=setdiff(indices,testind);
        trainind=setdiff(1:size(data{i},sten),testind);
        cln(sten)={testind};
        CVtestX=testX(cln{:});
        CVtesty=testy(cln{sten});
        cln(sten)={trainind};
        new=multitask2015(testX(cln{:}),testy(cln{sten}),'prior',priors{i},'cv_params',{'n',5});
        if sten==2
            accs(i)=accs(i)+getbinacc(CVtestX,CVtesty,new.mat,[]);
        elseif sten==3
            accs(i)=accs(i)+getbinacc(CVtestX,CVtesty,new.weight.mat,new.alpha.mat);
        end
    end
    
end
accs=accs/n_cv;








end