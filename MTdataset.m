function [priors, accs] = MTdataset(data,labels,pparams,varargin)
% Function that given a dataset and labels: does LOO cross-validation
% where priors are computed on all but one and cross-validated accuracy is
% computed within the one.
n_cv=invarargin(varargin,'cv');
cvparams = invarargin(varargin,'cv_params');
if ~iscell(cvparams)
    cvparams = {'n',5};
end
nitr = invarargin(varargin,'itr');
v=invarargin(varargin,'verbose');
if isempty(n_cv)
    n_cv=5;
end
if isempty(nitr)
    nitr=100;
end
sten=ndims(data{1});
cln(1:(ndims(data{1})-1)) = {':'};
sub=length(data);
priors=cell(1,sub);
accs=zeros(nitr,sub);
multisub = sub>1;
for i = 1:sub
    fprintf('Testing on dataset %d \n',i);
    testX=data{i};
    trainX=data(setdiff(1:sub,i));
    testy=labels{i};
    trainy=labels(setdiff(1:sub,i));
    if multisub
        priors{i}=multitask2015(trainX,trainy,pparams{:});
    end
    for it = 1:nitr;
        temp=zeros(1,n_cv);
        indices=1:size(data{i},sten);
        for j = 1:n_cv
            if v,fprintf('CV loop %d\n',j);end
            testind=indices(randperm(length(indices),floor(size(data{i},sten)/n_cv)));
            indices=setdiff(indices,testind);
            trainind=setdiff(1:size(data{i},sten),testind);
            cln(sten)={testind};
            CVtestX=testX(cln{:});
            CVtesty=testy(cln{sten});
            cln(sten)={trainind};
            if multisub
                new=multitask2015(testX(cln{:}),testy(cln{sten}),'prior',priors{i},'cv_params',cvparams);
            else
                new=multitask2015(testX(cln{:}),testy(cln{sten}),'cv_params',cvparams);
            end
            if sten==2
                accs(it,i)=accs(it,i)+getbinacc(CVtestX,CVtesty,new.mat,[]);
            elseif sten==3
                accs(it,i)=accs(it,i)+getbinacc(CVtestX,CVtesty,new.weight.mat,new.alpha.mat);
            end
        end
    end
end
accs=accs/n_cv;








end