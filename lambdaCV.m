function [l, cvout] = lambdaCV(f,loss,data,labels,varargin)
%% Documentation
% Function that cross-validates to find the best lambda from a range.
% Bootstraps to ensure classes have equal numbers
% Arguments
%       f:          Handle of function that determines decision rule.
%                   (X,y,lambda)->obj
%       loss:     Handle of function that determines loss, (obj,X,y)->loss
%                   measure
%       data:    data (cell array)
%       labels:  Class labels {1,-1} (cell array)
%
% Optional Arguments
%       n:             Number of CV loops (default 5)
%       parallel:    Parallel loops (<num cores> | none)
%       lrange:     Vector of lambda values (default [exp(-6),exp(-1:0.1:1),exp(6)])
%       verbose:             boolean, verbose (default 0)
%       bootstrap: boolean, bootstrap to equalize classes (default 1)

%% Argument parsing
n = invarargin(varargin,'n');
if isempty(n)
    n=5;
end

v = invarargin(varargin,'verbose');
if isempty(v)
    v=0;
end


bs = invarargin(varargin,'bootstrap');
if isempty(bs)
    bs=0;
elseif bs == 1
    disp('Bootstrapping within CV function');
end

parallel = invarargin(varargin,'parallel');
if isempty(parallel)
    parallel=0;
end

lrange = invarargin(varargin,'lrange');
if isempty(lrange)
    lrange=[exp(-6),exp(-1:0.1:1),exp(6)];
end

%% Main code
%Boostrap to ensure same number of samples per class [...probably we need
%to randomly sample more to average out but I'm leaving that out for now]

% Note that for unequal classes the cross-validation doesn't work
sten=ndims(data{1});
cln(1:(ndims(data{1})-1)) = {':'};
for i = 1:length(data)
    if sum(sign(labels{i})==1) ~= sum(sign(labels{i})==-1)  && bs == 1
        ordinalVec=[1,-1];
        [tmax,tind]=max([sum(sign(labels{i})==1),sum(sign(labels{i})==-1)]);
        smallerInd=find(sign(labels{i})==(-ordinalVec(tind)));
        biggerInd=setdiff(1:length(labels{i}),smallerInd)';
        cln(sten)={smallerInd};
        bigcln=cln;
        bigcln(sten)={biggerInd};
        smallerY=labels{i}(smallerInd);
        smallerX=data{i}(cln{:});
        bstrap = randi(length(smallerInd),tmax,1);
        cln(sten)={bstrap};
        data{i}=cat(ndims(data{i}),data{i}(bigcln{:}),smallerX(cln{:}));
        labels{i}=cat(1,labels{i}(biggerInd),smallerY(bstrap));
    end
end

cvloss=zeros(n,length(lrange));

if ~parallel
    for l = 1:length(lrange)
        if v; fprintf('Currently testing %dth value : %d\n',l,lrange(l));end
        test_indices={};
        
        %Initialize future test indices
        for i = 1:length(data)
            test_indices{i}=1:length(labels{i});
        end
        
        for it =1:n
            if v; fprintf('CV iteration : %d\n',it);end
            trialdata={};
            triallabels={};
            testlabels={};
            testdata={};
            for d = 1:length(data)
                % Choose test indices without trying to balance classes
                n_test=floor(size(data{d},sten)/n);
                testind=test_indices{d}(randperm(length(test_indices{d}),n_test));
                test_indices{d}= setdiff(test_indices{d},testind);
                cln(sten)={testind};
                testdata{d}=data{d}(cln{:});
                testlabels{d}=labels{d}(testind);
                cln(sten)={setdiff(1:length(labels{d}),testind)};
                trialdata{d}=data{d}(cln{:});
                triallabels{d}=labels{d}(setdiff(1:length(labels{d}),testind));
            end
            
            ret_obj = f(trialdata, triallabels, lrange(l));
            cvloss(it,l)=loss(ret_obj,testdata,testlabels);
        end
    end
else
    if v; fprintf('Using parallel feature\n');end
    pool = gcp('nocreate');
    if isempty(pool)
        pool=parpool(parallel);
    else
        disp('Using pre-existing pool');
    end
    for l = 1:length(lrange)
        if v; fprintf('Currently testing %dth value : %d\n',l,lrange(l));end
        test_indices={};
        
        %Initialize future test indices
        for i = 1:length(data)
            test_indices{i}=1:length(labels{i});
        end
        
        %partition outside parallel loop
        par_testind={};
        for d = 1:length(data)
            % Choose test indices without trying to balance classes
            n_test=floor(size(data{d},sten)/n);
            par_testind{d}=test_indices{d}(randperm(length(test_indices{d}),n_test));
            test_indices{d}= setdiff(test_indices{d},par_testind{d});
        end
        
        parfor it =1:n
            if v; fprintf('CV iteration : %d\n',it);end
            trialdata={};
            triallabels={};
            testlabels={};
            testdata={};
            par_cln=cln;
            for d = 1:length(data)
                % Choose test indices without trying to balance classes
                par_cln(sten)={par_testind{d}};
                testdata{d}=data{d}(par_cln{:});
                testlabels{d}=labels{d}(par_testind{d});
                par_cln(sten)={setdiff(1:length(labels{d}),par_testind{d})};
                trialdata{d}=data{d}(par_cln{:});
                triallabels{d}=labels{d}(setdiff(1:length(labels{d}),par_testind{d}));
            end
            
            ret_obj = f(trialdata, triallabels, lrange(l));
            cvloss(it,l)=loss(ret_obj,testdata,testlabels);
        end
    end
    delete(pool);
end
cvout=cvloss;
cvloss=mean(cvloss,1);
[~,l]=min(cvloss);
l=lrange(l(end));
end
