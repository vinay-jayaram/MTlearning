function [out] = multitask2015(data, labels, varargin)
%% Documentation
% function [out] = multitask2015(data, labels, varargin)
% Vinay Jayaram, 14.10.15, high-level implementation of Jayaram et al.
% 2016
%
% ***Man I really wish MATLAB had good argument parsing
%
%*** note: so far there is no possibility of regression, just binary
%classification. This can easily be added by the user.
%
% If data is 3D, it assumes frequency decomposition (and that the structure
% is [electrodes,features,trials]. 2D assumed structure is
% [features,trials]
%
% Optional arguments:
% 'eta'        - Define eta for learning algorithm (default 0.001)
% 'alpha_init' - Vector of size chans to initialize alpha with
% 'verbose'    - Turn on or off messages
% 'rr_init' - boolean, initialize with ridge regression solution,
% overwrites alpha_init
% 'prior' - None | cell array of objects, does ridge regression with
% specified prior
% 'cv_params' - cell array of varargin to give cross-validation function
% 'cv' - cross validate lambda instead of ML update (binary 1/0) [ default
% 0]
% 'boostrap' - Equalize classes if not equalized, default 0
% 'out_acc' - Return cross-validated multi-task accuracy over training data using
% learned lambda, default 0

%% Argument parsing

%first wrap data/labels if they're not in cell arrays
if ~iscell(data)
    data={data};
end

if ~iscell(labels)
    labels={labels};
end

switch ndims(data{1})
    case 2
        T='';
    case 3
        T='FD';
    otherwise
        error('Dimension of data is %d which is not allowed',ndim(data{1}));
end

v = invarargin(varargin,'verbose');
if isempty(v)
    v=0;
end

outacc = invarargin(varargin,'out_acc');
if isempty(outacc)
    outacc=0;
end


eta = invarargin(varargin,'eta');
if isempty(eta)
    eta=1e-3;
end

lambdaML = invarargin(varargin,'cv');
if isempty(lambdaML)
    lambdaML=1;
else
    lambdaML=~lambdaML;
end

bs = invarargin(varargin,'bootstrap');
if isempty(bs)
    bs=0;
end

cv_params=invarargin(varargin,'cv_params');
if isempty(cv_params)
    cv_params={};
else
    lambdaML=1;
end
cv_params=cat(2,cv_params,{'verbose',v,'bootstrap',bs});

a_init = invarargin(varargin,'alpha_init');
if isempty(a_init)
    a_init=ones(size(data{1},1),1)/sqrt(size(data{1},1));
end

rr_init = invarargin(varargin,'rr_init');
if isempty(rr_init)
    rr_init=0;
end



if rr_init
    if strcmp(T,'')
        error('No ridge regression solution for non-FD necessary');
    end
    if v; disp('Beginning ridge-regression computation');end
    Xall=[];yall=[];
    for i = 1:length(data)
        Xall=cat(3,Xall,data{i});
        yall=cat(1,yall,labels{i});
    end
    [lambda, rr_cv] = lambdaCV(@(x,y,l)(MT_FD(x,y,'lambda',l)),@(obj,x,y)(multibinloss(obj,x,y,'type','FD')),{Xall},{yall});
    rr_out=MT_FD({Xall},{yall},'lambda',lambda,'eta',eta,'verbose',v);
    a_init=rr_out.alpha.mat;
    if v; disp('Spatial prior computation finished');end
end

prior = invarargin(varargin,'prior');
if ~isstruct(prior)
    switch T
        case ''
            prior={zeros(size(data{1},1),1),eye(size(data{1},1))};
        case 'FD'
            prior{1,1}=zeros(size(data{1},2),1);
            prior{1,2}=eye(size(data{1},2));
            prior{2,1}=a_init;
            prior{2,2}=eye(length(a_init));
    end
else
    temp={};
    switch T
        case ''
            temp{1,1}=prior.mu;
            temp{1,2}=prior.sigma;
        case 'FD'
            temp{1,1}=prior.weight.mu;
            temp{1,2}=prior.weight.sigma;
            temp{2,1}=prior.alpha.mu;
            temp{2,2}=prior.alpha.sigma;
    end
    prior=temp;
end

if length(data)==1
    if v; disp('One task detected; cross-validation turned on'); end
    lambdaML=0;
end

%% Main control

%sample with replacement to equalize classes if bs option passed
if lambdaML &&bs
        disp('Bootstrapping before prior computation');
    sten=ndims(data{1});
    cln(1:(ndims(data{1})-1)) = {':'};
    for i = 1:length(data)
        if sum(labels{i}==1) ~= sum(labels{i}==-1)
            [~, tmax]=sort(sum(labels{i}==1),sum(labels{i}==-1));
            bstrap = randi(length(labels{i}),tmax,1);
            cln(sten)={bstrap};
            data{i}=data{i}(cln{:});
            labels{i}=labels{i}(bstrap);
        end
    end    
end

switch T
    case ''
        if ~lambdaML
            [ lam, cvaccs ] = lambdaCV(@(x,y,l)(MT(x,y,'lambda',l,'eta',eta,'prior',prior)),@(obj,x,y)(mean(multibinloss(obj,x,y,'type',''))),data,labels,cv_params{:});
            out=MT(data,labels,'lambda',lam,'prior',prior,'eta',eta,'verbose',v);
            out.cvacc=cvaccs;
        else
            out=MT(data,labels,'prior',prior,'eta',eta,'verbose',v);
        end
    case 'FD'
        if ~lambdaML
            [ lam, cvaccs ] = lambdaCV(@(x,y,l)(MT_FD(x,y,l,'eta',eta,'prior',prior)),@(obj,x,y)(mean(multibinloss(obj,x,y,'type','FD'))),data,labels,cv_params{:});
            out=MT_FD(data,labels,'lambda',lam,'prior',prior,'eta',eta,'verbose',v);
            out.cvacc=cvaccs;
        else
            out=MT_FD(data,labels,'prior',prior,'eta',eta,'verbose',v);
        end
end

% Warning: The below counts as double-dipping. We use all the data to learn
% a lambda then cross-validate with the learned lambda to generate decision functions. 
% The best way to ensure your results are reasonable is to train the lambda
% on a separate dataset from the test
if outacc
    n=10;
    sten=ndims(data{1});
    cln(1:(ndims(data{1})-1)) = {':'};
    cvacc=zeros(length(data),n);
    if v; disp('Computing cross-validated accuracy on training data using learned lambda');end
    %Initialize future test indices
    for i = 1:length(data)
        test_indices{i}=1:length(labels{i});
    end
    for it =1:n % currently hard-coded...should change?
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
        switch T
            case ''
                ret_obj=MT(trialdata, triallabels,'lambda',out.lambda);
                %not the smartest way to deal with this
                ret_obj.alpha=[];
            case 'FD'
                ret_obj=MT_FD(trialdata, triallabels, 'lambda',out.lambda);
        end
        cvacc(:,it)=multibinloss(ret_obj,testdata,testlabels);
    end
    out.trainacc=cvacc;
end

end
