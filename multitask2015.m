function [out] = multitask2015(data, labels, varargin)
%% Documentation
% Vinay Jayaram, 14.10.15, high-level implementation of Jayaram et al.
% 2016
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

eta = invarargin(varargin,'eta');
if isempty(eta)
    eta=1e-3;
end

cv_params=invarargin(varargin,'cv_params');
if isempty(cv_params)
    cv_params={};
end
cv_params=cat(2,cv_params,{'verbose',v});
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
    [lambda, rr_cv] = lambdaCV(@(x,y,l)(MT_FD(x,y,l)),@(obj,x,y)(multibinloss(obj,x,y,'type','FD')),{Xall},{yall});
    rr_out=MT_FD({Xall},{yall},lambda,'eta',eta);
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

%% Main control

switch T
    case ''
        [ lam, cvaccs ] = lambdaCV(@(x,y,l)(MT(x,y,l,'eta',eta,'prior',prior)),@(obj,x,y)(multibinloss(obj,x,y,'type','')),data,labels,cv_params{:});
        out=MT(data,labels,lam,'prior',prior,'eta',eta);
    case 'FD'
        [ lam, cvaccs ] = lambdaCV(@(x,y,l)(MT_FD(x,y,l,'eta',eta,'prior',prior)),@(obj,x,y)(multibinloss(obj,x,y,'type','FD')),data,labels,cv_params{:});
        out=MT_FD(data,labels,lam,'prior',prior,'eta',eta);
end

out.lambda=lam;
out.cvacc=cvaccs;
end
