function [ out] = MT_FD(X,y,lambda,varargin)
%%
%Function to compute the multi-task learning approach with feature decomposition. Takes:
%    X: cell array of data, each element is a (channels x features x
%    trials) array
%    y: Cell array of labels, elements are (trial x1)
%    lambda: controls ratio between bias and data-dependent solution, between 0 and 1
%
% Outputs:
%   out.weight: Struct with learned feature mu and sigma, as well as all learned discrimination vectors
%   in the (features x subjects) array mat
%   out.alpha: Struct with learned spatial mu and sigma, as well as all learned discrimination vectors
%   in the (features x subjects) array mat
%
%
%optional arguments:
%
%   eta: Optional eta for use in the covariance calulations. Default 1e-3
%   alpha_init: Initial alpha topography (should be of norm 1). Default is (normalized) ones
%   vector
%   'prior':      {frequency mean,cov;spatial mean, cov} cell array if one wants to bias calculations
%                     or do ridge regression-ish. Default is zeros and I



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Argument parsing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eta = invarargin(varargin,'eta');
if isempty(eta)
    eta=1e-3;
end

a_init = invarargin(varargin,'alpha_init');
if isempty(a_init)
    a_init=ones(size(X{1},1),1)/sqrt(size(X{1},1));
end

priors = invarargin(varargin,'prior');
if isempty(priors)
    priors{1,1}=zeros(size(X{1},2),1);
    priors{1,2}=eye(size(X{1},2));
    priors{2,2}=eye(size(X{1},1));
    priors{2,1}=a_init;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variable initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

chans=size(X{1},1);
weight.sigma=priors{1,2};
weight.mu=priors{1,1};
mu_prev=ones(size(weight.mu));
alpha.mat=repmat(a_init,1,length(X));
alpha.mu=priors{2,1};
alpha.sigma=priors{2,2};
subjects=length(X);
count=0;
features=length(weight.mu);
weight.mat=zeros(length(weight.sigma),subjects);
MAX_ITERATIONS=5000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while sum(or(abs(weight.mu) > (mu_prev+0.01*mu_prev),abs(weight.mu) < (mu_prev-0.01*mu_prev)))>0 && count < MAX_ITERATIONS
    
    mu_prev=abs(weight.mu);
    %%%%%%%%%%%%%%
    % W and alpha update
    %%%%%%%%%%%%%%
    
    %reset weights and alphas
    weight.mat=zeros(length(weight.sigma),subjects);
    %but make sure we don't do this on the first iteration
    if count~=0
        alpha.mat=repmat(alpha.mu,1,length(X));
    end
    
    for i = 1:subjects
        w_prev=ones(length(weight.sigma),1);
        count2=1;
        ntrials=size(X{i},3);
        while sum(or(abs(weight.mat(:,i)) > (w_prev+0.01*w_prev),abs(weight.mat(:,i)) < (w_prev-0.01*w_prev)))>0 && count2<MAX_ITERATIONS
            
            w_prev = abs(weight.mat(:,i));
            
            %update W with old alphas
            aX_s=zeros(ntrials,features);
            for j = 1:ntrials
                aX_s(j,:)=alpha.mat(:,i)'*reshape(X{i}(:,:,j),chans,features);
            end
            % update W
            weight.mat(:,i)=(lambda*weight.sigma*aX_s'*aX_s+(1-lambda)*eye(size(aX_s,2)))\...
                (lambda*weight.sigma*aX_s'*y{i}+(1-lambda)*weight.mu);
            
            % update alpha with old W
            wX_s=zeros(chans,ntrials);
            for j=1:ntrials
                wX_s(:,j)=reshape(X{i}(:,:,j),chans,features)*weight.mat(:,i);
            end
            % update alpha
            alpha.mat(:,i)=(lambda*alpha.sigma*wX_s*wX_s'+(1-lambda)*eye(size(wX_s,1)))\...
                (lambda*alpha.sigma*wX_s*y{i}+(1-lambda)*alpha.mu);
            count2=count2+1;
        end
    end
    
    %%%%%%%%%%%%%%%
    % mu/sigma update
    %%%%%%%%%%%%%%%
    
    %if only one subject, break
    if subjects == 1
        break
    end
    
    weight.mu=mean(weight.mat,2);
    alpha.mu=mean(alpha.mat,2);
    
    %update Sigma
    temp=weight.mat-repmat(weight.mu,1,subjects);
    atemp=alpha.mat-repmat(alpha.mu,1,subjects);
    % constrain covariance to be trace one
    weight.sigma= temp*temp'/(trace(temp*temp'))+ eta*eye(length(weight.sigma));
    alpha.sigma= atemp*atemp'/(trace(atemp*atemp'))+ eta*eye(length(alpha.sigma));
    count=count+1;
end
out.weight=weight;
out.alpha=alpha;
if count == MAX_ITERATIONS
    warning('FailedConvergence','convergence failed')
end
end
