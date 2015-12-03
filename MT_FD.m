function [ out] = MT_FD(X,y,varargin)
%%
% Experiment: set lambda as average residual
%
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
%   'lambda':    Define external lambda and don't use ML update
%   'verbose':   Print convergence information per iteration
w = warning ('off','MATLAB:nearlySingularMatrix');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Argument parsing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eta = invarargin(varargin,'eta');
if isempty(eta)
    eta=1e-3;
    itEta=1;
else
    itEta=0;
end

verbose = invarargin(varargin,'verbose');
if isempty(verbose)
    verbose=0;
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


inlambda=invarargin(varargin,'lambda');
if isempty(inlambda)
    lambdaML=1;
    lambda=1;
else
    lambda=inlambda;
    lambdaML=0;
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
PCT=0.02;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while sum(or(abs(weight.mu) > mu_prev+PCT*mu_prev,abs(weight.mu) < mu_prev-PCT*mu_prev))>0 && count < MAX_ITERATIONS
    if verbose
    fprintf('it: %d, lambda %d, diff %d\n',count,lambda, sum(or(abs(weight.mu) > (mu_prev+PCT*mu_prev),abs(weight.mu) < (mu_prev-PCT*mu_prev))));
    end
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
        while sum(or(abs(weight.mat(:,i)) > (w_prev+PCT*w_prev),abs(weight.mat(:,i)) < (w_prev-PCT*w_prev)))>0 && count2<MAX_ITERATIONS
            
            w_prev = abs(weight.mat(:,i));
            
            %update W with old alphas
            aX_s=zeros(ntrials,features);
            for j = 1:ntrials
                aX_s(j,:)=alpha.mat(:,i)'*reshape(X{i}(:,:,j),chans,features);
            end
            % update W
            weight.mat(:,i)=((1/lambda)*weight.sigma*(aX_s'*aX_s)+eye(size(aX_s,2)))\...
                ((1/lambda)*weight.sigma*aX_s'*y{i}+weight.mu);
            
            % update alpha with old W
            wX_s=zeros(chans,ntrials);
            for j=1:ntrials
                wX_s(:,j)=reshape(X{i}(:,:,j),chans,features)*weight.mat(:,i);
            end
            % update alpha
            alpha.mat(:,i)=((1/lambda)*alpha.sigma*(wX_s*wX_s')+eye(size(wX_s,1)))\...
                ((1/lambda)*alpha.sigma*wX_s*y{i}+alpha.mu);
            
            % EXPERIMENTAL--norm alpha to 1
            alpha.mat(:,i)=alpha.mat(:,i)/norm(alpha.mat(:,i));
            
            
            count2=count2+1;
            if ~mod(count2,100) && verbose
                fprintf('subj %d, inside: %d, diff %d\n',i,count2,sum(or(abs(weight.mat(:,i)) > (w_prev+PCT*w_prev),abs(weight.mat(:,i)) < (w_prev-PCT*w_prev))));
            end
        end
    end
    
    %%%%%%%%%%%%%%%
    % mu/sigma update
    %%%%%%%%%%%%%%%
    
    %if only one subject, break
    if subjects == 1 && ~lambdaML
        break
    elseif subjects == 1 && lambdaML
        res=0;
        for j = 1:size(X{1},3)
            res=res+(y{1}(j)-(alpha.mat'*squeeze(X{1}(:,:,j))*weight.mat))^2;
        end
        ntrials=size(X{1},3);
        lambda=res/ntrials;
        % If iterating make the loop until lambda stabilizes
        if lambda + 0.01*lambda < lambda_prev || lambda - 0.01*lambda > lambda_prev
            mu_prev=ones(length(out.sigma),1);
        end
    else
        weight.mu=mean(weight.mat,2);
        alpha.mu=mean(alpha.mat,2);
        
        %update Sigma
        temp=weight.mat-repmat(weight.mu,1,subjects);
        atemp=alpha.mat-repmat(alpha.mu,1,subjects);
        
        % adaptive eta **separate for weight/alpha??
        if itEta
            e = eig((1/(size(temp,2)-1))*(temp*temp'));
            eta=0.1*min(abs(e));
        end
        %     % standard ML sigma update
        %     weight.sigma= (1/(size(temp,2)-1))*(temp*temp')+ eta*eye(length(weight.sigma));
        %     alpha.sigma= (1/(size(atemp,2)-1))*(atemp*atemp')+ eta*eye(length(alpha.sigma));
        % trace-normalized sigma update
        weight.sigma= (1/trace(temp*temp'))*(temp*temp')+ eta*eye(length(weight.sigma));
        alpha.sigma= (1/trace(atemp*atemp'))*(atemp*atemp')+ eta*eye(length(alpha.sigma));
        
        if lambdaML
            res=0;
            ntrials=0;
            for i = 1:subjects
                for j = 1:size(X{i},3)
                    res=res+(y{i}(j)-(alpha.mat(:,i)'*squeeze(X{i}(:,:,j))*weight.mat(:,i)))^2;
                end
                ntrials=ntrials+size(X{i},3);
            end
            lambda=res/ntrials;
        end
    end
    

    count=count+1;
    
end
out.weight=weight;
out.alpha=alpha;
out.lambda=lambda;
if count == MAX_ITERATIONS
    warning('FailedConvergence','convergence failed')
end
end
