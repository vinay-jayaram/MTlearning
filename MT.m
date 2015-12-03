function [out] = MT(X,y,varargin)
%%
%Function to compute the original multi-task learning approach (Alamgir 2010). Takes:
%    X: cell array of data, each element is a (features x trials) array
%    y: Cell array of labels, each element is (trials x 1)
%    lambda: controls ratio between bias and data-dependent solution, between 0 and 1
%
% Outputs:
%   out: Struct with learned mu and sigma, as well as all learned discrimination vectors
%   in the (features x subjects) array mat
%optional arguments:
%
%   'eta':          Give number as the eta for use in covariance calculation
%   'prior':      {mean,cov} cell array if one wants to bias calculations.
%                    Default is zeros and I
%   'lambda':    Define external lambda and don't use ML update
%   'verbose':   Print convergence information per iteration

%Note: If 'lambda' is a single value, we assume a single lambda for the
% all datasets. If lambda is a vector then it is assumed to be the
% dataset-specific lambdas, and if it does not match the length of X an
% error is thrown.

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

priors = invarargin(varargin,'prior');
if isempty(priors)
    priors{1}=zeros(size(X{1},1),1);
    priors{2}=eye(size(X{1},1));
end

inlambda=invarargin(varargin,'lambda');
if isempty(inlambda)
    lambdaML=1;
    lambda=1;
else
    lambda=inlambda;
    lambdaML=0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variable Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda_prev=0;
out.sigma=priors{2};
out.mu=priors{1};
subjects=length(X);
out.mat=zeros(length(out.sigma),subjects);
mu_prev=ones(length(out.sigma),1);
count=1;
MAX_ITERATIONS=5000;

% Defining the bounds of convergence; more important in cases with low data
% when some dimensions do not converge. The current is demanding that 99%
% of dimensions vary less than 1% over two iterations
max_vary=floor(0.01*length(out.mu));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while sum(or(abs(out.mu) > (mu_prev+0.01*mu_prev),abs(out.mu) < (mu_prev-0.01*mu_prev)))>max_vary && count < MAX_ITERATIONS
    if verbose
        fprintf('it: %d, lambda %d, diff %d\n',count,lambda, sum(or(abs(out.mu) > (mu_prev+0.01*mu_prev),abs(out.mu) < (mu_prev-0.01*mu_prev))));
    end
    mu_prev = abs(out.mu);
    lambda_prev=lambda;
    %update W
    for i = 1:subjects
        Ax=out.sigma*X{i};
        out.mat(:,i)=((1/lambda)*Ax*X{i}'+eye(size(X{i},1)))\((1/lambda)*Ax*y{i}+out.mu);
    end
    
    % if only one subject, break to make ridge regression
    if subjects == 1 && ~lambdaML
        break
    elseif subjects == 1 && lambdaML
        lambda=2*sum((y{1}'-out.mat'*X{1}).^2)/size(X{1},2);
        % if iterating make the loop until lambda stabilizes
        if lambda + 0.01*lambda < lambda_prev || lambda - 0.01*lambda > lambda_prev
            mu_prev=ones(length(out.sigma),1);
        end
    else
        %update mu
        out.mu=mean(out.mat,2);
        
        % de-mean data for covariance calculation
        temp=out.mat-repmat(out.mu,1,subjects);
        
        % adaptive eta
        if itEta
            e = eig((1/(size(temp,2)-1))*(temp*temp'));
            eta=0.1*min(abs(e));
        end
        
        % standard ML covariance update
        out.sigma= (1/(size(temp,2)-1))*(temp*temp')+ eta*eye(length(out.sigma));
        
        %     % Trace-normalized update
        %         out.sigma=(1/trace(temp*temp'))*(temp*temp')+eta*eye(length(out.sigma));
        % update lambda
        if lambdaML
            res=0;
            ntrials=0;
            for i = 1:subjects
                res=res+sum((y{i}'-out.mat(:,i)'*X{i}).^2);
                ntrials=ntrials+size(X{i},2);
            end
            lambda=2*res/ntrials;
        end
    end
    
    
    
    
    % Increment counter
    count=count+1;
end
out.lambda=lambda;
if count == MAX_ITERATIONS
    warning('FailedConvergence','convergence failed')
end
end
