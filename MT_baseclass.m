classdef MT_baseclass < handle
    % Vinay Jayaram 10.11.16
    % Base class that implements some common methods to any
    % sort of MT learning that relies on an EM algorithm. Defines optional
    % arguments that should be accepted by any inheriting class (true/false are set with [1,0]):
    %    n_its:                    Number of iterations of the prior computation before
    %                                exiting (default 1000)
    %
    %    lambda_ml:          ML computation of the lambda value (see
    %                               paper). Default true (cross-validation is quite long)
    %
    %    zero_mean:         Binary. Force the prior mean to be zero for a
    %                               shrinkage effect. Independent of other parameters that affect the
    %                               prior
    %    cov_flag:             Flag that sets how the prior covariance is
    %                               computed. Current possibilities are {'l2' [default], 'l2-trace',
    %                               'l1', 'l1-diag'}
    %    cv_params:         Parameters to give the cross-validation
    %                                function lambdaCV.m. Input as cell array e.g. 
    %                                {'flag 1',val1, 'flag 2', val2...} (For possible parameters, run 'help lambdaCV')
    %
    %    verbose:              Output debugging information. Default false
    %
    %    parallel:                Attempts to start a parpool if one does
    %                                not exist and parallelizes computation of the individual datasets.
    %                                Default 0 (does not execute); can take integer arguments for
    %                                number of workers
    
    properties(Access = 'protected')
        % struct to contain priors
        prior
        % constants
        lambda
        eta
        % CV parameters
        cvParams
        % and switches
        nIts
        lambdaML
        zeroMean
        covFlag
        verbose
        parallel %0 or number of workers. Does not delete pool.
        varargin % used to allow for deep copy
    end
    
    methods
        
        %%%%%%%%%%%%%%%%%%%%%%
        %              Initialization
        %%%%%%%%%%%%%%%%%%%%%%
        
        function obj = MT_baseclass(varargin)
            % Constructor to initialize a MT model.
            %
            % Output:
            %    obj: This instance.
            obj.varargin = varargin;
            obj.nIts = invarargin(varargin,'n_its');
            if isempty(obj.nIts)
                obj.nIts = 1000;
            end
            obj.lambdaML = invarargin(varargin,'lambda_ml');
            if isempty(obj.lambdaML)
                obj.lambdaML = 1;
            end
            obj.zeroMean = invarargin(varargin,'zero_mean');
            if isempty(obj.zeroMean)
                obj.zeroMean = 0;
            end
            obj.covFlag = invarargin(varargin,'cov_flag');
            if isempty(obj.covFlag)
                obj.covFlag = 'l2';
            else
                switch obj.covFlag
                    case {'l1','l1-diag'}
                        disp('L1 norm chosen, prior mean set to zero');
                        obj.zeroMean = 1;
                end
                        
            end
            obj.cvParams = invarargin(varargin,'cv_params');
            if isempty(obj.cvParams)
                obj.cvParams = {};
            end
            obj.verbose = invarargin(varargin,'verbose');
            if isempty(obj.verbose)
                obj.verbose=0;
            else
                if obj.verbose ~= 0
                    obj.cvParams{end+1} = 'verbose';
                    obj.cvParams{end+1} = 1;
                end
            end
            obj.parallel = invarargin(varargin,'parallel');
            if isempty(obj.parallel)
                obj.parallel = 0;
            else
                fprintf('[MT base] Attempting parallel implementation with %d cores\n',obj.parallel);
            end
            obj.prior = struct();
            obj.prior.lambda = 1;
            obj.eta = 1e-3; % Should this be editable?
        end
        
        %%%%%%%%%%%%%%%%%%%%%%
        %            Update functions
        %%%%%%%%%%%%%%%%%%%%%%
        
        function prior = fit_prior(childObj, Xcell, ycell, varargin)
            % Function to fit prior given another class that defines loss
            % and parameter estimation steps. If given a nonsense lambda
            % value
            %
            % Output:
            %   prior: struct with final prior values.
            
            lambda = invarargin(varargin, 'lambda');
            
            its = 0;
            error = zeros(length(Xcell),1);
            outputs = cell(length(Xcell),1);
            
            % if possible initialize parallel pool
            if childObj.parallel
                p = gcp('nocreate');
                if isempty(p)
                    try
                        p = parpool(childObj.parallel);
                    catch Ex
                        error(Ex.message);
                    end
                end
            end
            
            childObj.prior.lambda = lambda;
            if isnan(childObj.prior.lambda)
                if childObj.lambdaML
                    disp('No lambda value given. Using maximum-likelihood estimation of lambda parameter.');
                    childObj.prior.lambda = 1;
                    updateLambda = 1;
                else
                    % if no ML and no lambda given then cross-validate (is
                    % recursive...)
                    updateLambda = 0;
                    disp('No lambda value given. Using cross-validation to estimate.');
                    childObj.prior.lambda = lambdaCV(@(X,y,lambda)(childObj.multi_task_f(childObj,X,y,lambda)),...
                        @(W, X, y)(childObj.multi_task_loss(childObj,W,X,y)),Xcell,ycell,childObj.cvParams{:});
                    
                end
            else
                updateLambda = 0;
            end
            
            % if lambda specificed we just need to run the optimization
            % once. If not, we do more.
            % loop to iterate update steps
            while its < childObj.nIts
                prev_prior = childObj.prior;
                if childObj.parallel
                    parfor i = 1:length(Xcell)
                        [outputs{i}, error(i)] = childObj.fit_model(Xcell{i}, ycell{i}, childObj.prior.lambda);
                    end
                else
                    for i = 1: length(Xcell)
                        [outputs{i}, error(i)] = childObj.fit_model(Xcell{i}, ycell{i}, childObj.prior.lambda);
                    end
                end
                if updateLambda
                    childObj.update_lambda(error);
                end
                tmp = childObj.prior.lambda;
                childObj.update_prior(outputs);
                childObj.prior.lambda = tmp;
                its = its + 1;
                [convergence, num] = childObj.convergence(childObj.prior, prev_prior);
                if convergence
                    if childObj.verbose
                    fprintf('[MT prior] Iteration %d, converged, error %d\n', its, num);
                    end
                    break
                else
                     if childObj.verbose
                    fprintf('[MT prior] Iteration %d, error %d\n', its, num);
                    end                   
                end

            end
            prior = childObj.prior;
            
        end
        
        function [] = update_prior(obj, outputCell)
            W = cat(2,outputCell{:});
            obj.prior = MT_baseclass.update_gaussian_prior(W, obj.zeroMean, obj.covFlag);
            
            % Set mean of weights as new model weights
            obj.w = mean(obj.prior.W,2);
        end
        
        function [] = update_lambda(obj,err)
            % Updates lambda with ML formulation if flag is set
            if obj.lambdaML
                obj.prior.lambda = 2*mean(err); % ...i *think* this is right
                if obj.verbose
                    fprintf('lambda: %.2e\n', obj.prior.lambda);
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%
        %         Printing  and access functions
        %%%%%%%%%%%%%%%%%%%%%%
        
        function [] = printswitches(obj)
            % function to print all the options for this implementation
            fprintf('[MT base] number of iterations: %d\n',obj.nIts);
            fprintf('[MT base] ML estimation of lambda: %d\n',obj.lambdaML);
            fprintf('[MT base] Covariance flag: %s\n',obj.covFlag);
            fprintf('[MT base] Restrict prior mean to zero: %d\n', obj.zeroMean);
            fprintf('[MT base] Verbose: %d\n',obj.verbose);
            fprintf('[MT base] Parallel: %d\n',obj.parallel);
        end
        
        
        
        function [] = printprior(obj)
            print(obj.prior);
        end
        
        function P = getprior(obj)
            P = obj.prior;
        end
        
        function [] = setprior(obj,P)
            obj.prior = P;
        end
             
    end
    
    methods(Static)
        %%%%%%%%%%%%%%%%%%%%%%
        %                       Generic helper functions
        %%%%%%%%%%%%%%%%%%%%%%
        
        function prior_struct = update_gaussian_prior(M, zeromean, flag)
            % Function that updates gaussian prior given samples and trace
            % adjust flag
            prior_struct.W = M;
            prior_struct.mu = mean(M,2);
            temp = M - repmat(prior_struct.mu,1, size(M,2));
            
            % compute eta beforehand
            %
            %             % regularize as necessary
            %             if rank(C) < size(C,1)
            e = eig((1/(size(temp,2)-1))*(temp*temp'));
            if ~sum(e>0)
                eta = 1;
            else
                eta = abs(min(e(e>0)));
            end
            
            
            switch flag
                case 'l2'
                    % standard ML covariance update
                    C = (1/(size(temp,2)-1))*(temp*temp');
                case 'l2-trace'
                    % Trace-normalized update
                    C = (1/trace(temp*temp'))*(temp*temp');
                case 'l1'
                    % Trace-normalized square root update
                    eta = 1e-4;
                    D = sqrtm(temp*temp' + eye(size(temp,1))*eta);
                    C = D/trace(D);
                case 'l1-diag'
                    
                    W_columns = zeros(size(temp,1),1);
                    for i = 1:length(W_columns)
                        W_columns(i) = norm(temp(i,:));
                    end
                    
                    W_21 = norm(W_columns,1);
                    C = diag(W_columns/W_21);
                otherwise
                    error('invalid covariance estimation flag given.');
             end
            
            if rank (C) < size(C,1)
                C = C + eta*eye(size(C,1));
            end
            
            prior_struct.sigma = C;
            if exist('eta','var')
                prior_struct.eta = eta;
            end
           
            if zeromean
                 prior_struct.mu = zeros(size(M,1),1);
            end
        end
        
        function y_switched = swap_labels(y, labels, forward)
            % Helper function to keep track of labels internally so
            % arbitrary labels can be given as input
            switch forward
                case 'to'
                    ind = 1;
                case 'from'
                    ind = 2;
                otherwise
                    error('last argument must be either to or from');
            end
            tmp = zeros(size(y));
            for i = 1:2
                tmp(y == labels(i,ind)) = labels(i,setdiff([1,2],ind));
            end
            y_switched = tmp;
        end
        
        function [W] = multi_task_f(obj, Xtrain, Ytrain, lambda)
            prior = obj.fit_prior(Xtrain, Ytrain, 'lambda', lambda, 'cv', 1);
            W = prior.W;
        end
        
        function [loss] = multi_task_loss(obj,W,Xtest,Ytest)
            loss = 0;
            for i = 1:length(Xtest)
                loss = loss + obj.loss(W(:,i),Xtest{i},Ytest{i});
            end
        end
        
        
    end
end


