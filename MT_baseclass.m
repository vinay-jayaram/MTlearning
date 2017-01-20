classdef MT_baseclass < handle
    % Vinay Jayaram 10.11.16
    % Base class that implements some common methods to any
    % sort of MT learning that relies on an EM algorithm. Defines optional 
    % arguments that should be accepted by any inheriting class:
    %    n_its:                    Number of iterations of the prior computation before
    %                                exiting (default 1000)
    %
    %    lambda_ml:          ML computation of the lambda value (see
    %                               paper). Default true (cross-validation is quite long)
    %
    %    tr_adjust:             Regularize computation of the prior covariance
    %                                by the trace as in Jayaram et al. 2016. Default false
    %
    %    cv_params:         Parameters to give the cross-validation
    %                                function lambdaCV.m (also in this repo)
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
        trAdjust
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
            obj.trAdjust = invarargin(varargin,'tr_adjust');
            if isempty(obj.trAdjust)
                obj.trAdjust = 0;
            end
            obj.cvParams = invarargin(varargin,'cv_params');
            if isempty(obj.cvParams)
                obj.cvParams = {};
            end
            obj.verbose = invarargin(varargin,'verbose');
            if isempty(obj.verbose)
                obj.verbose=0;
            end
            obj.parallel = invarargin(varargin,'parallel');
            if isempty(obj.parallel)
                obj.parallel = 0;
            end
            obj.prior = struct();
            obj.lambda = 1;
            obj.eta = 1e-3; % Should this be editable?
        end
        
        %%%%%%%%%%%%%%%%%%%%%%
        %            Update functions
        %%%%%%%%%%%%%%%%%%%%%%
        
        function prior = fit_prior(childObj, Xcell, ycell)
            % Function to fit prior given another class that defines loss
            % and parameter estimation steps
            %
            % Output:
            %   prior: struct with final prior values.
            
            
            % initialize 
            childObj.lambda = 1;
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
            
            % If no ML, cross-validate for appropriate lambda value
            if ~childObj.lambdaML
                % loss of interest is empirical rest loss averaged over all
                % datasets. 
                error('Currently only supports ML estimation of the lambda value for prior calculation (it works better anyway)');
            end
            
            % loop to iterate update steps 
            while its < childObj.nIts
                if childObj.verbose
                    fprintf('MT Iteration %d...\n', its);
                end
                prev_prior = childObj.prior;
                if childObj.parallel
                    parfor i = 1:length(Xcell)
                        [outputs{i}, error(i)] = childObj.fit_model(Xcell{i}, ycell{i}, childObj.lambda);
                    end
                else
                for i = 1: length(Xcell)
                    [outputs{i}, error(i)] = childObj.fit_model(Xcell{i}, ycell{i}, childObj.lambda);
                end
                end
                childObj.update_lambda(error);
                childObj.update_prior(outputs);
                its = its + 1;
                [convergence, num] = childObj.convergence(childObj.prior, prev_prior);
                if convergence
                    break
                end
                if childObj.verbose
                fprintf('[MT prior] iteration %d, remaining %d\n', its, num);
                end
            end
            prior = childObj.prior;
        end
        
        function [] = update_prior(obj, outputCell)
            W = cat(2,outputCell{:});
            obj.prior = MT_baseclass.update_gaussian_prior(W, obj.trAdjust);
            
            % Set mean weights as new model weights
            obj.w = obj.prior.mu;
        end
        
        function [] = update_lambda(obj,err)
            % Updates lambda with ML formulation if flag is set
            if obj.lambdaML
                obj.lambda = 2*mean(err); % ...i *think* this is right
                if obj.verbose
                fprintf('lambda: %.2e\n', obj.lambda);
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
            fprintf('[MT base] trace-adjusted covariance update: %d\n',obj.trAdjust);
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
        
        function prior_struct = update_gaussian_prior(M, trAdjust)
            % Function that updates gaussian prior given samples and trace
            % adjust flag
            prior_struct.mu = mean(M,2);
            temp = M - repmat(prior_struct.mu,1, size(M,2));
            
            if trAdjust
                % Trace-normalized update
                C = (1/trace(temp*temp'))*(temp*temp');
            else
                % standard ML covariance update
                C = (1/(size(temp,2)-1))*(temp*temp');
            end
            
            % regularize as necessary
            if rank(C) < size(C,1)
                e = eig((1/(size(temp,2)-1))*(temp*temp'));
                if ~sum(e>0)
                    eta = 1;
                else
                    eta = min(e(e>0));
                end
                C = C + eta*eye(size(C,1));
            end
            
            prior_struct.sigma = C;
            if exist('eta','var')
                prior_struct.eta = eta;
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
            
    end
end



