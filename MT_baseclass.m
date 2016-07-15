classdef MT_baseclass < handle
    % Base class that implements some common methods to any
    % sort of MT learning
    
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
        etaML
        trAdjust
    end
    
    methods
        function obj = MT_baseclass(nIts, lambaML, etaML, trAdjust, cvParams)
            % Constructor to initialize a MT model.
            %
            % Output:
            %    obj: This instance.
            
            obj.prior = struct();
            obj.lambda = 1;
            obj.eta = 1e-3; % Should this be editable?
            obj.nIts = nIts;
            obj.lambdaML = lambaML;
            obj.etaML = etaML;
            obj.trAdjust = trAdjust;
            obj.cvParams = cvParams;
        end
        
        function prior = fit_prior(childObj, Xcell, ycell)
            % Function to fit prior given another class that defines loss
            % and parameter estimation steps
            %
            % Output:
            %   superObj: This instance
            
            
            % initialize prior
            childObj.init_prior();
            its = 0;
            error = zeros(length(Xcell),1);
            outputs = cell(length(Xcell),1);
            
            % If no ML, cross-validate for appropriate lambda value
            if ~childObj.lambdaML
                error('Currently only supports ML estimation of the lambda value for prior calculation (it works better anyway)');
            end
            
            % loop to iterate update steps 
            while its < childObj.nIts
                prev_prior = childObj.prior;
                for i = 1: length(Xcell)
                    [outputs{i}, error(i)] = childObj.fit_model(Xcell{i}, ycell{i}, childObj.lambda);
                end
                childObj.update_prior(outputs);
                childObj.update_lambda(error);
                childObj.update_eta();
                its = its + 1;
                if childObj.convergence(childObj.prior, prev_prior)
                    break
                end
                disp(its);
            end
            
            prior = childObj.prior;
            
        end
        
        function [] = update_lambda(obj,err)
            if obj.lambdaML
                obj.lambda = mean(err); % ...i *think* this is right
            end
        end
        
        function [] = update_eta(obj,eta)
            if obj.etaML
                obj.eta = eta;
            end
        end

        
        function [] = printswitches(obj)
            % function to print all the options for this implementation
            fprintf('[MT base] number of iterations: %d\n',obj.nIts);
            fprintf('[MT base] ML estimation of lambda: %d\n',obj.lambdaML);
            fprintf('[MT base] iterative eta update: %d\n',obj.etaML);
            fprintf('[MT base] trace-adjusted covariance update: %d\n',obj.trAdjust);
        end
        
        function [] = printprior(obj)
            print(obj.prior);
        end
        
    end
    
end

