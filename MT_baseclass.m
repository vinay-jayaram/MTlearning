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
        trAdjust
    end
    
    methods
        
        %%%%%%%%%%%%%%%%%%%%%%
        %              Initialization
        %%%%%%%%%%%%%%%%%%%%%%
        
        function obj = MT_baseclass(nIts, lambaML, trAdjust, cvParams)
            % Constructor to initialize a MT model.
            %
            % Output:
            %    obj: This instance.
            
            obj.prior = struct();
            obj.lambda = 1;
            obj.eta = 1e-3; % Should this be editable?
            obj.nIts = nIts;
            obj.lambdaML = lambaML;
            obj.trAdjust = trAdjust;
            obj.cvParams = cvParams;
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
            
            
            % initialize prior
            childObj.init_prior();
            childObj.lambda = 1;
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
                childObj.update_lambda(error);
                childObj.update_prior(outputs);
                its = its + 1;
                [convergence, num] = childObj.convergence(childObj.prior, prev_prior);
                if convergence
                    break
                end
                fprintf('[MT prior] iteration %d, remaining %d\n', its, num);
            end
            
            prior = childObj.prior;
            
        end
        
        function [] = update_lambda(obj,err)
            if obj.lambdaML
                obj.lambda = 2*mean(err); % ...i *think* this is right
                fprintf('lambda: %.2e\n', obj.lambda);
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
                C = C + min(e(e>0))*eye(size(C,1));
            end
            
            prior_struct.sigma = C;
        end
        
    end
end

