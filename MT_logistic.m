classdef MT_logistic < MT_linear
        % Logistic regression that inherits from MT_linear, which inherits
        % prior computation code from MT_baseclass.
    % and implements functions to fit new models given the prior
    % distribution. Accepts the same arguments as MT_linear (reproduced below):
    %
   %      dim_reduce:      Use PCA for cross-subject dimensionality
   %                               reduction (default false)
   %
   %      max_it_var:        Maximum percentage of variation between
   %                                iterations allowed before convergence (default 1%)
   %
   %      max_pct_var:     Maximum number of dimensions allowed to be
   %                                 unconverged before algorithm exits (default 1%)
   %
   %      prior_init_val:     Value with which to initialize prior mean
   %                                (default 0)
    methods
        function obj = MT_logistic(d, varargin)
            % Constructor for multitask linear regression.
            %
            % Input:
            %     varargin: Flags
            
            % construct superclass
            obj@MT_linear(d, varargin{:})
        end
        
        function prior = fit_prior(obj, Xcell, ycell, varargin)
            % sanity checks
            assert(length(Xcell) == length(ycell), 'unequal data and labels arrays');
            assert(length(Xcell) > 1, 'only one dataset provided');
            for i = 1:length(Xcell)
                assert(size(Xcell{i},2) == length(ycell{i}), 'number of datapoints and labels differ');
                ycell{i} = reshape(ycell{i},[],1);
            end
            
            lambda = invarargin(varargin,'lambda');
            if isempty(lambda)
                lambda = NaN;
            end
            
            assert(length(unique(cat(1,ycell{:}))) == 2, 'more than two classes present in the data');
            if isempty(obj.labels)
                obj.labels = [unique(cat(1,ycell{:})),[1;0]];
            end
            % replace labels with {1,0} for algorithm
            for i = 1:length(ycell)
                ycell{i} = MT_baseclass.swap_labels(ycell{i}, obj.labels, 'to');
            end
            obj.init_prior(size(Xcell{1},1),1);
            prior = fit_prior@MT_linear(obj, Xcell, ycell, lambda);
        end
        
        function [w, error] = fit_model(obj, X, y, lambda)
            % Perform gradient descent based minimization of logistic regression
            % with robust error-adaptive learning rate.
            
            % Setup learning parameters
            eta = 0.1;
            inc_rate = 0.1;
            dec_rate = 0.025;
            max_iter = 10000;
            % Initialize weights and compute initial error
            w = obj.prior.mu;
            ce_curr =  crossentropy_loss(obj.logistic_func(X, w), y);
            % Run gradient descent until convergence or max_iter is reached
            for iter = 1:max_iter
                % Backup previous state
                w_prev = w;
                ce_prev = ce_curr;
                % Perform gradient descent step on spectral and spatial weights
                grad_w = obj.crossentropy_grad(X, y, w, lambda);
                w = w - eta .* grad_w;
                % Check for convergence
                ce_curr = crossentropy_loss(obj.logistic_func(X, w), y);
                diff_ce = abs(ce_prev - ce_curr);
                if diff_ce < 1e-4
                    break;
                end
                % Adapt learning rate
                if ce_curr >= ce_prev
                    % Decrease learning rate and withdraw iteration
                    eta = dec_rate*eta;
                    w = w_prev;
                    ce_curr = ce_prev;
                else
                    % Increase learning rate additively
                    eta = eta + inc_rate*eta;
                end
            end
            error = ce_curr;
            obj.w = w;
        end
              
        function grad = crossentropy_grad(obj, X, y, w, lam)
            pred = MT_logistic.logistic_func(X, w);
            % Compute plain crossentropy gradient
            grad = sum(repmat(pred - y, 1, length(w)).*X', 1)';
            % Add regularization term (avoiding inversion of the covariance prior)
            grad = obj.prior.sigma * grad + lam*(w - obj.prior.mu);
        end

    end
    
    methods(Static)
        function L = loss(w, X, y)
            L = crossentropy_loss(MT_logistic.logistic_func(X, w), y);
        end
        
        function h = logistic_func(X, w)
        %FD_LOGISTIC_FUNC Bilinear version of the logistic sigmoid function
            h = 1.0 ./ (1 + exp(-X'*w));
        end
        function y = predict(w, X, labels)
            pred = MT_logistic.logistic_func(X, w);
            y = MT_baseclass.swap_labels(pred > 0.5, labels, 'from');
        end        
    end
end