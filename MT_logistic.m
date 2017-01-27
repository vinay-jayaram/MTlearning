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
            % specify that internal labels are 1/0 and not 1/-1
            obj.classid = [1;0];
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