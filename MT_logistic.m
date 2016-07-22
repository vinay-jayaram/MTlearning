classdef MT_logistic < MT_baseclass
    
    properties(GetAccess = 'public', SetAccess = 'public')
        % vector that contains unique labels for prediction
        labels
        % optional dimensionality reduction matrix
        W
        % weight vector for classification
        w
        % binary flag for dimensionality reduction
        dimReduce
        % parameters for convergence
        maxItVar % maximum variation between iterations before convergence
        maxNumVar % maximum number of dimensions allowed to not converge
    end
    
    methods
        function obj = MT_logistic(d, varargin)
            % Constructor for multitask linear regression.
            %
            % Input:
            %     varargin: Flags 

            % construct superclass
            obj@MT_baseclass(varargin{:})
            
            obj.dimReduce = invarargin(varargin, 'dim_reduce');
            if isempty(obj.dimReduce)
                obj.dimReduce = 0;
            end
            obj.maxItVar = invarargin(varargin,'max_it_var');
            if isempty(obj.maxItVar)
                obj.maxItVar = 1e-2;
            end
            obj.maxNumVar = invarargin(varargin,'max_pct_var');
            if isempty(obj.maxNumVar)
                obj.maxNumVar = 1e-2;
            end
            init_val = invarargin(varargin,'prior_init_val');
            if isempty(init_val)
                init_val = 0;
            end
            
            obj.init_prior(d, init_val);
            obj.w = obj.prior.mu;
        end
        
        function [] = init_prior(obj, d, init_val)
            obj.prior.mu = init_val*ones(d, 1);
            obj.prior.sigma = eye(d);
        end
        
        function prior = fit_prior(obj, Xcell, ycell)
            % sanity checks
            assert(length(Xcell) == length(ycell), 'unequal data and labels arrays');
            assert(length(Xcell) > 1, 'only one dataset provided');
            assert(size(Xcell{1}, 1) == length(obj.prior.mu), ...
                'Feature dimensionality of the data does not match this model');
            for i = 1:length(Xcell) 
                assert(size(Xcell{i},2) == length(ycell{i}), 'number of datapoints and labels differ');
                ycell{i} = reshape(ycell{i},[],1);
            end
            assert(length(unique(cat(1,ycell{:}))) == 2, 'more than two classes present in the data');
            
            % replace labels with {1,0} for algorithm
            obj.labels = [unique(cat(1,ycell{:})),[1;0]];
            for i = 1:length(ycell)
                ycell{i} = MT_baseclass.swap_labels(ycell{i}, obj.labels, 'to');
            end
            obj.w = zeros(size(Xcell{1},1),1);
            % Perform PCA
            if obj.dimReduce
                Xall = cat(2,Xcell{:});
                Xcov = cov((Xall-kron(mean(Xall,2),ones(1,size(Xall,2))))');
                [V,D] = eig(Xcov);
                if min(diag(D)) > 0
                    D = D / sum(sum(D));
                    V = V(:,diag(D)>1e-8);
                else
                    D2 = D(:,diag(D)>0);
                    D = D / sum(sum(D2));
                    V = V(:,diag(D)>1e-8);
                end
                obj.W = V;
                for i = 1:length(Xcell)
                    Xcell{i} = obj.W'*Xcell{i};
                end
                obj.w = zeros(size(obj.W,2),1);
            else
                obj.W = [];
            end
            
            prior = fit_prior@MT_baseclass(obj, Xcell, ycell);
        end
        
        function [b, converged] = convergence(obj, prior, prev_prior)
            mu = abs(prior.mu);
            mu_prev = abs(prev_prior.mu);
            converged = sum(or(mu > (mu_prev+obj.maxItVar*mu_prev), mu < (mu_prev - obj.maxItVar * mu_prev)));
            b = converged < (obj.maxNumVar * length(mu));
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
        
        function out = fit_new_task(obj, X, y, varargin)
            assert(size(X, 1) == length(obj.prior.mu), ...
                'Feature dimensionality of the data does not match this model');
            % argument parsing
            
            ML = invarargin(varargin,'ml');
            if isempty(ML)
                ML = 0;
            end
            out = struct();
            
            if obj.dimReduce
                X = obj.W'*X;
            end
            % switch input labels using instance dictionary
            obj.labels = [unique(cat(1,y)), [1; 0]];
            y_train = MT_baseclass.swap_labels(y, obj.labels,'to');
            if ML
                prev_w = ones(size(X,1),1);
                out.lambda = 1;
                out.loss = 1;
                count = 0;
                out.w = zeros(size(X,1),1);
                while sum(or(abs(out.w) > (prev_w+obj.maxItVar*prev_w), abs(out.w) < (prev_w - obj.maxItVar * prev_w)))...
                         && count < obj.nIts
                    prev_w = abs(out.w);
                    [out.w, out.loss] = obj.fit_model(X, y_train, out.lambda);
                    out.lambda = 2*out.loss;
                    count = count+1;
                    if obj.verbose
                    fprintf('[new task fitting] ML lambda Iteration %d, lambda %.4e \n', count, out.lambda);
                    end
                end
            else
                out.lambda = lambdaCV(@(X,y,lambda)(obj.fit_model(X{1},y{1},lambda)),...
                    @(w, X, y)(obj.loss(w, X{1}, y{1})),{X},{y_train});
                [out.w, out.loss] = obj.fit_model(X, y_train, out.lambda);
            end
            if obj.dimReduce
                out.predict = @(X)(obj.predict(out.w, obj.W'*X, obj.labels));
            else
                out.predict = @(X)(obj.predict(out.w, X, obj.labels));
            end
            out.training_acc = mean(y == out.predict(X));
        end
        
        function [] = update_prior(obj, outputCell)
            W = cat(2,outputCell{:});
            obj.prior = MT_baseclass.update_gaussian_prior(W, obj.trAdjust);
            
            if obj.dimReduce
                obj.prior.mu = obj.W*obj.prior.mu;
                obj.prior.sigma = obj.W*obj.prior.mu*obj.W';
            end
            % Set mean weights as new model weights
            obj.w = obj.prior.mu;
        end
        
        
        function grad = crossentropy_grad(obj, X, y, w, lam)
            pred = MT_logistic.logistic_func(X, w);
            % Compute plain crossentropy gradient
            grad = sum(repmat(pred - y, 1, length(w)).*X', 1)';
            % Add regularization term (avoiding inversion of the covariance prior)
            grad = obj.prior.sigma * grad + lam*(w - obj.prior.mu);
        end
        
        function y = predict(obj, X, varargin)
            labels = invarargin(varargin, 'labels');
            if isempty(labels)
                labels = obj.labels;
            else
                labels = [labels,[0;1]];
            end
            y = MT_baseclass.swap_labels(sign(X'*obj.w), labels, 'from');
            pred = MT_logistic.logistic_func(X, obj.w);
            y = MT_baseclass.swap_labels(pred > 0.5, labels, 'from');
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
        
    end
end