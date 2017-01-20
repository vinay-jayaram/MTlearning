classdef MT_linear < MT_baseclass
    % Base class for models that inherit from MT_baseclass and perform
    % classification. Inherits prior computation code from MT_baseclass
    % and implements functions to fit new models given the prior
    % distribution. Accepts the following arguments *in addition*
    % to those accepted by MT_baseclass:
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
    %
    %
    properties(GetAccess = 'public', SetAccess = 'public')
        % vector that contains unique labels for prediction
        labels
        % optional dimensionality reduction matrix
        W
        % weight vector for classification
        w
        % binary flag for dimensionality reduction
        dimReduce
        % binary flag for LDA labelling
        %         LDA
        % parameters for convergence
        maxItVar % maximum variation between iterations before convergence
        maxNumVar % maximum number of dimensions allowed to not converge
    end
    
    methods
        function obj = MT_linear(d, varargin)
            % Constructor for multitask linear regression.
            %
            % Input:
            %     d           : Dimension of data in order to construct
            %     prior
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
            obj.labels = [];
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
            if isempty(obj.labels)
                obj.labels = [unique(cat(1,ycell{:})),[1;-1]];
            end
            % replace labels with {1,-1} for algorithm
            for i = 1:length(ycell)
                ycell{i} = MT_baseclass.swap_labels(ycell{i}, obj.labels, 'to');
            end
            obj.w = zeros(size(Xcell{1},1),1);
            
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
                obj.init_prior(size(obj.W,2),0);
            else
                obj.W = [];
                % obj.w was already initialized
                obj.init_prior(size(Xcell{1},1),0);
            end
            prior = fit_prior@MT_baseclass(obj, Xcell, ycell);
        end
        
        function [b, converged] = convergence(obj, prior, prev_prior)
            mu = abs(prior.mu);
            mu_prev = abs(prev_prior.mu);
            converged = sum(or(mu > (mu_prev+obj.maxItVar*mu_prev), mu < (mu_prev - obj.maxItVar * mu_prev)));
            b = converged <= (obj.maxNumVar * length(mu));
        end
        
        function [w, error] = fit_model(obj, X, y, lambda)
            Ax=obj.prior.sigma*X;
            w = ((1 / lambda)*Ax*X'+eye(size(X,1)))\((1 / lambda)*Ax*y + obj.prior.mu);
            error = obj.loss(w, X, y);
            obj.w = w;
        end
        
        function out = fit_new_task(obj, X, y, varargin)
            if obj.dimReduce
                assert(size(X, 1) == size(obj.W,1), ...
                    'Feature dimensionality of the data does not match this model');
            else
                assert(size(X, 1) == length(obj.prior.mu), ...
                    'Feature dimensionality of the data does not match this model');
            end
            % argument parsing
            
            ML = invarargin(varargin,'ml');
            if isempty(ML)
                ML = 0;
            end
            out = struct();
            Xoriginal = X;
            if obj.dimReduce
                X = obj.W'*X;
            end
            
            % switch input labels using instance dictionary
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
            out.training_acc = mean(y == out.predict(Xoriginal));
        end
       
        
        function y = prior_predict(obj, X, varargin)
            labels = invarargin(varargin,'labels');
            if isempty(labels)
                labels = obj.labels;
            end
            if isempty(labels)
                error('Model has not yet been trained');
            end
            if obj.dimReduce
                X = obj.W'*X;
            end
            y = obj.predict(obj.prior.mu, X, labels);
        end
        
    end
    
    methods(Static)
        function L = loss(w, X, y)
            % implements straight (average) squared loss
            L = (norm(X'*w-y,2)^2)/length(y);
        end
        function y = predict(w, X, labels)
            y = MT_baseclass.swap_labels(sign(X'*w), labels, 'from');
        end
    end
end