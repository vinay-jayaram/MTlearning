classdef MT_linear < MT_baseclass
    
    properties(GetAccess = 'public', SetAccess = 'private')
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
        function obj = MT_linear(varargin)
            % Constructor for multitask linear regression. 
            %
            % Input:
            %     Xcell: cell array of datasets
            %     ycell: cell array of labels
            %     varargin: Flags 
        
            nIts = invarargin(varargin,'n_its');
            if isempty(nIts)
                nIts = 1000;
            end
            lambdaML = invarargin(varargin,'lambda_ml');
            if isempty(lambdaML)
                lambdaML = 1;
            end
            etaML = invarargin(varargin,'eta_ml');
            if isempty(etaML)
                etaML = 0;
            end
            trAdjust = invarargin(varargin,'tr_adjust');
            if isempty(trAdjust)
                trAdjust = 0;
            end
            cvParams = invarargin(varargin,'cv_params');
            if isempty(cvParams)
                cvParams = {};
            end

            % construct superclass
            obj@MT_baseclass(nIts, lambdaML, etaML, trAdjust, cvParams)
            
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
            
        end
        
        function [] = init_prior(obj)
            obj.prior.mu = zeros(size(obj.w));
            obj.prior.sigma = eye(length(obj.w));
        end
        
        function prior = fit_prior(obj, Xcell, ycell)
            % sanity checks
            assert(length(Xcell) == length(ycell), 'unequal data and labels arrays');
            assert(length(Xcell) > 1, 'only one dataset provided');
            for i = 1:length(Xcell)
                assert(size(Xcell{i},2) == length(ycell{i}), 'number of datapoints and labels differ');
                ycell{i} = reshape(ycell{i},[],1);
            end
            
            
            assert(length(unique(cat(1,ycell{:}))) == 2, 'more than two classes present in the data');
            obj.labels = [unique(cat(1,ycell{:})),[1;-1]];
            % replace labels with {1,-1} for algorithm
            for i = 1:length(ycell)
                tmp = zeros(size(ycell{i}));
                for j = 1:2
                    tmp(ycell{i}==obj.labels(j,1))=obj.labels(j,2);
                end
                ycell{i} = tmp;
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
                obj.w = zeros(size(obj.W,1),1);
            else
                obj.W = [];
            end
            prior = fit_prior@MT_baseclass(obj, Xcell, ycell);
        end
        
        function b = convergence(obj, prior, prev_prior)
            mu = abs(prior.mu);
            mu_prev = abs(prev_prior.mu);
            converged = sum(or(mu > (mu_prev+obj.maxItVar*mu_prev), mu < (mu_prev - obj.maxItVar * mu_prev)));
            b = converged < (obj.maxNumVar * length(mu));
        end
        
        function [w, error] = fit_model(obj, X, y, lambda)
            Ax=obj.prior.sigma*X;
            w = ((1 / lambda)*Ax*X'+eye(size(X,1)))\((1 / lambda)*Ax*y + obj.prior.mu);
            error = obj.loss(w, X, y);
        end
        
        function out = fit_new_task(obj, X, y, varargin)
            % argument parsing
            
            ML = invarargin(varargin,'ml');
            if isempty(ML)
                ML = 0;
            end
            out = struct();

            if ML
                prev_loss = 0;
                out.lambda = 1;
                out.loss = 1;
                count = 0;
                while abs(prev_loss - out.loss) > obj.maxItVar * prev_loss && count < obj.nIts
                    [out.w, out.loss] = obj.fit_model(X, y, out.lambda);
                    out.lambda = out.loss;
                    count = count+1;
                    fprintf('[new task fitting] ML lambda Iteration %d, lambda %.2f \n', count, out.lambda);
                end
            else
                out.lambda = lambdaCV(@(X,y,lambda)(obj.fit_model(X{1},y{1},lambda)),...
                    @(w, X, y)(obj.loss(w, X{1}, y{1})),{X},{y});
                [out.w, out.loss] = obj.fit_model(X, y, out.lambda);
            end
            
            out.predict = @(X)(obj.predict(out.w, X, obj.labels));
            out.training_acc = mean(y == out.predict(X));
        end
        
        function [] = update_prior(obj, outputCell)
            W = cat(2,outputCell{:});
            obj.prior.mu = mean(W,2);
            temp = W - repmat(obj.prior.mu,1,length(outputCell));
            if obj.trAdjust
                % Trace-normalized update
                C = (1/trace(temp*temp'))*(temp*temp');
            else
                % standard ML covariance update
                 C = (1/(size(temp,2)-1))*(temp*temp');
            end
            
            % regularize as necessary
            if rank(C) < size(C,1)
                e = eig((1/(size(temp,2)-1))*(temp*temp'));
                C = C + mean(e(e>0))*eye(size(C,1));
            end
            
            obj.prior.sigma = C;
        end
        
    end
    
    methods(Static)
        function L = loss(w, X, y)
            % implements straight squared loss
            L = norm(X'*w-y,2);
        end
        
        function y = predict(w, X, labels)
            y = sign(X'*w);
            for i = 1:2
                y(y == labels(i,2)) = labels(i,1);
            end
        end
        
    end
end