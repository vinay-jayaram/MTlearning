classdef MT_FD_linear < MT_baseclass
    
    properties(GetAccess = 'public', SetAccess = 'private')
        % vector that contains unique labels for prediction
        labels
        % weight vector for classification
        w
        a
        % parameters for convergence
        maxItVar % maximum variation between iterations before convergence
        maxNumVar % maximum number of dimensions allowed to not converge
    end
    
    methods
        function obj = MT_FD_linear(varargin)
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
            obj.prior.alpha.mu = zeros(size(obj.a));
            obj.prior.alpha.sigma = eye(length(obj.a));
            obj.prior.weight.mu = zeros(size(obj.w));
            obj.prior.weight.sigma = eye(length(obj.w));
        end
        
        function prior = fit_prior(obj, Xcell, ycell)
            % sanity checks
            assert(length(Xcell) == length(ycell), 'unequal data and labels arrays');
            assert(length(Xcell) > 1, 'only one dataset provided');
            for i = 1:length(Xcell)
                assert(size(Xcell{i},3) == length(ycell{i}), 'number of datapoints and labels differ');
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
            obj.w = zeros(size(Xcell{1},2),1);
            obj.a = zeros(size(Xcell{1},1),1);
            
            prior = fit_prior@MT_baseclass(obj, Xcell, ycell);
        end
        
        function [b, conv] = convergence(obj, prior, prev_prior)
            mu = abs(prior.weight.mu);
            mu_prev = abs(prev_prior.weight.mu);
            converged_weight = sum(or(mu > (mu_prev+obj.maxItVar*mu_prev), mu < (mu_prev - obj.maxItVar * mu_prev)));
            
            mu = abs(prior.alpha.mu);
            mu_prev = abs(prev_prior.alpha.mu);
            converged_alpha = sum(or(mu > (mu_prev+obj.maxItVar*mu_prev), mu < (mu_prev - obj.maxItVar * mu_prev)));
            b = (converged_alpha + converged_weight) < (obj.maxNumVar * (length(obj.a) + length(obj.w)));
            conv = (converged_alpha + converged_weight);
        end
        
        function [w, error] = fit_model(obj, X, y, lambda)
            features = size(X,2);
            chans = size(X,1);
            w{2} = ones(chans,1);
            w{1} = zeros(features,1); % I'm not convinced that a cell array is best here...but.....does it make sense to store things
                                                       % as rank-1 matrices
                                                       % and SVD them every
                                                       % time we want the
                                                       % components?
                                                       
            w_prev=ones(features,1);
            count2=0;
            ntrials=size(X,3);
            while sum(or(abs(w{1}) > (w_prev+obj.maxItVar*w_prev),abs(w{1}) < (w_prev-obj.maxItVar*w_prev)))>0 && count2< obj.nIts
                
                w_prev = abs(w{1});
                
                %update W with old alphas
                aX_s=zeros(ntrials,features);
                for j = 1:ntrials
                    aX_s(j,:)=w{2}'*reshape(X(:,:,j),chans,features);
                end
                % update W
                w{1}=((1/lambda)*obj.prior.weight.sigma*(aX_s'*aX_s)+eye(size(aX_s,2)))\...
                    ((1/lambda)*obj.prior.weight.sigma*aX_s'*y+obj.prior.weight.mu);
                
                % update alpha with old W
                wX_s=zeros(chans,ntrials);
                for j=1:ntrials
                    wX_s(:,j)=reshape(X(:,:,j),chans,features)*w{1};
                end
                % update alpha
                w{2}=((1/lambda)*obj.prior.alpha.sigma*(wX_s*wX_s')+eye(size(wX_s,1)))\...
                    ((1/lambda)*obj.prior.alpha.sigma*wX_s*y+obj.prior.alpha.mu);
                
                % EXPERIMENTAL--norm alpha to 1
                w{2}=w{2}/norm(w{2});

                count2=count2+1;
            end
            
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
                    prev_loss = out.loss;
                    [out.w, out.loss] = obj.fit_model(X, y, out.lambda);
                    out.lambda = 2*out.loss;
                    count = count+1;
                    fprintf('[new task fitting] ML lambda Iteration %d, lambda %.4e \n', count, out.lambda);
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
            W = [];
            A = [];
            for i = 1:length(outputCell)
                W = cat(2,W,outputCell{i}{1});
                A = cat(2,A, outputCell{i}{2});
            end
            obj.prior.weight.mu = mean(W,2);
            obj.prior.alpha.mu = mean(A,2);
            
            temp_W = W - repmat(obj.prior.weight.mu,1,length(outputCell));
            temp_A = A - repmat(obj.prior.alpha.mu,1,length(outputCell));
            if obj.trAdjust
                % Trace-normalized update
                C_W = (1/trace(temp_W*temp_W'))*(temp_W*temp_W');
                C_A = (1/trace(temp_A*temp_A'))*(temp_A*temp_A');
            else
                % standard ML covariance update
                 C_W = (1/(size(temp_W,2)-1))*(temp_W*temp_W');
                 C_A = (1/(size(temp_A,2)-1))*(temp_A*temp_A');
            end
            
            % regularize as necessary
            if rank(C_W) < size(C_W,1)
                e = eig((1/(size(temp_W,2)-1))*(temp_W*temp_W'));
                C_W = C_W + mean(e(e>0))*eye(size(C_W,1));
            end
            
            if rank(C_A) < size(C_A,1)
                e = eig((1/(size(temp_A,2)-1))*(temp_A*temp_A'));
                C_A = C_A + mean(e(e>0))*eye(size(C_A,1));
            end
            
            obj.prior.weight.sigma = C_W;
            obj.prior.alpha.sigma = C_A;
        end
        
    end
    
    methods(Static)
        function L = loss(w, X, y)
            L = 0;
            for i = 1:length(y)
                % implements straight (average) squared loss
                L = L + (w{2}'*X(:,:,i)*w{1}-y(i))^2;
            end
            L = L/length(y);
        end
        
       function y = predict(w, X, labels)
           y = zeros(size(X,3),1);
           for i = 1:length(y)
               y(i) = sign(w{2}'*X(:,:,i)*w{1});
           end
           tmp = zeros(size(y));
            for i = 1:2
                tmp(y == labels(i,2)) = labels(i,1);
            end
            y = tmp;
        end
    end
end