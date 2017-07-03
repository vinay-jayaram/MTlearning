classdef MT_FD_model < MT_baseclass
    
    properties(GetAccess = 'public', SetAccess = 'private')
        % remember what sort of model you are
        Type
        % weight vector for classification
        labels
        model_spec
        model_spat
        % internal label representation
        classid
        % parameters for convergence
        maxItVar % maximum variation between iterations before convergence
        maxNumVar % maximum number of dimensions allowed to not converge
    end
    
    methods
        function obj = MT_FD_model(type, varargin)
            % Constructor for multitask linear regression. 
            %
            % Input:
            %     d:    number of spectral features
            %     k:    number of spatial features
            %     type: type of model used for spatial/spectral priors
            obj@MT_baseclass(varargin{:})

            obj.maxItVar = invarargin(varargin,'max_it_var');
            if isempty(obj.maxItVar)
                obj.maxItVar = 1e-2;
            end
            obj.maxNumVar = invarargin(varargin,'max_pct_var');
            if isempty(obj.maxNumVar)
                obj.maxNumVar = 1e-2;
            end
            
            % Initialize models for each dimension
            assert(strcmp(type, 'linear') || strcmp(type, 'logistic'), ...
                'type has to be linear or logistic.');
            if strcmp(type, 'linear')
                obj.model_spat = MT_linear('dim_reduce', 0, 'n_its', 1, 'prior_init_val', 1);
                obj.model_spec = MT_linear('dim_reduce', 0, 'n_its', 1, 'prior_init_val', 0);
                obj.classid=[1;-1];
            elseif strcmp(type, 'logistic')
                obj.model_spat = MT_logistic('dim_reduce', 0, 'n_its', 1, 'prior_init_val', 1);
                obj.model_spec = MT_logistic('dim_reduce', 0, 'n_its', 1, 'prior_init_val', 0);
                obj.classid=[1;0];
            else
                fprintf('Unknown model type, something went terribly wrong!\n');
            end
            obj.Type = type;
        end
        
        %function [] = init_prior(obj)
        %    obj.prior.spec =  obj.spec_model.prior;
        %    obj.prior.spat =  obj.spat_model.prior;
        %end
        
        function [] = init_prior(obj, d, k)
             obj.model_spat.init_prior(d,1);
             obj.model_spec.init_prior(k,0);
            obj.prior.spec = obj.model_spec.prior;
            obj.prior.spat = obj.model_spat.prior;
        end
        
        function prior = fit_prior(obj, Xcell, ycell, varargin)
            % sanity checks
            assert(length(Xcell) == length(ycell), 'unequal data and labels arrays');
            assert(length(Xcell) > 1, 'only one dataset provided');
            for i = 1:length(Xcell)
                assert(size(Xcell{i},3) == length(ycell{i}), 'number of datapoints and labels differ');
                ycell{i} = reshape(ycell{i},[],1);
            end

            lambda = invarargin(varargin,'lambda');
            if isempty(lambda)
                lambda = NaN;
            end
            
            cv = invarargin(varargin, 'cv');
            % flag to get around infinite recursion...
            if isempty(cv)
                cv = 0;
            end
            if ~cv
                assert(length(unique(cat(1,ycell{:}))) == 2, 'more than two classes present in the data');
                obj.labels = [unique(cat(1,ycell{:})),obj.classid];
                % replace labels for algorithm
                for i = 1:length(ycell)
                    ycell{i} = MT_baseclass.swap_labels(ycell{i}, obj.labels, 'to');
                end
                obj.model_spat.labels = repmat(obj.classid,1,2);
                obj.model_spec.labels = repmat(obj.classid,1,2);
                obj.init_prior(size(Xcell{1},1),size(Xcell{1},2));
                obj.prior.spat.W = zeros(size(obj.prior.spat.mu,1),length(Xcell));
                obj.prior.spec.W = zeros(size(obj.prior.spec.mu,1),length(Xcell));
                prior = fit_prior@MT_baseclass(obj, Xcell, ycell, 'lambda', lambda);
            else
                obj.init_prior(size(Xcell{1},1),size(Xcell{1},2));
                obj.prior.spat.W = zeros(size(obj.prior.spat.mu,1),length(Xcell));
                obj.prior.spec.W = zeros(size(obj.prior.spec.mu,1),length(Xcell));
                prior = fit_prior@MT_baseclass(obj, Xcell, ycell, 'lambda', lambda);
            end
        end
        
        function [b, conv] = convergence(obj, prior, prev_prior)
            [b_spec, conv_spec] = obj.model_spec.convergence(prior.spec, prev_prior.spec);
            [b_spat, conv_spat] = obj.model_spat.convergence(prior.spat, prev_prior.spat);
            b = b_spec && b_spat;
            conv = conv_spec + conv_spat;
        end
        
        function [w, error] = fit_model(obj, X, y, lambda)
            num_chans = size(X,1);
            %w{2} = ones(chans,1);
            %w{1} = zeros(features,1); % I'm not convinced that a cell array is best here...but.....does it make sense to store things
                                                       % as rank-1 matrices
                                                       % and SVD them every
                                                       % time we want the
                                                       % components?
            % Init FD model parameters
            obj.model_spec.w = obj.model_spec.prior.mu;
            obj.model_spat.w = obj.model_spat.prior.mu;
            
             w_prev=zeros(num_chans, 1);
             count2=0;
             w = {obj.model_spec.w, obj.model_spat.w};
             while sum(or(abs(w{2}) > (w_prev+obj.maxItVar*w_prev),abs(w{2}) < (w_prev-obj.maxItVar*w_prev)))>0 && count2< obj.nIts                
                w_prev = abs(w{2});
                aX = dot3d(permute(X, [3, 2, 1]), obj.model_spat.w)';
                obj.model_spec.w = obj.model_spec.fit_model(aX, y, lambda);
                Xw = dot3d(permute(X, [3, 1, 2]), obj.model_spec.w)';
                obj.model_spat.w = obj.model_spat.fit_model(Xw, y, lambda);
                 w = {obj.model_spec.w, obj.model_spat.w};
                 % EXPERIMENTAL--norm alpha to 1
                 %w{2}=w{2}/norm(w{2});
                 count2=count2+1;
            end
            error = obj.loss(w, X, y);
        end
        
        function out = fit_new_task(obj, X, y, varargin)
            % fit_new_task(obj, X, y, varargin)
            
            ML = invarargin(varargin,'ml');
            if isempty(ML)
                ML = 0;
            end
            out = struct();

            % switch input labels using instance dictionary
            y_train = MT_baseclass.swap_labels(y, obj.labels, 'to');
            
            if ML
                prev_loss = 0;
                out.lambda = 1;
                out.loss = 1;
                count = 0;
                while abs(prev_loss - out.loss) > obj.maxItVar * prev_loss && count < obj.nIts
                    prev_loss = out.loss;
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
            
            out.predict = @(X)(obj.predict(out.w, X, obj.labels));
            out.training_acc = mean(y == out.predict(X));
        end
        
        function [] = update_prior(obj, outputCell)
            spec_weights = cell(length(outputCell), 1);
            spat_weights = cell(length(outputCell), 1);
            for i = 1: length(outputCell)
                spec_weights{i} = outputCell{i}{1};
                spat_weights{i} = outputCell{i}{2};
            end
            obj.model_spec.update_prior(spec_weights);
            obj.model_spat.update_prior(spat_weights);
            obj.prior.spec = obj.model_spec.prior;
            obj.prior.spat = obj.model_spat.prior;
            %outputCell
            %W = [];
            %A = [];
            %for i = 1:length(outputCell)
            %    W = cat(2,W,outputCell{i}{1});
            %    A = cat(2,A, outputCell{i}{2});
            %end
            %obj.prior.weight = MT_baseclass.update_gaussian_prior(W, obj.trAdjust);
            %obj.prior.alpha = MT_baseclass.update_gaussian_prior(A, obj.trAdjust);
        end
        
       function L = loss(obj, w, X, y)
            Xw = dot3d(permute(X, [3, 1, 2]), w{1})';            
            aX = dot3d(permute(X, [3, 2, 1]), w{2})';
            err1 = obj.model_spec.loss(w{1}, aX, y);
            err2 = obj.model_spat.loss(w{2}, Xw, y);
            L = err1 + err2;
       end
        
        function y = prior_predict(obj, X, varargin)
            Xw = dot3d(permute(X, [3, 1, 2]), obj.model_spec.w)';
            labels = invarargin(varargin, 'labels');
            if isempty(labels)
                y = obj.model_spat.prior_predict(Xw, 'labels', obj.labels);
            else
                y = obj.model_spat.prior_predict(Xw, 'labels', labels);
            end
        end 
        
        function [] = printswitches(obj)
            fprintf('[MT FD Model] Model class: %s\n',obj.Type);
            printswitches@MT_baseclass(obj);
            
            
        end
       
    end
    
    methods(Static)
        
       function y = predict(w, X, labels)
          y = zeros(size(X,3),1);
          for i = 1:length(y)
              y(i) = sign(w{2}'*X(:,:,i)*w{1});
          end
          y = MT_baseclass.swap_labels(y, labels, 'from');
       end
       
       function [W] = multi_task_f(obj, Xtrain, Ytrain, lambda)
           prior = obj.fit_prior(Xtrain, Ytrain, 'lambda', lambda, 'cv', 1);
           W = {prior.spec.W, prior.spat.W};
       end
       
       function [loss] = multi_task_loss(obj,W,Xtest,Ytest)
           loss = 0;
           for i = 1:length(Xtest)
               loss = loss + obj.loss({W{1}(:,i),W{2}(:,i)},Xtest{i},Ytest{i});
           end
       end
       
    end
end