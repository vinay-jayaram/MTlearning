classdef MT_linear_classification < MT_linear_regression
    % Base class for models that inherit from MT_regression.
    %
    %
    % Major difference is that it internally swaps out labels. 
    %
    properties(GetAccess = 'public', SetAccess = 'public')
        % vector that contains unique labels for prediction
        labels
        % Internal class labels
        classid
    end
    
    methods
        function obj = MT_linear_classification(varargin)
            % Constructor for multitask linear regression.
            %
            % Input:
            %     d           : Dimension of data in order to construct
            %     prior
            %     varargin: Flags
            
            % construct superclass
            obj@MT_linear_regression(varargin{:})
            obj.labels = [];
            obj.classid = [1;-1];
        end

        function prior = fit_prior(obj, Xcell, ycell, varargin)
                           assert(length(unique(cat(1,ycell{:}))) == 2, 'more than two classes present in the data');
                                           % always re-update the labels for each use of the prior
                obj.labels = [unique(cat(1,ycell{:})),obj.classid];
                % replace labels with {1,-1} for algorithm
                for i = 1:length(ycell)
                    ycell{i} = MT_baseclass.swap_labels(ycell{i}, obj.labels, 'to');
                end
            prior = fit_prior@MT_linear_regression(Xcell, ycell, varargin);
        end
        
        function out = fit_new_task(obj, X, y, varargin)
            
            regression_out = fit_new_task@MT_linear_regression(X, y, varargin);
            
            out.w = regression_out.w;
            out.loss = regression_out.loss;
            out.predict = @(X)(MT_baseclass.swap_labels(regression_out.predict(X), obj.labels, 'from'));
            out.training_acc = mean(y == out.predict(X));
        end
       
    end
    
    methods(Static)
        function y = predict(w, X, labels)
            y = MT_baseclass.swap_labels(sign(X'*w), labels, 'from');
        end
    end
end