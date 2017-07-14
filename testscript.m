%clear
clc
fprintf('Script started...\n')

% load fast features
load MTtestdata;

% Instantiate models
n_its = 10;

order = {'l2','l2-trace','l1-diag','l1'};

%%%%%%%%%%%%%%%%%%%%%%%%%%
% How to use the linear version of this approach
%%%%%%%%%%%%%%%%%%%%%%%%%%

% These models show off each sort of covariance covariance update with the
% linear version
for i = 1:length(order)
    disp(['********************* Covariance update: ', order{i}, '*************************']);
    linear_model{i} = MT_linear('dim_reduce',1,'n_its',1e2,'lambda_ml',0,'cov_flag',order{i},'zero_mean',1);
    regression_model{i} = MT_linear_regression('dim_reduce',0,'n_its',1e2,'lambda_ml',0,'cov_flag',order{i},'zero_mean',0);
    log_model{i} = MT_logistic('dim_reduce',0,'n_its',n_its,'lambda_ml',0,'cov_flag',order{i});
    disp('Confirm prior computation switches: ');
    linear_model{i}.printswitches;
    
    % Code to fit the prior (training on the first 4)
    disp('Training L2 loss prior...')
    linear_model{i}.fit_prior(T_X2d(1:4), T_y(1:4));
    disp('Training regression prior...')  
    regression_model{i}.fit_prior(T_X2d(1:4), T_y(1:4));
    disp('Training logistic loss prior...')
    log_model{i}.fit_prior(T_X2d(1:4), T_y(1:4));

    
    % Code that computes prior accuracy on the held-out session data
    pacc_lin = mean(linear_model{i}.prior_predict(T_X2d{5}) == T_y{5});
    pacc_log = mean(log_model{i}.prior_predict(T_X2d{5}) == T_y{5});
    fprintf('Prior accuracies on held-out session: \n Linear: %.2f\n Logistic: %.2f\n', pacc_lin, pacc_log);
    fprintf('Prior rmse on held-out session for regression model: %.2f\n', sqrt(mean((regression_model{i}.prior_predict(T_X2d{5}) - T_y{5}).^2)));
    
    % Code to fit the new task (with cross-validated lambda)
    fitted_new_linear_task = linear_model{i}.fit_new_task(T_X2d{5}, T_y{5}, 'ml', 0);
    fitted_new_log_task = log_model{i}.fit_new_task(T_X2d{5},T_y{5}, 'ml', 0);
    new_regression = regression_model{i}.fit_new_task(T_X2d{5},T_y{5},'ml',0);
    
    % Classifying after the new task update
    fprintf('New task *training set* accuracy: \n Linear: %.2f\nLogistic: %.2f\n',...
    mean(fitted_new_linear_task.predict(T_X2d{5}) == T_y{5}), ...
    mean(fitted_new_log_task.predict(T_X2d{5}) == T_y{5}));
    fprintf('rmse on new task for regression model: %.2f\n', sqrt(mean((new_regression.predict(T_X2d{5}) - T_y{5}).^2)));
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
% How to use the bilinear version of this approach
%%%%%%%%%%%%%%%%%%%%%%%%%%%

type = {'linear', 'logistic'};

for i = 1:length(type)
    disp(['********************FD ', type{i},'**********************']);
    FD{i} = MT_FD_model(type{i},'n_its',5,'verbose',0);
    FD{i}.printswitches;
    FD{i}.fit_prior(T_X(1:4), T_y(1:4));
    acc = mean(FD{i}.prior_predict(T_X{5}) == T_y{5});
    fprintf('Prior accuracy on held-out data: %.2f\n', acc*100);
    out = FD{i}.fit_new_task(T_X{5}, T_y{5}, 'ml', 0,'verbose',1);
    acc = mean(out.predict(T_X{5}) == T_y{5});
    fprintf('New task training accuracy: %.2f\n', acc*100);
end

fprintf('Script finished!\n');
