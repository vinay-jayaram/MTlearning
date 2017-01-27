%clear
clc
fprintf('Script started...\n')

% load fast features
load MTtestdata;

% Instantiate models
n_its = 10;

order = {'l2','l2-trace','l1-diag'};

%%%%%%%%%%%%%%%%%%%%%%%%%%
% How to use the linear version of this approach
%%%%%%%%%%%%%%%%%%%%%%%%%%

% These models show off each sort of covariance covariance update with the
% linear version
for i = 1:length(order)
    disp(['********************* Covariance update: ', order{i}, '*************************']);
    linear_model{i} = MT_linear('dim_reduce',0,'n_its',1e2,'lambda_ml',0,'cov_flag',order{i},'zero_mean',1);
    log_model{i} = MT_logistic('dim_reduce',0,'n_its',n_its,'lambda_ml',0,'cov_flag',order{i});
    disp('Confirm prior computation switches: ');
    linear_model{i}.printswitches;
    
    % Code to fit the prior
    disp('Training L2 loss prior...')
    linear_model{i}.fit_prior(T_X2d, T_y);
    disp('Training logistic loss prior...')
    log_model{i}.fit_prior(T_X2d, T_y);
    
    % Code that computes prior accuracy on the training data
    pacc_lin = mean(linear_model{i}.prior_predict(X2d_s) == y_s);
    pacc_log = mean(log_model{i}.prior_predict(X2d_s) == y_s);
    fprintf('prior accuracies: \n Linear: %.2f\n Logistic: %.2f\n', pacc_lin, pacc_log);
    
    % Code to fit the new task (with cross-validated lambda)
    fitted_new_linear_task = linear_model{i}.fit_new_task(X2d_s, y_s, 'ml', 1);
    fitted_new_log_task = log_model{i}.fit_new_task(X2d_s, y_s, 'ml', 1);

    % Classifying after the new task update
        fprintf('New task *training set* accuracy: \n Linear: %.2f\nLogistic: %.2f\n',...
        mean(fitted_new_linear_task.predict(X2d_s) == y_s), ...
        mean(fitted_new_log_task.predict(X2d_s) == y_s));
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
% How to use the bilinear version of this approach
%%%%%%%%%%%%%%%%%%%%%%%%%%%

type = {'linear'};%,'logistic'};

for i = 1:length(type)
    disp(['********************FD ', type{i},'**********************']);
    FD{i} = MT_FD_model(type{i},'n_its',5,'verbose',1);
    FD{i}.printswitches;
    FD{i}.fit_prior(T_X, T_y);
    acc = mean(FD{i}.prior_predict(X_s) == y_s);
fprintf('Prior accuracy: %.2f\n', acc*100);
out = FD{i}.fit_new_task(X_s, y_s, 'ml', 1);
acc = mean(out.predict(X_s) == y_s);
fprintf('New task training accuracy: %.2f\n', acc*100);
end

fprintf('Script finished!\n');
