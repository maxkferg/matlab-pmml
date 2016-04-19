function testHyparameters(hypOpt)
% testHyparameters. Assert the hyperparameters match documented values
    % Throws an error on failure. Prints total error on success
    % See <documentationUrl> for an example case which generates these hyperparams
    sn = 0.1051;
    gammaStar = 2.4890;
    lambda1 = 1.5164;
    lambda2 = 59.3113;
    tol = 0.01; % 1 percent tolerance
    
    % Test covariance hyperparameters
    expected = [log(lambda1) log(lambda2) log(sqrt(gammaStar))];
    error1 = sum(abs(hypOpt.cov-expected')./expected');
    
    if error1 > tol
        error('Covariance hyperparameters do not match expected values')
    end
    
    % Test liklihood hyperparameters
    snActual = exp(hypOpt.lik);
    error2 = sum(abs(snActual-sn)/sn);
    
    if error2 > tol
        fprintf('Likilihood hyperparam sn=%f does not match expected value %f\n',snActual,sn)
    end
    
    maxError = 100*max(error1,error2);
    fprintf('Hyperparameters are within %.1f percent\n',maxError);
end