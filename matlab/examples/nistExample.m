function nistExample()
    % Complete the nist example using the gpml package
    
    % x is defined in the GP PMML example as:
    x = [1,3; 2,6];
    y = [1; 2]; 
    xnew = [1,4];
    ynew = [1;2];
    
    % Define mean and cov function
    meanfunc = {@meanZero};  % Zero mean function
    covfunc = {@covSEard};   % ARD Squared exponential cov function
    gamma = sqrt(3);         % Realistic starting value for gamma
    lambda1 = 2;             % Realistic starting value for lambda1
    lambda2 = 60;            % Realistic starting value for lambda2
    
    sn = 0.1; % sigma
    likfunc = @likGauss; 
    
    hyp.lik = log(sn);
    hyp.mean = [];
    hyp.cov = log([lambda1; lambda2; gamma]);

    % Optimise hyperparameters
    hypOpt = minimize(hyp, @gp, -100, @infExact, [], covfunc, likfunc, x, y);
    
    % Test that the hyperparameters are correct
    testHyparameters(hypOpt);
    
    % Perform regression
    nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y);
    
    % Test that we are predicting the right values
    [mu,s] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, xnew);
end


function testHyparameters(hypOpt)
    % Check that the hyperparameters match the ones in the example
    % Throws an error on failure. Prints total error on success
    % ?* = 2.4890, ?* = (1.5164,59.3113) and ??*= 0.1051. 
    gammaStar = 2.4890;
    lambda1 = 1.5164;
    lambda2 = 59.3113;
    
    tol = 0.01; % 1 percent tolerance
    expected = [log(lambda1) log(lambda2) log(sqrt(gammaStar))];
    error = sum(abs(hypOpt.cov-expected')./expected');
    
    if error > 0.01
        throw 'Hyperparameters do not match expected values'
    end
    fprintf('Hyperparameters are within %.1f percent\n',100*error);
end


function testPrediction(muPredict,sPredict)
    % Test that the prediction is close to the expected value
    % Uses the optimum hyperparameters, as tested in @testHyparameters
    

end







function randomNistExample()
% Generate some random values for training
% Train using the same mean and cov functions as the NIST example

    % Define mean function
    meanfunc = {@meanZero};
    covfunc = {@covSEard};
    ell = 1/4;
    sf = 1; 
    likfunc = @likGauss; 
    sn = 0.1; 
    hyp.lik = log(sn);
    
    hyp.mean = [];
    hyp.cov = log([ell; sf]);

    % Generate random results
    n = 20;
    x = gpml_randn(0.3, n, 1);
    K = feval(covfunc{:}, hyp.cov, x);
    mu = feval(meanfunc{:}, hyp.mean, x);
    y = chol(K)'*gpml_randn(0.15, n, 1) + mu + exp(hyp.lik)*gpml_randn(0.2, n, 1);

    plot(x, y, '+')

    % Perform regression
    nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y);

    z = linspace(-1.9, 1.9, 101)';
    [m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

    f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
    fill([z; flipdim(z,1)], f, [7 7 7]/8);
    hold on; plot(z, m); plot(x, y, '+');
end
