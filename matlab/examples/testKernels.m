function testKernels()
% This test is used to test that skikit is using the same kernal
% and mean function as the Matlab GPML package. We have a similar
% python test called TestNistExample.test_kernels that should 
% generate exactly the same values
    xTrain = [1,3; 2,6];
    yTrain = [1,2]';
    xTest = [1,4];
    xWild = [1,9]; 
    
    % Set the hyperparameters
    sn = 0.1; % sigma
    sf = 1; % Scaling parameter
    ell1 = 3; % Length scale
    ell2 = 50;
    
    hyp.lik = log(sn);
    hyp.mean = [];
    hyp.cov = log([ell1,ell2,sf]);
    
    % Check that the covSEisoU kernel matches the RBF kernel
    %kernel = covSEard(hyp.cov, xTest, xNew);
    
    % Assert that the result is correct
    %assert((rbf-0.7165)<0.001);
   
    % Define mean and cov function
    likfunc = @likGauss;
    meanfunc = @meanZero;  % Zero mean function
    covfunc = @covSEard;  % RBF Kernel

    % Optimize hyperparameters
    hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, xTrain, yTrain);
    
    % Match the python kernel and take it for a test drive
    %hyp.lik = log(0.1051);
    %hyp.mean = [];
    %hyp.cov = log([1.09, 59, 1]);
    %bfn = covSEard(hyp.cov, xTest, xWild)
    exp(hyp.cov)
    
    % Print out the equivalent python length scale
    %ell1 = exp(hyp.cov(1));
    %ell2 = exp(hyp.cov(2));
    %sf = exp(hyp.cov(3));
    %length_scale = 1 ./ (log(sf^2) * [1/ell1^2, 1/ell2^2])
    %noise_var = exp(hyp.lik)
  
    % Make a new prediction
    [mu,s] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, xTrain, yTrain, xTest)
end