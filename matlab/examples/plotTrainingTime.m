function plotTrainingTime()
% Plot the training time against the number of training samples
    clc; clear;
    close all; 
    addpath('gpml');
    
    % Proof of concept
    n = 10;
    isPlot = true;
    [x,y] = generateGPdata(n,isPlot);
    trainGPModel(x,y,isPlot);
    
    % Time the process
    isPlot = false;
    nsamples = 2.^[4 5 6 7 8];
    t = zeros(1,length(nsamples));
    for i=1:length(nsamples)
        n = nsamples(i);
        [x,y] = generateGPdata(n,isPlot);
        f = @() trainGPModel(x,y,isPlot);
        t(i) = timeit(f);
    end
    plot(nsamples,t)
end




function trainGPModel(x,y,isPlot)
% Train the Gaussian Process model
%   We assume the mean and convariance functions are unknown
%   We optimize the hypoparameters to minimize the log likilihood
    n = length(y);
    z = linspace(-4, 3, 101)';
    meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [0.5; 1];
    covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
    likfunc = @likGauss;

    hyp.cov = [0; 0]; hyp.mean = [0; 0]; hyp.lik = log(0.1);
    hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, x, y);
    
    if isPlot
        [m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);
        plotGPModel(m,s2,x,y,z)
    end
end


function plotGPModel(m,s2,x,y,z)
% Plot the GP Model
    % @param m - a list of mean values
    % @param s2 - a list of variances
    % @param x - the training points
    % @param y - the training values
    n = length(y);
    f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];

    figure
    fill([z; flipdim(z,1)], f, [7 7 7]/8)
    hold on; 
    axis([-2 2 -3 4]);
    plot(z, m,'b');
    plot(x, y, '+');
    xlabel('Input, x');
    ylabel('Output,y');
    title(sprintf('Gaussian Process Regression, n=%i',n));
end


function [x,y] = generateGPdata(n,isPlot)
    % Generate data from a Gaussian process with random parameters
    meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [0.5; 1];
    covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
    likfunc = @likGauss; sn = 0.1; 
    hyp.lik = log(sn);

    x = gpml_randn(0.3, n, 1);
    K = feval(covfunc{:}, hyp.cov, x);
    mu = feval(meanfunc{:}, hyp.mean, x);
    y = chol(K)'*gpml_randn(0.15, n, 1) + mu + exp(hyp.lik)*gpml_randn(0.2, n, 1);

    figure;
    plot(x, y, '+')
    axis([-2 2 -3 4]);
    xlabel('Input, x');
    ylabel('Output,y');
    title(sprintf('Samples from GP, n=%i',n));
        
    % Plot the mean function and 2xstandard deviations from the mean
    if isPlot
        nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y)

        z = linspace(-1.9, 1.9, 101)';
        [m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

        f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 

        figure;
        fill([z; flipdim(z,1)], f, [7 7 7]/8);
        hold on; 
        plot(z, m,'b');
        plot(x, y, '+')
        axis([-2 2 -3 4]);
        xlabel('Input, x');
        ylabel('Output,y');
        title(sprintf('True Gaussian Process, n=%i',n));
    end
end

