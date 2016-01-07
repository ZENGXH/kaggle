clearvars;

% --- THERE ARE TWO SMALL DATASETS. PLAY WITH BOTH OF THEM --
% which dataset to use
% either 1 or 2
datasetNumber = 1;

if datasetNumber == 1
    X = [1 2 3 4 5 6 7 8 9]';
    y = [1.2 0.4 0.7 0.2 0.1 0.3 0.7 0.9 1.1]';
else
    X = [1 2 3 4 5 6 7 8 9]';
    y = [0.3 0.2 0.3 0.1 0.1 0.5 0.9 1.7 2.9]';
end

% -- WRITE HERE YOUR FUNCTION FOR GRID SEARCH ---
% -- THAT MAXIMIZES p(y|X) for the training data ---


% -- Plot points + Gaussian Process
clf();
plot( X, y, 'ro' );

% -- COMPUTE THE PREDICTION AND PLOT IT, SIMILAR TO ex6_priorplot ---
% -- Also plot the confidence interval --
