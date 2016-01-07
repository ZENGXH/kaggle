clearvars;

% Load data
load mauna.txt
z = mauna(:,2) ~= -99.99;             % get rid of missing data
year = mauna(z,1); co2 = mauna(z,2);  % extract year and CO2 concentration

% Center data
yMean = mean(co2);
co2 = co2 - yMean;

% divide in training (x) and testing (x_test)
x = year(year<2004); y = co2(year<2004);            % training data
x_test = year(year>2004); y_test = co2(year>2004);  % test data

% predictions, x star. You will have to make GP predictions for each x_star
x_star = linspace(min(year), 2030, 2000)';  % predict for all years


% -----> FILL IN K(X,X), K(X,X*), ..etc
Kx_x = ...
Kx_xstar = ...
Kxstar_xstar = ...


% predict mean and variance, GP equations
% You don't need to do anything here, all done.
f_star_mu = Kx_xstar' * (Kx_x)^-1 * y;
f_star_var = Kxstar_xstar - Kx_xstar' * (Kx_x)^-1 * Kx_xstar;

clf();

% show prediction. You don't need to do anything here, all done.
uncert = 2*sqrt(diag(f_star_var));  % +/- 2 sigma => 95% confidence interval
jbfill(x_star, f_star_mu + uncert, ...
               f_star_mu - uncert, [0.5,0.5,0.5]); hold on;

plot( x, y, 'b.'); hold on;
plot( x_test, y_test, 'r.'); hold on;  % new data for testing

