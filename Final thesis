%%% Simple Geometric Brownian motion
%% Load data and set up some thing
GSPC_data = readtable('GSPC.csv');
GSPC_adjusted = table2array(GSPC_data(:, 7));
y = 100 * log(GSPC_adjusted(2:end) ./ GSPC_adjusted(1:end-1));
T = size(y, 1);
X = ones(T, 1);
nsim = 5000;
burnin =1250;
mu_draw = zeros(nsim, 1);
sigma2_draw = zeros(nsim, 1);
an_mu_draw = zeros(nsim,1);
an_sigma2_draw = zeros(nsim,1);

%% Prior setting
mu0 = 0;
sigma0_2 = 10;
nu0 = 3;
S0 = 1 * (nu0 - 1);

%% Initialize the Markov chain
mu = (X' * X) \ (X' * y);
sig2 = sum((y - X * mu).^2) / T;

%% Standard Gibbs sampler and Antithetic Gibbs Sampler
for isim = 1:(nsim + burnin)
    randn_val = randn(1,1);
    % Sample mu
    Dmu = 1 / (1 / sigma0_2 + X' * X / sig2);
    mu_hat = Dmu * (mu0 / sigma0_2 + X' * y / sig2);
    mu = mu_hat + sqrt(Dmu) * randn_val;

    % Sample sig2
    e = y - X * mu;
    sig2 = 1 / gamrnd(nu0 + T / 2, 1 / (S0 + e' * e / 2));
    
    % Antithetic update
    if isim==1
    antithetic_mu = mu_hat - sqrt(Dmu) * randn_val;
    antithetic_e = y - X * antithetic_mu;
    antithetic_sig2 = 1 / gamrnd(nu0 + T / 2, 1 / (S0 + antithetic_e' * antithetic_e / 2));
    else 
    Dmu_1 = 1 / (1 / sigma0_2 + X' * X / antithetic_sig2);
    mu_hat_1 = Dmu_1 * (mu0 / sigma0_2 + X' * y / antithetic_sig2); 
    antithetic_mu = mu_hat_1 - sqrt(Dmu_1) * randn_val;
    antithetic_e = y - X * antithetic_mu;
    antithetic_sig2 = 1 / gamrnd(nu0 + T / 2,  1/(S0 + antithetic_e' * antithetic_e / 2));
    end
    
    
    % Store the parameters
    if isim > burnin
        isave = isim - burnin;
        mu_draw(isave, :) = mu;
        sigma2_draw(isave, :) = sig2;
         an_mu_draw(isave,:) = antithetic_mu;
        an_sigma2_draw(isave,:) = antithetic_sig2;
    end
end


 avg_mu = (mu_draw+an_mu_draw)/2;
 avg_sigma2 = (sigma2_draw+an_sigma2_draw)/2;

%% Mean, Variance and 95% HPD region of mu and sigma2
%% Normal case 
mu_hat = mean(mu_draw);
sigma2_hat = mean(sigma2_draw);
sd_mu = std(mu_draw);
sd_sigma2 = std(sigma2_draw);
q_mu = quantile(mu_draw,[0.025,0.975]);
q_sig2 = quantile(sigma2_draw,[0.025,0.975]);
param_table1 = table(mu_hat, sigma2_hat, sd_mu, sd_sigma2, {q_mu}, {q_sig2}, ...
    'VariableNames',...
    {'mu_hat', 'sigma2_hat', 'sd_mu', 'sd_sig2', 'q_mu', 'q_sig2'});
writetable(param_table1, 'param_normal.csv');

%% Antithetic case
mu1_hat = mean(avg_mu);
sigma21_hat = mean(avg_sigma2);
cov1 = cov(mu_draw,an_mu_draw);
cov2 = cov(sigma2_draw,an_sigma2_draw);
sd_mu1 = sqrt(cov1(1,1)+cov1(1,2));
sd_sigma21 = sqrt(cov2(1,1)+cov2(1,2));
q_mu1 = quantile(avg_mu,[0.025,0.975]);
q_sig21 = quantile(avg_sigma2,[0.025,0.975]);
param_table2 = table(mu1_hat, sigma21_hat, sd_mu1, sd_sigma21, {q_mu1}, {q_sig21}, ...
    'VariableNames',...
    {'mu1_hat', 'sigma21_hat', 'sd_mu1', 'sd_sigma21', 'q_mu1', 'q_sig21'});
writetable(param_table2, 'param_antithetic.csv');

%%
mu_seq_post =zeros(nsim,1);
sigma2_seq_post =zeros(nsim,1);
mu1_seq_post = zeros(nsim,1);
sigma21_seq_post = zeros(nsim,1);

for i=1:nsim
   mu_seq_post(i) = sum(mu_draw(1:i))/i;
   sigma2_seq_post(i) = sum(sigma2_draw(1:i))/i;
   mu1_seq_post(i)=sum(avg_mu(1:i))/i;
   sigma21_seq_post(i)=sum(avg_sigma2(1:i))/i;
end



%% Draw a graph 
%% Output of the Gibbs samler draw for mean and variance
%% Trace plot
%Mean
fs=14;
figure(1)
subplot(2,1,1)
plot((1:nsim)',mu_draw);
ylabel('$\mu$','interpreter','latex')
title("Gibbs Sampler")
set(gca,'FontSize',fs);
subplot(2,1,2)
plot((1:nsim)',avg_mu);
ylabel('$\mu_1$','interpreter','latex')
title("Antithetic Gibbs Sampler")
set(gca,'FontSize',fs);

% Standard Deviation
figure(2)
subplot(2,1,1)
plot((1:nsim)',sigma2_draw);
ylabel('$\sigma^2$','interpreter','latex')
title("Gibbs Sampler")
set(gca,'FontSize',fs);
subplot(2,1,2)
plot((1:nsim)',avg_sigma2);
ylabel('$\sigma_{1}^{2}$','interpreter','latex')
title("Antithetic Gibbs Sampler")
set(gca,'FontSize',fs);

%% Histograms Means and Variance (histogram)
% Mean
figure(3)
subplot(2,1,1)
histogram(mu_draw,100);
ylabel('$\mu$','interpreter','latex')
title("Gibbs Sampler")   
set(gca,'FontSize',12);
subplot(2,1,2)
histogram(avg_mu,100);
ylabel('$\mu_1$','interpreter','latex')
title("Antithetic Gibbs Sampler")   
set(gca,'FontSize',12);
% Variance 
figure(4)
subplot(2,1,1)
histogram(sigma2_draw,100);
ylabel('$\sigma^2$','interpreter','latex')
title("Gibbs Sampler")                              
set(gca,'FontSize',12);
subplot(2,1,2)
histogram(avg_sigma2,100);
ylabel('$\sigma_{1}^{2}$','interpreter','latex')
title("Antithetic Gibbs Sampler")
set(gca,'FontSize',12);
%% Sequential estimate and Whole sample estimate 
%% Mean
figure(5)
subplot(2,1,1)
plot((1:nsim)',mu_seq_post,'k-')
hold on
plot((1:nsim)',mu_hat*ones(nsim,1),'r-')
hold off
ylabel('$\mu$','interpreter','latex')
legend('Sequential Mean Estimate','Posterior Mean Whole Sample','location','southeast');
title("Gibbs Sampler")
set(gca,'fontsize',12);
print(gcf,'PosteriorSeq.eps','-depsc');
subplot(2,1,2)
plot((1:nsim)',mu1_seq_post,'k-')
hold on
plot((1:nsim)',mu1_hat*ones(nsim,1),'r-')
hold off
ylabel('$\mu_1$','interpreter','latex')
legend('Sequential Mean Estimate','Posterior Mean Whole Sample','location','southeast');
set(gca,'fontsize',12);
title("Antithetic Gibbs Sampler")
print(gcf,'PosteriorSeq.eps','-depsc');

% Variance
figure(6)
subplot(2,1,1)
plot((1:nsim)',sigma2_seq_post,'k-')
hold on
plot((1:nsim)',sigma2_hat*ones(nsim,1),'r-')
hold off
ylabel('$\sigma^2$','interpreter','latex')
legend('Sequential Variance Estimate','Posterior Variance Whole Sample','location','southeast');
title("Gibbs Sampler")
set(gca,'fontsize',12);
print(gcf,'PosteriorSeq.eps','-depsc');
subplot(2,1,2)
plot((1:nsim)',sigma21_seq_post,'k-')
hold on
plot((1:nsim)',sigma21_hat*ones(nsim,1),'r-')
hold off
ylabel('$\sigma_{1}^{2}$','interpreter','latex')
legend('Sequential Variance Estimate','Posterior Variance Whole Sample','location','southeast');
title("Antithetic Gibbs Sampler")
set(gca,'fontsize',12);
print(gcf,'PosteriorSeq.eps','-depsc');

