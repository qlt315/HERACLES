%% Figure 1 Reward Curve
figure(1);
% Assume rewards_list is a cell array of MxN reward matrices for different experiments
% For example: rewards_list = {rewards1, rewards2, rewards3};

% Train data
load("rainbow_dqn_lizhi/train_data/train_proposed_step_reward_matrix.mat")
load("baselines/dqn/train_data/dqn_train_proposed_step_reward_matrix.mat")
load("baselines/random_reward.mat")
load("baselines/amac_reward.mat")

% Eval deta
load("baselines/eval_amac_data.mat")
load("baselines/eval_random_data.mat")
load("rainbow_dqn/eval_data/rainbow_dqn_eval_proposed_data.mat")
load("baselines/dqn/eval_data/dqn_eval_proposed_data.mat")

num_matrices = 4;  % Number of reward matrices
colors = lines(num_matrices+1);         % Generate distinct colors for each curve
% step_reward_matrix = {proposed_step_reward_matrix, tem_step_reward_matritemx, sse_step_reward_matrix};
step_reward_matrix = {rainbow_dqn_train_proposed_step_reward_matrix,dqn_train_proposed_step_reward_matrix,amac_step_reward_list, random_step_reward_matrix};
hold on;
set(gca,'gridlinestyle','--','Gridalpha',0.8);


for i = 1:num_matrices
    rewards = step_reward_matrix{i};  % Get the i-th reward matrix

    % data smooth

    window_size = 300;


    smooth_method = 'movmean';  
    
    smoothed_rewards = zeros(size(rewards));
    
    for j = 1:size(rewards, 1)
        smoothed_rewards(j, :) = smoothdata(rewards(j, :), smooth_method, window_size);
    end
    
    rewards = smoothed_rewards;

    [M, N] = size(rewards);     % Get the number of experiments and time steps
    
    % Calculate mean and standard deviation for this reward matrix
    mean_rewards = mean(rewards, 1);   % Mean reward at each time step
    std_rewards = std(rewards, 0, 1);  % Standard deviation at each time step
    
    % Time steps
    x = 1:N;
    
    % Plot the mean curve (solid line)
    plot(x, mean_rewards, 'LineWidth', 2, 'Color', colors(i,:));  % Solid line for mean reward
    
    % Plot the shaded area for standard deviation (without adding to the legend)
    fill([x fliplr(x)], [mean_rewards+std_rewards fliplr(mean_rewards-std_rewards)], ...
        colors(i,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');  % Shaded area
end


% Add labels, title, and only solid line legends
grid on;
xlabel('Step Index'),ylabel('Reward');
set(gca,'FontName','Times New Roman','FontSize',12);
title('Reward Curves with Mean and Standard Deviation');

% Generate legend with solid lines only

legend_names = {"Proposed","DQN","AMAC", "Random"};
legend(legend_names, 'Location', 'best');

%% Figure 2 Modulation SNR-BER Fitting
figure(2);
load("system_data/modulation_fitting_data.mat");
scatter(hm_snr_list, hm_ber_1, "o","LineWidth",2,"ColorVariable",colors(1,:));hold on;
scatter(hm_snr_list, hm_ber_2, "o","LineWidth",2,"ColorVariable",colors(2,:));
scatter(tm_snr_list_qpsk, tm_ber_qpsk, "o","LineWidth",2,"ColorVariable",colors(3,:));
scatter(tm_snr_list_16qam, tm_ber_16qam, "o","LineWidth",2,"ColorVariable",colors(4,:));
scatter(tm_snr_list_64qam, tm_ber_64qam, "o","LineWidth",2,"ColorVariable",colors(5,:));

plot(hm_snr_fit_list,hm_ber_1_fit,"LineWidth",2,"Color",colors(1,:));
plot(hm_snr_fit_list,hm_ber_2_fit,"LineWidth",2,"Color",colors(2,:));
plot(tm_snr_fit_list_qpsk, tm_ber_qpsk_fit,"LineWidth",2,"Color",colors(3,:));
plot(tm_snr_fit_list_16qam, tm_ber_16qam_fit,"LineWidth",2,"Color",colors(4,:));
plot(tm_snr_fit_list_64qam, tm_ber_64qam_fit,"LineWidth",2,"Color",colors(5,:));

grid on;
xlabel('SNR (dB)'),ylabel('Bit Error Rate (BER)');
set(gca,'FontName','Times New Roman','FontSize',12);
title('LDPC Code Rate=1/2 (64-QAM=2/3), Channel Estimation Error Parameter = 0.5');
legend("QPSK / 16QAM Real Data (Layer 1)","QPSK / 16QAM Real Data (Layer 2)", "QPSK Real Data", "16-QAM Modulation Real Data", "64-QAM Modulation Real Data", ...
    "QPSK / 16QAM Fit Data (Layer 1)", "QPSK / 16QAM Fit Data (Layer 2)","QPSK Fit Data", "16-QAM Modulation Fit Data", "64-QAM Modulation Fit Data");


%% Figure 3 Bar Chart of reward, acc, delay and energy
% time_list = zeros(8,1);
% figure('visible','on')
% time_list_multi = [1.599, -0.33, -3.07, -0.35];
% b = bar(time_list_multi,1,'EdgeColor','k','LineWidth',1);
% hatchfill2(b(1),'cross','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
% ax = gca;
% ax.XTickLabel = {"Proposed","DQN","AMAC", "Random"};
% grid on;
% set(gca,'FontName','Times New Roman','FontSize',12);
% 
% % time_list = zeros(8,1);
% figure('visible','on')
% time_list_multi = [1200,1073;1140,1013;1018,905;1753 0;0,1093];
% b = bar(time_list_multi,1,'EdgeColor','k','LineWidth',1);
% hatchfill2(b(1),'cross','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
% hatchfill2(b(2),'single','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
% legend([b(1),b(2)],'Coop','Non-Coop');
% ax = gca;
% ax.XTickLabel = {'Proposed','CBO','MPO','MADDPG','IQL'};
% grid on;
% set(gca,'FontName','Times New Roman','FontSize',12);