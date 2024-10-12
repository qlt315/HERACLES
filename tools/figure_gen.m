%% Figure 1 Reward Curve
figure(1);
% Assume rewards_list is a cell array of MxN reward matrices for different experiments
% For example: rewards_list = {rewards1, rewards2, rewards3};

% Train data
load("rainbow_dqn_lizhi/train_data/train_proposed_step_reward_matrix.mat")
load("rainbow_dqn_lizhi/train_data/train_tem_step_reward_matrix.mat")
load("rainbow_dqn_lizhi/train_data/train_sse_step_reward_matrix.mat")
load("baselines/dqn/train_data/dqn_train_proposed_step_reward_matrix.mat")
load("baselines/random_reward.mat")
load("baselines/amac_reward.mat")


colors = lines(7);         % Generate distinct colors for each curve
num_matrices = 5;  % Number of reward matrices
% step_reward_matrix = {proposed_step_reward_matrix, tem_step_reward_matritemx, sse_step_reward_matrix};
step_reward_matrix = {rainbow_dqn_train_proposed_step_reward_matrix,rainbow_dqn_train_sse_step_reward_matrix,...
    rainbow_dqn_train_tem_step_reward_matrix,dqn_train_proposed_step_reward_matrix,amac_step_reward_list};
hold on;
set(gca,'gridlinestyle','--','Gridalpha',0.8);


for i = 1:num_matrices
    rewards = step_reward_matrix{i};  % Get the i-th reward matrix

    % data smooth

    window_size = 1000;


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
ylim([-0.5, 0.5]);
set(gca, "YTick",-0.5:0.1:0.5);
legend_names = {"Proposed", "SSE", "TEM", "DQN","AMAC"};
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
% % Eval deta
% load("baselines/eval_amac_data.mat")
% load("baselines/eval_random_data.mat")
% load("rainbow_dqn/eval_data/rainbow_dqn_eval_proposed_data.mat")
% load("baselines/dqn/eval_data/dqn_eval_proposed_data.mat")


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

load("baselines\observations\observation.mat")
snr_list = 1:0.2:3;
%% Figure 4.1 Observation Energy
figure(4)
for i=1:7
    plot(snr_list, obs_energy(i,:), "-o", 'LineWidth', 2,"Color",colors(i,:)); hold on;
end

% for i=1:length(obs_energy_no_retrans)
%     plot(snr_list, obs_energy_no_retrans(i,:), "--o", 'LineWidth', 2,"Color",colors(i,:)); hold on;
% end
grid on;

% legend("Camera (Res101)","Radar (Res101)","LiDAR (Res101)","Dual Camera (Res101)", "Radar+LiDAR (Res101)","Camera+LiDAR (Res101)","All Sensors (Res18)",...
%     "Camera (Res101 No Re-trans)","Radar (Res101 No Re-trans)","LiDAR (Res101 No Re-trans)","Dual Camera (Res101 No Re-trans)", "Radar+LiDAR (Res101 No Re-trans)","Camera+LiDAR (Res101 No Re-trans)","All Sensors (Res18 No Re-trans)");
legend("Camera (Res101)","Radar (Res101)","LiDAR (Res101)","Dual Camera (Res101)", "Radar+LiDAR (Res101)","Camera+LiDAR (Res101)","All Sensors (Res18)");
set(gca, "YTick",0:2:15);
set(gca, "XTick",1:0.2:3);
xlabel('SNR (dB)'),ylabel('Total Energy Consumption (J)');
set(gca,'FontName','Times New Roman','FontSize',12);
set(gca,'gridlinestyle','--','Gridalpha',0.8);
%% Figure 4.2 Observation Delay
figure(5)
for i=1:7
    plot(snr_list, obs_delay(i,:), "-o", 'LineWidth', 2,"Color",colors(i,:)); hold on;
end

% for i=1:length(obs_delay_no_retrans)
%     plot(snr_list, obs_delay_no_retrans(i,:), "--o", 'LineWidth', 2,"Color",colors(i,:)); hold on;
% end
grid on;

% legend("Camera (Res101)","Radar (Res101)","LiDAR (Res101)","Dual Camera (Res101)", "Radar+LiDAR (Res101)","Camera+LiDAR (Res101)","All Sensors (Res18)",...
%     "Camera (Res101 No Re-trans)","Radar (Res101 No Re-trans)","LiDAR (Res101 No Re-trans)","Dual Camera (Res101 No Re-trans)", "Radar+LiDAR (Res101 No Re-trans)","Camera+LiDAR (Res101 No Re-trans)","All Sensors (Res18 No Re-trans)");
legend("Camera (Res101)","Radar (Res101)","LiDAR (Res101)","Dual Camera (Res101)", "Radar+LiDAR (Res101)","Camera+LiDAR (Res101)","All Sensors (Res18)");
set(gca, "YTick",0:1:15);
set(gca, "XTick",1:0.5:3);
xlabel('SNR (dB)'),ylabel('Total Delay (S)');
set(gca,'FontName','Times New Roman','FontSize',12);
set(gca,'gridlinestyle','--','Gridalpha',0.8);




%% Figure 4.2 Observation Re-trans
figure(5)
for i=1:7
    plot(snr_list, obs_re_trans(i,:), "-o", 'LineWidth', 2,"Color",colors(i,:)); hold on;
end

grid on;

legend("Camera (Res101)","Radar (Res101)","LiDAR (Res101)","Dual Camera (Res101)", "Radar+LiDAR (Res101)","Camera+LiDAR (Res101)","All Sensors (Res18)");
% set(gca, "YTick",0:1:10);
% set(gca, "XTick",1:0.5:3);
xlabel('SNR (dB)'),ylabel('Re-transmission Number');
set(gca,'FontName','Times New Roman','FontSize',12);
set(gca,'gridlinestyle','--','Gridalpha',0.8);



%% Figure 4.3 Observation Accuracy
figure(6)
load("baselines\observations\obs_acc.mat")
data=obs_acc';
% X = ['Snow','Fog','Motorway','Night','Rain','Sunny'];
X = [1,2,3,4,5,6];
GO = bar(X,data,1,'EdgeColor','k','LineWidth',1);

hatchfill2(GO(1),'cross','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(2),'single','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(3),'single','HatchAngle',0,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(4),'single','HatchAngle',-45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(5),'cross','HatchAngle',-60,'HatchDensity',30,'HatchColor','k');
hatchfill2(GO(6),'cross','HatchAngle',60,'HatchDensity',30,'HatchColor','k');

GO(1).FaceColor = colors(1,:);
GO(2).FaceColor = colors(2,:);
GO(3).FaceColor = colors(3,:);
GO(4).FaceColor = colors(4,:);
GO(5).FaceColor = colors(5,:);
GO(6).FaceColor = colors(6,:);
 
% Draw the legend
legendData = {'Camera (Res101)','Radar (Res101)','LiDAR (Res101)','Dual Camera (Res101)', 'Radar+LiDAR (Res101)','Camera+LiDAR (Res101)','All Sensors (Res18)'};
[legend_h, object_h, plot_h, text_str] = legendflex(GO, legendData, 'Padding', [2, 2, 10], 'FontSize', 11, 'Location', 'NorthWest');
% object_h(1) is the first bar's text
% object_h(2) is the second bar's text
% object_h(3) is the first bar's patch
% object_h(4) is the second bar's patch
%
% Set the two patches within the legend
hatchfill2(object_h(8), 'cross', 'HatchAngle', 45, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(9), 'single', 'HatchAngle', 45, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(10), 'single', 'HatchAngle', 0, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(11), 'single', 'HatchAngle', -45, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(12), 'cross', 'HatchAngle', -60, 'HatchDensity', 30, 'HatchColor', 'k');
hatchfill2(object_h(13), 'cross', 'HatchAngle', 60, 'HatchDensity', 30, 'HatchColor', 'k');
% Some extra formatting to make it pretty :)
set(gca,'FontName','Times New Roman','FontSize',12);
set(gca,'gridlinestyle','--','Gridalpha',0.8);
set(gca, 'XMinorTick','on', 'XMinorGrid','on', 'YMinorTick','on', 'YMinorGrid','on');
% xlim([0.5, 2.5]);
ylim([0, 100]);

% hTitle = title('Texture filled bar chart');
% hXLabel = xlabel('Samples');
hYLabel = ylabel('Accuracy (mAP)');
ax = gca;
ax.XTickLabel = {'Snow','Fog','Motorway','Night','Rain','Sunny'};

