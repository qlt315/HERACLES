%% Figure 1 Reward Curve
figure(1);
% Assume rewards_list is a cell array of MxN reward matrices for different experiments
% For example: rewards_list = {rewards1, rewards2, rewards3};

% Train data
load("rainbow_dqn/train_data/train_proposed_erf_data.mat")
load("rainbow_dqn/train_data/train_proposed_origin_data.mat")
load("rainbow_dqn/train_data/train_tem_data.mat")
load("rainbow_dqn/train_data/train_sse_data.mat")
load("baselines/dqn/train_data/train_proposed_erf_data.mat")
load("baselines/random_reward.mat")
load("baselines/amac_data.mat")

% colors = [184/255, 219/255, 179/255;
%     114/255,176/255,99/255;
%     113/255,154/255,172/255;
%     226/255,145/255,53/255;
%     148/255,198/255,205/255;
%     74/255,95/255,126/255];


% colors = [141/255, 47/255, 37/255;
%     78/255,25/255,69/255;
%     203/255,148/255,117/255;
%     140/255,191/255,135/255;
%     62/255,96/255,141/255;
%     144/255,146/255,145/255];

% colors = [191/255, 223/255, 210/255;
%     37/255,125/255,139/255;
%     104/255,190/255,217/255;
%     239/255,206/255,135/255;
%     234/255,165/255,88/255;
%     237/255,141/255,90/255];


colors = [193/255, 86/255, 94/255;
    193/255,86/255,94/255;
    220/255,169/255,106/255;
    130/255,173/255,127/255;
    126/255,164/255,209/255;
    121/255,67/255,142/255;
    128/255,124/255,125/255];

colors = lines(7);         % Generate distinct colors for each curve
num_matrices = 6;  % Number of reward matrices
% step_reward_matrix = {proposed_step_reward_matrix, tem_step_reward_matritemx, sse_step_reward_matrix};
step_reward_matrix = ...
    {rainbow_dqn_proposed_erf_step_reward_matrix,... 
    rainbow_dqn_proposed_origin_step_reward_matrix,...
    rainbow_dqn_sse_step_reward_matrix,...
    rainbow_dqn_tem_step_reward_matrix,...
    dqn_proposed_erf_step_reward_matrix,amac_step_reward_list};
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
legend_names = {"Proposed (erf)", "Proposed (origin)", "SSE", "TEM", "DQN","AMAC"};
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

%% Fig.5 Different Reward Weight
% load("experiments\diff_kappa_data\diff_kappa_data.mat")
load("experiments\diff_kappa_data\rainbow_proposed_erf_diff_kappa_matrix.mat")
load("experiments\diff_kappa_data\rainbow_proposed_origin_diff_kappa_matrix.mat")
load("experiments\diff_kappa_data\rainbow_sse_diff_kappa_matrix.mat")
load("experiments\diff_kappa_data\rainbow_tem_diff_kappa_matrix.mat")
load("experiments\diff_kappa_data\dqn_diff_kappa_matrix.mat")
load("experiments\diff_kappa_data\amac_diff_kappa_matrix.mat")
% data = {rainbow_proposed_erf_diff_kappa_matrix, rainbow_proposed_origin_diff_kappa_matrix, rainbow_sse_diff_kappa_matrix,...
%     rainbow_tem_diff_kappa_matrix, dqn_diff_kappa_matrix, amac_diff_kappa_matrix};
% 
% title_list = ["Performance Comparison Under Different Accuracy Reward Weights",...
%     "Performance Comparison Under Different Delay Reward Weights",...
%     "Performance Comparison Under Different Energy Consumption Reward Weights"];
% for i=1:3
%     figure(6+i)
%         for j=1:6
%             data_curr = data{j};
%             % Data cleaning
%             filtered_data_curr = remove_outliers(data_curr(:,:,i));
%             x = filtered_data_curr(1,:); % delay
%             y = filtered_data_curr(2,:); % energy
%             z = filtered_data_curr(4,:); % acc vio prob
%             [X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),'v4');
%             surf(X, Y, Z, 'FaceColor', colors(j,:), 'FaceAlpha', 0.5); hold on;
%         end
%             title(title_list(i));
%             xlabel('Delay (S)');
%             ylabel('Energy Consumption (J)');
%             zlabel('Accuracy Violation Rate');
%             legend_names = {"Proposed (erf)", "Proposed (origin)", "SSE", "TEM", "DQN","AMAC"};
%             legend(legend_names, 'Location', 'best');
%             set(gca,'gridlinestyle','--','Gridalpha',0.8);
%             set(gca,'FontName','Times New Roman','FontSize',12);
% end

% Plot wih Lines
data = {rainbow_proposed_erf_diff_kappa_matrix, rainbow_proposed_origin_diff_kappa_matrix, rainbow_sse_diff_kappa_matrix,...
    rainbow_tem_diff_kappa_matrix, dqn_diff_kappa_matrix, amac_diff_kappa_matrix};
title_list = ["Performance Comparison Under Different Accuracy Reward Weights",...
    "Performance Comparison Under Different Delay Reward Weights",...
    "Performance Comparison Under Different Energy Consumption Reward Weights"];
for i=1:3
    figure(6+i)
    for j=1:6
        data_curr = data{j};
        % Data cleaning
        filtered_data_curr = remove_outliers(data_curr(:,:,i));
        x = filtered_data_curr(1,:); % delay
        y = filtered_data_curr(2,:); % energy
        z = filtered_data_curr(4,:); % acc vio prob
        plot3(x,y,z,'-o','LineWidth',2); hold on;


    end

    grid on;

    legend_names = {"Proposed (erf)", "Proposed (origin)", "SSE", "TEM", "DQN","AMAC"};
    legend(legend_names, 'Location', 'best');
    set(gca,'FontName','Times New Roman','FontSize',12);
    xlabelHandle = xlabel('Delay (S)'); 
    ylabelHandle = ylabel("Energy Consumption (J)"); 
    zlabelHandle = zlabel("Accuracy Violation Prob");
    title(title_list(i))
    set(gca,'gridlinestyle','--','Gridalpha',0.8);
end




% Scatter
labels = {'0.5', '1', '1.5', '2', '2.5','3','3.5','4'};
data = {rainbow_proposed_erf_diff_kappa_matrix, rainbow_proposed_origin_diff_kappa_matrix, rainbow_sse_diff_kappa_matrix,...
    rainbow_tem_diff_kappa_matrix, dqn_diff_kappa_matrix, amac_diff_kappa_matrix};
title_list = ["Performance Comparison Under Different Accuracy Reward Weights",...
    "Performance Comparison Under Different Delay Reward Weights",...
    "Performance Comparison Under Different Energy Consumption Reward Weights"];
marklist = ["+","o","*","x","p","d"];
for i=1:3
    figure(6+i)
    for j=1:6
        data_curr = data{j};
        % Data cleaning
        data_curr_array = zeros(size(data_curr));
        for k = 1:3
            data_curr_array(:,:,k) = cell2mat(data_curr(:,:,k));
        end
        filtered_data_curr = remove_outliers(data_curr_array(:,:,i));
        x = filtered_data_curr(1,:); % delay
        y = filtered_data_curr(2,:); % energy
        z = filtered_data_curr(4,:); % acc vio prob
        scatter3(x,y,z, 50,marklist(j),'LineWidth',2); hold on;
    end

    for j=1:6
        data_curr = data{j};
        % Data cleaning
                data_curr_array = zeros(size(data_curr));
        for k = 1:3
            data_curr_array(:,:,k) = cell2mat(data_curr(:,:,k));
        end
        filtered_data_curr = remove_outliers(data_curr_array(:,:,i));
        x = filtered_data_curr(1,:); % delay
        y = filtered_data_curr(2,:); % energy
        z = filtered_data_curr(4,:); % acc vio prob
        % for k = 1:length(x)
        %     text(x(k), y(k), z(k), labels{k}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
        % end
            for k=1:size(filtered_data_curr, 2)
            plot3([x(k), x(k)], [y(k), y(k)], [0, z(k)], "--","Color",colors(j,:),'LineWidth',0.5);  
            plot3([0, x(k)], [y(k), y(k)], [z(k), z(k)], "--","Color",colors(j,:),'LineWidth',0.5);  
            plot3([x(k), x(k)], [0, y(k)], [z(k), z(k)], "--","Color",colors(j,:),'LineWidth',0.5);  
            end
    end
    grid on;
       
    legend_names = {"Proposed (erf)", "Proposed (origin)", "SSE", "TEM", "DQN","AMAC"};
    legend(legend_names, 'Location', 'best');
    set(gca,'FontName','Times New Roman','FontSize',12);
    xlabelHandle = xlabel('Delay (S)'); 
    ylabelHandle = ylabel("Energy Consumption (J)"); 
    zlabelHandle = zlabel("Accuracy Violation Prob");
    title(title_list(i))
    set(gca,'gridlinestyle','--','Gridalpha',0.8);
    set(gca, 'ZScale', 'log');
end


% 2-D Version
labels = {'0.5', '1', '1.5', '2', '2.5','3','3.5','4'};
data = {rainbow_proposed_erf_diff_kappa_matrix, rainbow_proposed_origin_diff_kappa_matrix, rainbow_sse_diff_kappa_matrix,...
    rainbow_tem_diff_kappa_matrix, dqn_diff_kappa_matrix, amac_diff_kappa_matrix};
title_list = ["Performance Comparison Under Different Accuracy Reward Weights",...
    "Performance Comparison Under Different Delay Reward Weights",...
    "Performance Comparison Under Different Energy Consumption Reward Weights"];
marklist = ["+","o","*","x","p","d"];
for i=1:3
    figure();
    for j=1:6
        data_curr = data{j};
        % Data cleaning
        data_curr_array = zeros(size(data_curr));
        for k = 1:3
            data_curr_array(:,:,k) = cell2mat(data_curr(:,:,k));
        end
        filtered_data_curr = remove_outliers(data_curr_array(:,:,i));
        x = filtered_data_curr(1,:); % delay
        y = filtered_data_curr(2,:); % energy
        z = filtered_data_curr(4,:); % acc vio prob

        
        scatter(x,y, 50,marklist(j),'LineWidth',2); hold on;
    end
    grid on;
    legend_names = {"Proposed (erf)", "Proposed (origin)", "SSE", "TEM", "DQN","AMAC"};
    legend(legend_names, 'Location', 'best');
    set(gca,'FontName','Times New Roman','FontSize',12);
    xlabel('Delay (S)'); 
    ylabel("Energy Consumption (J)"); 
    title(title_list(i))
    set(gca,'gridlinestyle','--','Gridalpha',0.8);
    

    figure();
    for j=1:6
        data_curr = data{j};
        % Data cleaning
        data_curr_array = zeros(size(data_curr));
        for k = 1:3
            data_curr_array(:,:,k) = cell2mat(data_curr(:,:,k));
        end
        filtered_data_curr = remove_outliers(data_curr_array(:,:,i));
        x = filtered_data_curr(1,:); % delay
        y = filtered_data_curr(2,:); % energy
        z = filtered_data_curr(4,:); % acc vio prob
        scatter(x,z, 50,marklist(j),'LineWidth',2); hold on;
    end
    grid on;
    legend_names = {"Proposed (erf)", "Proposed (origin)", "SSE", "TEM", "DQN","AMAC"};
    legend(legend_names, 'Location', 'best');
    set(gca,'FontName','Times New Roman','FontSize',12);
    xlabel('Delay (S)');  
    ylabel("Accuracy Violation Prob");
    title(title_list(i))
    set(gca,'gridlinestyle','--','Gridalpha',0.8);
    set(gca, 'YScale', 'log');


    figure();
    for j=1:6
        data_curr = data{j};
        % Data cleaning
        data_curr_array = zeros(size(data_curr));
        for k = 1:3
            data_curr_array(:,:,k) = cell2mat(data_curr(:,:,k));
        end
        filtered_data_curr = remove_outliers(data_curr_array(:,:,i));
        x = filtered_data_curr(1,:); % delay
        y = filtered_data_curr(2,:); % energy
        z = filtered_data_curr(4,:); % acc vio prob
        scatter(y,z, 50,marklist(j),'LineWidth',2); hold on;
    end
    grid on;
    legend_names = {"Proposed (erf)", "Proposed (origin)", "SSE", "TEM", "DQN","AMAC"};
    legend(legend_names, 'Location', 'best');
    set(gca,'FontName','Times New Roman','FontSize',12);
    xlabel("Energy Consumption (J)"); 
    ylabel("Accuracy Violation Prob");
    title(title_list(i))
    set(gca,'gridlinestyle','--','Gridalpha',0.8);
    set(gca, 'YScale', 'log');
       
    
end

%% Fig.6 Different SNR
% load("experiments\diff_snr_data\diff_snr_data.mat")
load("experiments\diff_snr_data\rainbow_proposed_erf_diff_snr_matrix.mat")
load("experiments\diff_snr_data\rainbow_proposed_origin_diff_snr_matrix.mat")
load("experiments\diff_snr_data\rainbow_sse_diff_snr_matrix.mat")
load("experiments\diff_snr_data\rainbow_tem_diff_snr_matrix.mat")
load("experiments\diff_snr_data\dqn_diff_snr_matrix.mat")
load("experiments\diff_snr_data\amac_diff_snr_matrix.mat")

ylabel_list = ["Delay (S)", "Energy Consumption (J)", "Accuracy (mAP)", "Accuracy Violation Prob", "Re-transmission Number","Average Reward","Accuracy Violation"];
line_style_list = ["-+","-o","-*","-x","-p","-d","^"];
snr_num = size(snr_db_list, 2);
for i=1
    figure(9+i)
    % proposed erf
    mean_list = zeros(1,snr_num);
    std_list = zeros(1,snr_num);
    for j=1:snr_num
        mean_list(1,j) = mean(rainbow_proposed_erf_diff_snr_matrix{i,j});
        % std_list(1,j) = std(rainbow_proposed_erf_diff_snr_matrix{i,j});
    end
    % errorbar(snr_db_list, mean_list, std_list, line_style_list(1,1), 'LineWidth', 2,"Color",colors(1,:)); hold on;
    % shading_errorbar(snr_db_list, mean_list, std_list,line_style_list(1,1), colors(1,:)); hold on;
    plot(snr_db_list, mean_list,line_style_list(1,1),'LineWidth', 2,"Color",colors(1,:)); hold on;

        
    % proposed origin
    for j=1:snr_num
        mean_list(1,j) = mean(rainbow_proposed_origin_diff_snr_matrix{i,j});
        % std_list(1,j) = std(rainbow_proposed_origin_diff_snr_matrix{i,j});
    end
    % errorbar(snr_db_list, mean_list, std_list, line_style_list(1,2), 'LineWidth', 2,"Color",colors(2,:)); hold on;
    % shading_errorbar(snr_db_list, mean_list, std_list,line_style_list(1,2), colors(2,:));
    plot(snr_db_list, mean_list,line_style_list(1,2),'LineWidth', 2,"Color",colors(2,:)); hold on;

    % sse
    for j=1:snr_num
        mean_list(1,j) = mean(rainbow_sse_diff_snr_matrix{i,j});
        % std_list(1,j) = std(rainbow_sse_diff_snr_matrix{i,j});
    end
    % errorbar(snr_db_list, mean_list, std_list, line_style_list(1,3), 'LineWidth', 2,"Color",colors(3,:)); hold on;
    % shading_errorbar(snr_db_list, mean_list, std_list,line_style_list(1,3), colors(3,:));
    plot(snr_db_list, mean_list,line_style_list(1,3),'LineWidth', 2,"Color",colors(3,:)); hold on;


    % tem
    for j=1:snr_num
        mean_list(1,j) = mean(rainbow_tem_diff_snr_matrix{i,j});
        % std_list(1,j) = std(rainbow_tem_diff_snr_matrix{i,j});
    end
    % errorbar(snr_db_list, mean_list, std_list, line_style_list(1,4), 'LineWidth', 2,"Color",colors(4,:)); hold on;
    % shading_errorbar(snr_db_list, mean_list, std_list,line_style_list(1,4), colors(4,:));
    plot(snr_db_list, mean_list,line_style_list(1,4),'LineWidth', 2,"Color",colors(4,:)); hold on;


    % dqn
    for j=1:snr_num
        mean_list(1,j) = mean(dqn_diff_snr_matrix{i,j});
        % std_list(1,j) = std(dqn_diff_snr_matrix{i,j});
    end
    % errorbar(snr_db_list, mean_list, std_list, line_style_list(1,5), 'LineWidth', 2,"Color",colors(5,:)); hold on;
    % shading_errorbar(snr_db_list, mean_list, std_list,line_style_list(1,5), colors(5,:));
    plot(snr_db_list, mean_list,line_style_list(1,5),'LineWidth', 2,"Color",colors(5,:)); hold on;

    % amac
    for j=1:snr_num
        mean_list(1,j) = mean(amac_diff_snr_matrix{i,j});
        % std_list(1,j) = std(amac_diff_snr_matrix{i,j});
    end
    % errorbar(snr_db_list, mean_list, std_list, line_style_list(1,6), 'LineWidth', 2,"Color",colors(6,:)); hold on;
    % shading_errorbar(snr_db_list, mean_list, std_list,line_style_list(1,6), colors(6,:));
    plot(snr_db_list, mean_list,line_style_list(1,6),'LineWidth', 2,"Color",colors(6,:)); hold on;
    
    if i==4  % acc vio use log Y-axis
        set(gca, 'YScale', 'log')
    end

grid on;

    if i==3 % acc add aver_min_acc
        plot(snr_db_list, aver_min_acc*ones(size(snr_db_list)),"Color",[0 0 0],'LineWidth', 2); hold on;
        legend_names = {"Proposed (erf)", "Proposed (origin)", "SSE", "TEM", "DQN","AMAC","Accuracy Threshold"};
    else
        legend_names = {"Proposed (erf)", "Proposed (origin)", "SSE", "TEM", "DQN","AMAC"};
    end
legend(legend_names, 'Location', 'best');
set(gca,'FontName','Times New Roman','FontSize',12);
xlabel('SNR (dB)'),ylabel(ylabel_list(i));
set(gca,'gridlinestyle','--','Gridalpha',0.8);
end


%% Fig.7 Different Est Err
% load("experiments\diff_est_err_data\diff_est_err_data.mat")
load("experiments\diff_est_err_data\rainbow_proposed_erf_diff_est_err_matrix.mat")
load("experiments\diff_est_err_data\rainbow_proposed_origin_diff_est_err_matrix.mat")
load("experiments\diff_est_err_data\rainbow_sse_diff_est_err_matrix.mat")
load("experiments\diff_est_err_data\rainbow_tem_diff_est_err_matrix.mat")
load("experiments\diff_est_err_data\dqn_diff_est_err_matrix.mat")
load("experiments\diff_est_err_data\amac_diff_est_err_matrix.mat")
ylabel_list = ["Delay (S)", "Energy Consumption (J)", "Accuracy (mAP)", "Accuracy Violation Prob", "Re-transmission Number","Average Reward", "Accuracy Violation",];
y_max_list = [0.5,3,1,0.1,40000,0.5,0.1];
y_min_list = [0,0,0,0,0,-0.5,0];
est_err_num = 3;
for i=1
    figure(15+i)
    error_data = zeros(6, est_err_num);
    data = zeros(6,est_err_num);
    for j=1:est_err_num
        data(1,j) = mean(rainbow_proposed_erf_diff_est_err_matrix{i,j});
        % error_data(1,j) = std(rainbow_proposed_erf_diff_est_err_matrix{i,j});

        data(2,j) = mean(rainbow_proposed_origin_diff_est_err_matrix{i,j});
        % error_data(2,j) = std(rainbow_proposed_origin_diff_est_err_matrix{i,j});

        data(3,j) = mean(rainbow_sse_diff_est_err_matrix{i,j});
        % error_data(3,j) = std(rainbow_sse_diff_est_err_matrix{i,j});

        data(4,j) = mean(rainbow_tem_diff_est_err_matrix{i,j});
        % error_data(4,j) = std(rainbow_tem_diff_est_err_matrix{i,j});

        data(5,j) = mean(dqn_diff_est_err_matrix{i,j});
        % error_data(5,j) = std(dqn_diff_est_err_matrix{i,j});

        data(6,j) = mean(amac_diff_est_err_matrix{i,j});
        % error_data(6,j) = std(amac_diff_est_err_matrix{i,j});
    end

    data=data';
    X = [1,2,3];
    GO = bar(X,data,1,'EdgeColor','k','LineWidth',1); hold on;
    
    if i==4  % acc vio use log Y-axis
        set(gca, 'YScale', 'log')
    end

    % numGroups = size(data, 1);
    % numBars = size(data, 2);   
    % groupWidth = min(0.8, numBars / (numBars + 1.5));
    % for j = 1:numBars
    %     xCenter = X - groupWidth / 2 + (2 * j - 1) * groupWidth / (2 * numBars); hold on;
    %     errorbar(xCenter, data(:, j), error_data(j, :), 'k', 'LineStyle', 'none', 'LineWidth', 1);
    % end
    
    
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
    
    if i==3 % acc add aver_min_acc
        yline(aver_min_acc, "Color",[0 0 0], 'LineWidth', 2);hold on;
        legendData = {'Proposed (erf)', 'Proposed (origin)', 'SSE', 'TEM', 'DQN','AMAC','Accuracy Threshold'};
    else
        legendData = {'Proposed (erf)', 'Proposed (origin)', 'SSE', 'TEM', 'DQN','AMAC'};
    end

    % Draw the legend
    [legend_h, object_h, plot_h, text_str] = legendflex(GO, legendData, 'Padding', [2, 2, 10], 'FontSize', 11, 'Location', 'NorthWest');
    % object_h(1) is the first bar's text
    % object_h(2) is the second bar's text
    % object_h(3) is the first bar's patch
    % object_h(4) is the second bar's patch
    %
    % Set the two patches within the legend
    hatchfill2(object_h(7), 'cross', 'HatchAngle', 45, 'HatchDensity', 40, 'HatchColor', 'k');
    hatchfill2(object_h(8), 'single', 'HatchAngle', 45, 'HatchDensity', 40, 'HatchColor', 'k');
    hatchfill2(object_h(9), 'single', 'HatchAngle', 0, 'HatchDensity', 40, 'HatchColor', 'k');
    hatchfill2(object_h(10), 'single', 'HatchAngle', -45, 'HatchDensity', 40, 'HatchColor', 'k');
    hatchfill2(object_h(11), 'cross', 'HatchAngle', -60, 'HatchDensity', 30, 'HatchColor', 'k');
    hatchfill2(object_h(12), 'cross', 'HatchAngle', 60, 'HatchDensity', 30, 'HatchColor', 'k');
    % Some extra formatting to make it pretty :)
    set(gca,'FontName','Times New Roman','FontSize',12);
    set(gca,'gridlinestyle','--','Gridalpha',0.8);
    set(gca, 'XMinorTick','on', 'XMinorGrid','on', 'YMinorTick','on', 'YMinorGrid','on');
    
    % hTitle = title('Texture filled bar chart');
    hXLabel = xlabel('Channel Estimation Error Parameter');
    hYLabel = ylabel(ylabel_list(i));
    ax = gca;
    
    ylim([y_min_list(i), y_max_list(i)]);
    

    % hTitle = title('Texture filled bar chart');
    % hXLabel = xlabel('Samples');
    ax.XTickLabel = {'0','0.3','0.5'};


end