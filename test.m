%% Fig.7 Different Est Err
load("experiments\diff_est_err_data\diff_est_err_data.mat")
ylabel_list = ["Average Delay Per Slot (S)", "Average Energy Consumption Per Slot (J)", "Average Accuracy Per Slot (mAP)", "Average Accuracy Violation Prob", "Average Re-transmission Number","Average Reward"];
y_max_list = [0.5,3,1,0.1,40000];
for i=6
    figure(14+i)
    data = zeros(6,3);
    data(1,:) = rainbow_env_proposed_erf_diff_est_err_matrix(i,:);
    data(2,:) = rainbow_env_proposed_origin_diff_est_err_matrix(i,:);
    data(3,:) = rainbow_env_sse_diff_est_err_matrix(i,:);
    data(4,:) = rainbow_env_tem_diff_est_err_matrix(i,:);
    data(5,:) = dqn_diff_est_err_matrix(i,:);
    data(6,:) = amac_diff_est_err_matrix(i,:);

data=data';
X = [1,2,3];
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
legendData = {'Proposed (erf)', 'Proposed (origin)', 'SSE', 'TEM', 'DQN','AMAC'};
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

% hTitle = title('Texture filled bar chart');
% hXLabel = xlabel('Samples');
ax.XTickLabel = {'0','0.3','0.5'};

end