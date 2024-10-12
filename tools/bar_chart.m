%If you want to adjust the pattern to 6 bar such as " applyhatch(gcf,'.-+/|x');",
%try to type this "applyhatch(gcf,'.-++/||xx');" instedly. 
%So you can avoid the duplicated pattern at least, even order problem is still not solved. 
data=[100, 200, 900, 1400, 788;111, 222, 333, 444, 555];%三列的柱状图，分为了两种，具体见图
X = [1,2];

GO = bar(X,data,1,'EdgeColor','k','LineWidth',1);

hatchfill2(GO(1),'cross','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(2),'single','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(3),'single','HatchAngle',0,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(4),'single','HatchAngle',-45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(5),'cross','HatchAngle',-60,'HatchDensity',30,'HatchColor','k');

GO(1).FaceColor = [0.000, 0.447, 0.741];
GO(2).FaceColor = [0.850, 0.325, 0.098];
GO(3).FaceColor = [0.929, 0.694, 0.125];
GO(4).FaceColor = [0.494, 0.184, 0.556];
GO(5).FaceColor = [0.466, 0.674, 0.188];

% Draw the legend
legendData = {'Legend 1','Legend 2','Legend 3', 'Legend 4', 'Legend 5'};
[legend_h, object_h, plot_h, text_str] = legendflex(GO, legendData, 'Padding', [2, 2, 10], 'FontSize', 11, 'Location', 'NorthEast');
% object_h(1) is the first bar's text
% object_h(2) is the second bar's text
% object_h(3) is the first bar's patch
% object_h(4) is the second bar's patch
%
% Set the two patches within the legend
hatchfill2(object_h(6), 'cross', 'HatchAngle', 45, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(7), 'single', 'HatchAngle', 45, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(8), 'single', 'HatchAngle', 0, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(9), 'single', 'HatchAngle', -45, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(10), 'cross', 'HatchAngle', -60, 'HatchDensity', 30, 'HatchColor', 'k');
% Some extra formatting to make it pretty :)
set(gca, 'FontSize', 11);
set(gca, 'XMinorTick','on', 'XMinorGrid','on', 'YMinorTick','on', 'YMinorGrid','on');
% xlim([0.5, 2.5]);
ylim([0, 1500]);

% hTitle = title('Texture filled bar chart');
% hXLabel = xlabel('Samples');
hYLabel = ylabel('AAAA');
