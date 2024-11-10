%If you want to adjust the pattern to 6 bar such as " applyhatch(gcf,'.-+/|x');",
%try to type this "applyhatch(gcf,'.-++/||xx');" instedly. 
%So you can avoid the duplicated pattern at least, even order problem is still not solved. 
data=[43.37,37.77,15.87;
      38.2, 33.63, 14.1;
      15.47,15.4,14.73;
      77.8,10.2,4.4;
      21.27,13.6,7.93;
      47.43,26.97,25.6];
X = [1,2,3,4,5,6];

GO = bar(X,data,1,'EdgeColor','k','LineWidth',1);

hatchfill2(GO(1),'cross','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(2),'single','HatchAngle',45,'HatchDensity',40,'HatchColor','k');
hatchfill2(GO(3),'single','HatchAngle',0,'HatchDensity',40,'HatchColor','k');

GO(1).FaceColor = colors(1,:);
GO(2).FaceColor = colors(2,:);
GO(3).FaceColor = colors(3,:);

% Draw the legend
legendData = {};
[legend_h, object_h, plot_h, text_str] = legendflex(GO, legendData, 'Padding', [2, 2, 10], 'FontSize', 11, 'Location', 'NorthWest');
% object_h(1) is the first bar's text
% object_h(2) is the second bar's text
% object_h(3) is the first bar's patch
% object_h(4) is the second bar's patch
%
% Set the two patches within the legend
hatchfill2(object_h(4), 'cross', 'HatchAngle', 45, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(5), 'single', 'HatchAngle', 45, 'HatchDensity', 40, 'HatchColor', 'k');
hatchfill2(object_h(6), 'single', 'HatchAngle', 0, 'HatchDensity', 40, 'HatchColor', 'k');
% Some extra formatting to make it pretty :)
set(gca,'FontName','Times New Roman','FontSize',12);
set(gca,'gridlinestyle','--','Gridalpha',0.8);
set(gca, 'XMinorTick','on', 'XMinorGrid','on', 'YMinorTick','on', 'YMinorGrid','on');
% xlim([0.5, 2.5]);
ylim([0, 100]);

% hTitle = title('Texture filled bar chart');
% hXLabel = xlabel('Samples');
hYLabel = ylabel('Action Pick Probability (%)');
ax = gca;
ax.XTickLabel = {'Proposed (erf)', 'Proposed (origin)', 'SSE', 'TEM', 'DQN','AMAC'};
