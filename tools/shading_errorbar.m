function shading_errorbar(x, y, err, lineStyle, lineColor)
 % PLOTWITHERRORSHADING Plot a curve with error shading
    %   x - x-axis data
    %   y - y-axis data
    %   err - error range
    %   lineColor - color of the curve (e.g., 'b', 'r', etc.)

    % Plot the curve with error shading
    hold on;

    % Construct the coordinates for the shaded area
    x_shaded = [x, fliplr(x)]; % x data, creating a closed region
    y_shaded = [y + err, fliplr(y - err)]; % y data, upper and lower limits

    % Use the fill function to draw the error shading
    fill(x_shaded, y_shaded, lineColor, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % Plot the original data curve
    plot(x, y,  lineStyle, 'Color', lineColor, 'LineWidth', 2,'MarkerSize', 8);

    % Configure plot settings
    xlabel('X Axis');
    ylabel('Y Axis');
    grid on;
    hold off;
end

   