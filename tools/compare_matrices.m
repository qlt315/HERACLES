function [relative_diff, min_value] = compare_matrices(matrix_a, matrix_b)
    % Compare two matrices and compute the relative improvement/reduction
    % between each corresponding element, as well as the minimum improvement/reduction value.
    %
    % Parameters:
    %   matrix_a (MxN matrix): The first matrix (algorithm A metrics).
    %   matrix_b (MxN matrix): The second matrix (algorithm B metrics).
    %
    % Returns:
    %   relative_diff (MxN matrix): Matrix of relative differences (%).
    %   min_value (scalar): Minimum improvement/reduction value across all metrics.
    
    % Calculate the relative difference
    relative_diff = ((matrix_a - matrix_b) ./ abs(matrix_b)) * 100;
    
    % Find the minimum improvement/reduction
    min_value = min(relative_diff(:));
end

