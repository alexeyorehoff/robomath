function route = GradientBasedPlanner (f, start_coords, end_coords, max_its)
% требуется найти путь на плоскости на основании градиента функции f 
% входные данные:
%     start_coords и end_coords -- координаты начальной и конечной точек
%     max_its -- максимальное число возможных итераций 
% выходные данные:
%     route -- массив из 2 столбцов и n строк
%     каждая строка соответствует координатам x, y робота (по мере прохождения пути)

[gx, gy] = gradient (-f);

% *******************************************************************

step_size = 0.25;
result_margin = 0.5;

route = start_coords;
current_pos = start_coords;

for i = 1:max_its   
    grad = [interp2(gx, current_pos(1), current_pos(2), 'linear', 0), ... 
            interp2(gy, current_pos(1), current_pos(2), 'linear', 0)];

    current_pos = current_pos + step_size * grad / norm(grad);
    route = [route; current_pos];

    if norm(current_pos - end_coords) < result_margin
        break;
    end
end


% *******************************************************************

end
