bla = [1, 2, 3; 4 ,5 ,6; 7, 8, 9; 10, 11, 12];
directions = [1, 2, 3, 4];
% directions - right, down, left, up
numRow = size(bla, 1);
numCol = size(bla, 2);
result = [];

temp = [];
lastDirection = 0;
while 1
    if lastDirection == 0
        temp = bla(1, :);
        bla(1, :) = [];
    elseif lastDirection == 1
        temp = transpose(bla(:, 1));
        bla(:, 1) = [];
    end
    
    result = [result, temp];
    lastDirection = findDirection(lastDirection);
    if isempty(bla) == 1
        break;
    end
end

function direction = findDirection(lastDirection)
% direction: 0 - row, 1 - column
if lastDirection == 0
    direction = 1;
elseif lastDirection == 1
    direction = 0;
end
end
