function [ string ] = POP2str( population )
string =  [];
for mark = 1:size(population,1)
    string = [string num2str(population(mark))];
end

end

