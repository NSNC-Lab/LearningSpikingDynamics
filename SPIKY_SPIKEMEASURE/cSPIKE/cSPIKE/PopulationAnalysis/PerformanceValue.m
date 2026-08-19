function [ performance,SMatrix,RMatrix,Distances,Statistics] = PerformanceValue( Matrix,S,R )
Intra = 0;
IntraCount = 0;
Inter = 0;
InterCount = 0;
Statistics = cell(S);
Distances = double(zeros(S));
for Sy = 1:S
    for Ry = 1:R
        for Sx = 1:S
            for Rx = 1:R
                % Calculating the index so that we only calculate each
                % pair once
                Xindex = (Sx-1)*R + Rx;
                Yindex = (Sy-1)*R + Ry;
                %Only calculating upper half of the matrix. Not taking the
                %equal index since we do not want to compare with itself.
                if Xindex > Yindex
                    % If both are sets for the same stimulus it is intra
                    % distance. If different it is inter.
                    if Sy == Sx
                        Intra = Intra + Matrix(Xindex, Yindex);
                        IntraCount = IntraCount+1;
                        
                    else
                        Inter = Inter + Matrix(Xindex, Yindex);
                        InterCount = InterCount+1;
                    end
                    if Distances(Sy,Sx) < Matrix(Xindex, Yindex)
                        Distances(Sy,Sx) = Matrix(Xindex, Yindex);
                    end
                    Statistics{Sy,Sx} = [Statistics{Sy,Sx} Matrix(Xindex, Yindex)];
                    
                end
                
            end
        end
    end
end
for Sy = 1:S
    for Sx = Sy:S
        RMatrix(Sy,Sx) = mean(Statistics{Sy,Sx});
        RMatrix(Sx,Sy) = RMatrix(Sy,Sx);
    end
end

%% Identify groups
SMatrix = ones(S);
for Sy = 1:S
    for Sx = Sy:S
        %alpha = 10^-10;
        %T1 = ttest2(Statistics{Sy,Sx},Statistics{Sy,Sy},'Alpha',alpha);
        %T2 = ttest2(Statistics{Sy,Sx},Statistics{Sx,Sx},'Alpha',alpha);
        %T3 = ttest2(Statistics{Sx,Sx},Statistics{Sy,Sy},'Alpha',alpha);
        %if ~T1&&~T2&&~T3
        %    SMatrix(Sy,Sx) = 0;
        %    SMatrix(Sx,Sy) = 0;
        %end
        alpha = 10^-4;
        [~,W1] = ranksum(Statistics{Sy,Sx},Statistics{Sy,Sy},'alpha',alpha);
        [~,W2] = ranksum(Statistics{Sy,Sx},Statistics{Sx,Sx},'alpha',alpha);
        [~,W3] = ranksum(Statistics{Sx,Sx},Statistics{Sy,Sy},'alpha',alpha);
        if ~W1&&~W2&&~W3
            SMatrix(Sy,Sx) = 0;
            SMatrix(Sx,Sy) = 0;
        end
    end
end

if SMatrix == ones(S)
    SMatrix = zeros(S);
end

performance = Inter/InterCount - Intra/IntraCount;

end

