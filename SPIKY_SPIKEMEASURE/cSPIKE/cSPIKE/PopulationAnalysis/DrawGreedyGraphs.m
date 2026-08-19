figure('Position',[100 100 800 600]);
Matrix1 = TDpopulations{1}';
for i = 2:size(TDpopulations,2)
    Matrix1 = [Matrix1;TDpopulations{i}'];
end
subplot(2,2,1)
colormap('jet')
Matrix1(:,1:SPcoding) = Matrix1(:,1:SPcoding)*0.2;
Matrix1(:,SPcoding+1:SPcoding+LLcoding) = Matrix1(:,SPcoding+1:SPcoding+LLcoding)*0.5;
Matrix1(:,SPcoding+LLcoding+1:end) = Matrix1(:,SPcoding+LLcoding+1:end)*0.7;
imagesc(Matrix1)
NeuronsInPOP = sum(spGPOP);
yindex = 1:Neurons;
yindex = yindex(logical(spGPOP));
% Draw Xs for the best population
for x = 1:NeuronsInPOP
    DrawCross(yindex(x),10-NeuronsInPOP+1)
end
% If the best population is the best overall population: draw boxes around
% it
if(sum(spGPOP == spBPOP) == Neurons)
    for x = 1:NeuronsInPOP
        DrawBox(yindex(x),10-NeuronsInPOP+1)
    end
end
title('Top-Down: Selected Neurons')
ylabel('Iteration')
xlabel('Neuron')
colorbar;

Matrix2 = BUpopulations{1}';
for i = 2:size(BUpopulations,2)
    Matrix2 = [Matrix2;BUpopulations{i}'];
end
subplot(2,2,2)
colormap('jet')
Matrix2(:,1:SPcoding) = Matrix2(:,1:SPcoding)*0.2;
Matrix2(:,SPcoding+1:SPcoding+LLcoding) = Matrix2(:,SPcoding+1:SPcoding+LLcoding)*0.5;
Matrix2(:,SPcoding+LLcoding+1:end) = Matrix2(:,SPcoding+LLcoding+1:end)*0.7;
imagesc(Matrix2)
NeuronsInPOP = sum(spGBPOP);
yindex = 1:Neurons;
yindex = yindex(logical(spGBPOP));
% Draw Xs for the best population
for x = 1:NeuronsInPOP
    DrawCross(yindex(x),NeuronsInPOP)
end
% If the best population is the best overall population: draw boxes around
% it
if(sum(spGBPOP == spBPOP) == Neurons)
    for x = 1:NeuronsInPOP
        DrawBox(yindex(x),NeuronsInPOP)
    end
end
set(gca,'Ydir','normal')
title('Bottom-Up: Selected Neurons')
ylabel('Iteration')
xlabel('Neuron')
colorbar;

subplot(2,2,3)
colormap('jet')
imagesc(TDSUBS)
title('Top-Down: Discrimination Without N_i')
ylabel('Iteration')
xlabel('Neuron')
colorbar;

subplot(2,2,4)
colormap('jet')
imagesc(BUSUBS)
title('Bottom-Up: Discrimination With N_i')
ylabel('Iteration')
xlabel('Neuron')
set(gca,'Ydir','normal')
colorbar;