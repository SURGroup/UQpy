function[] = matlab_model(index)

% Read the necessary .txt file
filename = sprintf('modelInput_%d.txt', index);
fileID_input = fopen(filename,'r');
formatSpec = '%f';
X = fscanf(fileID_input,formatSpec);

%Run the Model
QoI = sum(X);

% Save the results
filename = sprintf('solution_%d.txt', index);
fileID_input = fopen(filename,'w');
formatSpec = '%f';
fprintf(fileID_input,formatSpec, QoI);

end







