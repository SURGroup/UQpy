function output = dummy_model(index)
% disp(index)
x = csvread(sprintf('InputFiles/inpt_%d.inp', index));
output = sum(x);
if ~ exist('OutputFiles', 'dir')
status = mkdir('OutputFiles');
end
csvwrite(sprintf('OutputFiles/oupt_%d.out',index),output);
