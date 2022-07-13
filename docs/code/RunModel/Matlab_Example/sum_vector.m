x = [<x0>];
output = sum(x);
if ~ exist('OutputFiles', 'dir')
status = mkdir('OutputFiles');
end
csvwrite(sprintf('OutputFiles/oupt.out'),output);
