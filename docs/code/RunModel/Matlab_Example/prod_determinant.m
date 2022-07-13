x = <x0>;
y = [<x1>];
y = reshape(y,3,3)';
output = x*det(y);
if ~ exist('OutputFiles', 'dir')
status = mkdir('OutputFiles');
end
csvwrite(sprintf('OutputFiles/oupt.out'),output);
