x = <x0>;
y = [<x1[0, 0]>, <x1[0, 1]>, <x1[0, 2]>; <x1[1, 0]>, <x1[1, 1]>, <x1[1, 2]>; <x1[2, 0]>, <x1[2, 1]>, <x1[2, 2]>];
output = x*det(y);
if ~ exist('OutputFiles', 'dir')
status = mkdir('OutputFiles');
end
csvwrite(sprintf('OutputFiles/oupt.out'),output);
