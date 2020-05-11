x = zeros(3,1);
x(1) = <x0[0]>;
x(2) = <x0[1]>;
x(3) = <x0[2]>;
output = sum(x);
if ~ exist('OutputFiles', 'dir')
status = mkdir('OutputFiles');
end
csvwrite(sprintf('OutputFiles/oupt.out'),output);
