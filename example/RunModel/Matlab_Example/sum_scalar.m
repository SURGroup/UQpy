x = zeros(3,1);
x(1) = <var1>;
x(2) = <var11>;
x(3) = <var111>;
output = sum(x);
if ~ exist('OutputFiles', 'dir')
status = mkdir('OutputFiles');
end
csvwrite(sprintf('OutputFiles/oupt.out'),output);
