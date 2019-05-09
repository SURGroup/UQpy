x = <var1>;
y = x^2;
fid = fopen(‘y.txt’,‘w’);
fprintf(fid, ‘%d’, y);
fclose(fid);