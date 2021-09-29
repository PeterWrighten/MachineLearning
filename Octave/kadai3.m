n = input('Enter n ');
x = input('Enter x ');

y = 1;
a = 1;
for i = 1:n
    a = a * x;
    y = y + a;
endfor
y
