x = input('Enter a x: ');
S = pi * x.^2;
C = 2 * pi * x;
sprintf('円の面積は %d, 円周は %d',S,C)

y = input('Enter a y: ');
if y == 1,
r = 1:20;
s = pi * r.^2;
plot(r,s);
end;
