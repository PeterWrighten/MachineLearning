
%kadai5-1

v = ones(8,1);
A = [v v+1 v+2 v+3 v+4 v+5 v+6 v+7];

for i = 1:8,
 for j = 1:8,
     for c = 1:8,
  if i + j == 10 - c || i + j == 18 - c,
    B(i,j) = c;
  endif;
end;
end;
end;

A,B

%kadai5-2

sum_AB = A + B
prod_AB = A * B
sub_AB = A - B

%kadai5-3

C = [1 1 1 0 1; 1 1 1 1 1; 1 0 1 1 0; 1 0 0 0 1];
D = [0 0 1; 1 0 1; 0 0 1; 1 1 0; 1 0 1];
Prod_GF2_CD = zeros(4,3);
for i = 1:4,
    for j = 1:5,
     for k = 1:3,
     Prod_GF2_CD(i,k) = xor(Prod_GF2_CD(i,k),C(i,j) * D(j,k));
 end;
end;
end;

Prod_GF2_CD
