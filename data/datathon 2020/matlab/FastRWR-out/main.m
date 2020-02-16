%A = large;
%A = medium;
%A = small;
%A = tiny;
A = best
[Lam,M,time] = BBLin_Pre(A,[0.95 1]);
r=[]
for i = 1:12
    r = [r BBLin_OQ(M,Lam,0.95,i,5)]
end