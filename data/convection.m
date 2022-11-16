clear all
clc
close all
%%%
% https://github.com/maziarraissi/DeepHPMs/blob/master/Matlab/gen_data_KS.m
% https://github.com/chebfun/examples/blob/master/pde/Kuramoto.m

nn = 511;
steps = 200;

dom = [0 2*pi]; x = chebfun('x',dom); tspan = linspace(0,1,steps+1);
S = spinop(dom, tspan);
S.lin = @(u) - 10*diff(u,1);
S.init = sin(x);
u = spin(S, nn, 1e-5,'plot','off');

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(0,2*pi,nn+1);
usol = [usol;usol(1,:)];
t = tspan;
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
save('convec_10.mat','t','x','usol')