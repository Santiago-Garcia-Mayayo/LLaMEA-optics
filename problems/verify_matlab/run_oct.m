% run from the official Software/CEC2020 folder of P-N-Suganthan/2020-RW-Constrained-Optimisation
global initial_flag
probs = [3 10 12 13 14 15 16 17 18 19 21 24 25 26 28 30];
for n = probs
  X = load(sprintf('X%d.txt', n));
  [f, g, h] = cec20_func(X, n);
  M = [f, g, h];
  dlmwrite(sprintf('M%d.txt', n), M, 'precision', 17, 'delimiter', ' ');
end
disp('done');
