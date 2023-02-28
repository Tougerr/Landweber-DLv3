function [Z_normalized] = normalize(Z)
Z_normalized = (Z-min(min(Z)))/(max(max(Z))-min(min(Z)));
end