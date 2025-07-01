
x = 73.482
p = 96.472
sd = 0.866

ss = 1 - ( x**2 / p**2 )
ss_min = 1 - ( (x - sd)**2 / p**2 )
ss_max = 1 - ( (x + sd)**2 / p**2 )
print (ss)
print(ss_min - ss_max)