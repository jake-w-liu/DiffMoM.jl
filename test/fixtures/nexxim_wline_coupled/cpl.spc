* coupled-line smoke
.subckt cpl a1 a2 b1 b2
w1 a1 a2 0 b1 b2 0 n=2 l=0.05 rlgcmodel=cplw
.model cplw w modeltype=rlgc n=2 fdie=1e9
+ lo= 4.0e-7 1.0e-7 3.8e-7
+ co= 1.3e-10 -3.5e-11 1.4e-10
+ ro= 2.0 0.5 2.0
+ go= 1.0e-5 -2.0e-6 1.2e-5
.ends
