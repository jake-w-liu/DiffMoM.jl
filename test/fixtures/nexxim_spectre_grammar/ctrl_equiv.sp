.param rin=50 rload=100
.subckt gainstage a b params: rg=50
Gm m 0 a 0 0.01
Rm m 0 rg
Ev b 0 m 0 1.0
Rl b 0 rload
.ends gainstage
.subckt top in out
X1 in n1 gainstage
Vs n1 n2 dc 0
F1 out 0 Vs 0.5
H1 n3 0 Vs 10
Rh n3 n2 5
C1 n2 out 1p
R2 out 0 rload
Ix out 0 dc 0
.ends top
