* F2 Spectre-grammar oracle: controlled-source subckt equivalence
.option PARHIER='local'
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
Xtop in out top
RPort1 in 0 PORTNUM=1 RZ=50 IZ=0
.PORT in 0 1 RPort1
RPort2 out 0 PORTNUM=2 RZ=50 IZ=0
.PORT out 0 2 RPort2
.LNA POI 2 0.5e9 1.5e9 flag='LNA'
.print LNA S(1,1) S(2,1) S(1,2) S(2,2)
.end
