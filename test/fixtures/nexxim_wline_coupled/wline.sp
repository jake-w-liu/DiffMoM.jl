* F1 W-element oracle: 2-conductor coupled RLGC line, all four ports driven
.option PARHIER='local'
W1 a1 a2 0 b1 b2 0 N=2 L=0.05 RLGCMODEL=cplw
.MODEL cplw W MODELTYPE=RLGC N=2 FDIE=1e9
+ Lo= 4.0e-7 1.0e-7 3.8e-7
+ Co= 1.3e-10 -3.5e-11 1.4e-10
+ Ro= 2.0 0.5 2.0
+ Go= 1.0e-5 -2.0e-6 1.2e-5
RPort1 a1 0 PORTNUM=1 RZ=50 IZ=0
.PORT a1 0 1 RPort1
RPort2 a2 0 PORTNUM=2 RZ=50 IZ=0
.PORT a2 0 2 RPort2
RPort3 b1 0 PORTNUM=3 RZ=50 IZ=0
.PORT b1 0 3 RPort3
RPort4 b2 0 PORTNUM=4 RZ=50 IZ=0
.PORT b2 0 4 RPort4
.LNA POI 2 0.5e9 1.5e9 flag='LNA'
.print LNA S(1,1) S(2,1) S(3,1) S(4,1) S(1,2) S(2,2) S(3,2) S(4,2) S(1,3) S(2,3) S(3,3) S(4,3) S(1,4) S(2,4) S(3,4) S(4,4)
.end
