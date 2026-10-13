.option PARHIER='local'
* ideal transformer 2:1 via unity-coupled inductors L1/L2=4
L1 in 0 1
L2 a 0 0.25
K1 L1 L2 1
RPort1 in 0 PORTNUM=1 RZ=50 IZ=0
.PORT in 0 1 RPort1
RPort2 a 0 PORTNUM=2 RZ=50 IZ=0
.PORT a 0 2 RPort2
.LNA POI 1 1e9 flag='LNA'
.print LNA S(1,1) S(2,1) S(1,2) S(2,2)
.PRINTLNAFMT CSV
