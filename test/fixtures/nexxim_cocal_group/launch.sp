* A3 coupled launch group: 4-port with cross-coupled resistive/reactive lattice
* port order: c1 c2 (cal side) | r1 r2 (raw/DUT side)
.option PARHIER='local'
RPort1 c1 0 PORTNUM=1 RZ=50 IZ=0
.PORT c1 0 1 RPort1
RPort2 c2 0 PORTNUM=2 RZ=50 IZ=0
.PORT c2 0 2 RPort2
RPort3 r1 0 PORTNUM=3 RZ=50 IZ=0
.PORT r1 0 3 RPort3
RPort4 r2 0 PORTNUM=4 RZ=50 IZ=0
.PORT r2 0 4 RPort4
Rs1 c1 r1 150
Rs2 c2 r2 200
Rx1 c1 r2 470
Rx2 c2 r1 330
Rc12 c1 c2 270
Rr12 r1 r2 390
Rg1 c1 0 1000
Rg2 c2 0 1200
Rg3 r1 0 1500
Rg4 r2 0 1100
C1 c1 0 2e-12
L2 c2 0 30e-9
.LNA POI 2 1e9 2e9 flag='LNA'
.print LNA S(1,1) S(2,1) S(3,1) S(4,1) S(1,2) S(2,2) S(3,2) S(4,2) S(1,3) S(2,3) S(3,3) S(4,3) S(1,4) S(2,4) S(3,4) S(4,4)
.end
