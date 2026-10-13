* A3 composite: coupled launch (c1,c2 -> r1,r2) cascaded into DUT (r1,r2,d3)
* measured at cal plane ports c1 c2 d3
.option PARHIER='local'
RPort1 c1 0 PORTNUM=1 RZ=50 IZ=0
.PORT c1 0 1 RPort1
RPort2 c2 0 PORTNUM=2 RZ=50 IZ=0
.PORT c2 0 2 RPort2
RPort3 d3 0 PORTNUM=3 RZ=50 IZ=0
.PORT d3 0 3 RPort3
* launch lattice
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
* DUT junction
Ra r1 n 75
Rb r2 n 82
Rc d3 n 68
Rgd1 r1 0 500
Rgd2 r2 0 600
Rgd3 d3 0 700
Cd r1 d3 0.5e-12
.LNA POI 2 1e9 2e9 flag='LNA'
.print LNA S(1,1) S(2,1) S(3,1) S(1,2) S(2,2) S(3,2) S(1,3) S(2,3) S(3,3)
.end
