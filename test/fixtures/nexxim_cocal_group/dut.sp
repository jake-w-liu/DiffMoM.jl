* A3 DUT: 3-port resistive/reactive junction, port order r1 r2 d3
.option PARHIER='local'
RPort1 r1 0 PORTNUM=1 RZ=50 IZ=0
.PORT r1 0 1 RPort1
RPort2 r2 0 PORTNUM=2 RZ=50 IZ=0
.PORT r2 0 2 RPort2
RPort3 d3 0 PORTNUM=3 RZ=50 IZ=0
.PORT d3 0 3 RPort3
Ra r1 n 75
Rb r2 n 82
Rc d3 n 68
Rg1 r1 0 500
Rg2 r2 0 600
Rg3 d3 0 700
Cd r1 d3 0.5e-12
.LNA POI 2 1e9 2e9 flag='LNA'
.print LNA S(1,1) S(2,1) S(3,1) S(1,2) S(2,2) S(3,2) S(1,3) S(2,3) S(3,3)
.end
