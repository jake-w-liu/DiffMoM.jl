* source library: dimensions in SI/SPICE engineering suffixes
.param scale = 2
.subckt Section a b ref PARAMS: rval=25 lval=1nH cval=2pF
.param seriesr = {rval * scale / 2}
+ pad = 4
Rseries A mid {seriesr}
Lseries mid B {lval}
Cshunt b REF {cval}
.ends SECTION
