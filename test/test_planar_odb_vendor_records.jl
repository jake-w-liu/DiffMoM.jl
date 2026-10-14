using DiffMoM,Test,LinearAlgebra

# Independent geometric oracles for the vendor-translator records: moire,
# cross/dogbone stencil symbols, and the B (UPC39 / Code 39) barcode.
# These do not share code with the production composite builders.

D=DiffMoM

function ref_moire(x,y,rw,rg,nr,lw,ll,la)
    r=hypot(x,y)
    for k in 1:nr
        inner=(k-1)*(rw+rg)
        inner<=r<=inner+rw && return true
    end
    lw>0 && ll>0 || return false
    # shape rotated -la (ODB clockwise): evaluate p in the unrotated frame
    th=deg2rad(la);c,s=cos(th),sin(th)
    fx,fy=c*x-s*y,s*x+c*y
    (abs(fx)<=ll/2 && abs(fy)<=lw/2) || (abs(fx)<=lw/2 && abs(fy)<=ll/2)
end

ref_cross_s(x,y,w,h,hs,vs,hc,vc)=
    (abs(x)<=w/2 && abs(y-vc)<=hs/2) || (abs(y)<=h/2 && abs(x-hc)<=vs/2)

capsule(x,y,ax,ay,bx,by,r)=hypot(x-clamp(x,min(ax,bx),max(ax,bx)),
    y-clamp(y,min(ay,by),max(ay,by)))<=r

ref_cross_r(x,y,w,h,hs,vs,hc,vc)=
    capsule(x,y,-(w-hs)/2,vc,(w-hs)/2,vc,hs/2) ||
    capsule(x,y,hc,-(h-vs)/2,hc,(h-vs)/2,vs/2)

function ref_dogbone(x,y,w,h,hs,vs,hc,round_ends)
    yoff=(h-hs)/2
    for sy in (-1.,1.)
        if round_ends
            capsule(x,y,-(w-hs)/2,sy*yoff,(w-hs)/2,sy*yoff,hs/2) && return true
        else
            abs(x)<=w/2 && abs(y-sy*yoff)<=hs/2 && return true
        end
    end
    round_ends ? capsule(x,y,hc,-yoff,hc,yoff,vs/2) : (abs(x-hc)<=vs/2 && abs(y)<=yoff+hs/2)
end

const _B_FONT="""
XSIZE 2
YSIZE 2
OFFSET 1
CHAR A
LINE 0 0 0 2 P R .2
LINE 0 2 2 2 P R .2
LINE 2 2 2 0 P R .2
ECHAR
CHAR B
LINE 0 0 0 2 P R .2
ECHAR
"""

function _vendor_features(text;font=nothing,kwargs...)
    mktempdir() do dir
        font===nothing || write(joinpath(dir,"standard"),font)
        feat=joinpath(dir,"features");write(feat,text)
        read_odb_features(feat;font_directory=dir,kwargs...)
    end
end

@testset "ODB moire symbol rings and crosshair" begin
    u=1e-3
    for (rw,rg,nr,lw,ll,la) in ((1.,.5,3,.3,7.,0.),(.8,.2,4,.4,6.,30.),(1.2,0.,2,.5,5.,75.),(2.,.4,1,.2,8.,45.))
        sh=D._odb_standard_symbol("moire$(rw)x$(rg)x$(nr)x$(lw)x$(ll)x$(la)",u)
        checked=0
        outer=rw+(nr-1)*(rw+rg)
        ex=max(outer,ll/2)*u*1.05;ey=max(outer,lw/2,ll/2)*u*1.05
        for i in 0:40,j in 0:40
            x=-ex+2ex*i/40;y=-ey+2ey*j/40
            D._artwork_contains(sh,x,y)==ref_moire(x,y,rw*u,rg*u,nr,lw*u,ll*u,la) && (checked+=1)
        end
        @test checked==41*41
    end
    sh=D._odb_standard_symbol("moire1x0.5x2x0.3x6x0",u)
    @test D._artwork_contains(sh,0.0005,0.)
    @test !D._artwork_contains(sh,0.00125,0.0008)  # gap, off the crosshair bars
    @test D._artwork_contains(sh,0.002,0.)      # ring 2 spans 1.5..2.5 mm
    @test_throws ArgumentError D._odb_standard_symbol("moire0x1x3x1x5x0",u)
    @test_throws ArgumentError D._odb_standard_symbol("moire1x1x0x1x5x0",u)
    @test D._odb_standard_symbol("moire1x1x3x1x5",u)===nothing
    @test D._odb_standard_symbol("moire1x1x3x1x5x",u)===nothing
end

@testset "ODB cross and dogbone stencil symbols" begin
    u=1e-3
    for (w,h,hs,vs,hc,vc) in ((10.,6.,2.,1.,0.,0.),(10.,6.,2.,1.,1.,0.5),(8.,5.,1.5,.8,-1.2,-.6))
        for (style,ref) in (("s",ref_cross_s),("r",ref_cross_r))
            sh=D._odb_standard_symbol("cross$(w)x$(h)x$(hs)x$(vs)x$(hc)x$(vc)x$style",u)
            checked=0
            for i in 0:30,j in 0:30
                x=(-w/2-1.2hs)*u+(w+2.4hs)*u*i/30
                y=(-h/2-1.2vs)*u+(h+2.4vs)*u*j/30
                D._artwork_contains(sh,x,y)==ref(x,y,w*u,h*u,hs*u,vs*u,hc*u,vc*u) && (checked+=1)
            end
            @test checked==31*31
        end
        @test_throws ArgumentError D._odb_standard_symbol("cross$(w)x$(h)x$(h)x$(vs)x$(hc)x$(vc)xs",u)
        @test_throws ArgumentError D._odb_standard_symbol("cross$(w)x$(h)x$(hs)x$(w)x$(hc)x$(vc)xs",u)
    end
    for style in ("s","r")
        sh=D._odb_standard_symbol("dogbone10x6x1.5x1x2x$style",u)
        checked=0
        for i in 0:30,j in 0:30
            x=-6u+12u*i/30;y=-4.5u+9u*j/30
            D._artwork_contains(sh,x,y)==ref_dogbone(x,y,10u,6u,1.5u,1u,2u,style=="r") && (checked+=1)
        end
        @test checked==31*31
    end
    @test_throws ArgumentError D._odb_standard_symbol("dogbone10x6x3.5x1x2xs",u)
    @test_throws ArgumentError D._odb_standard_symbol("dogbone10x6x1.5x10x2xs",u)
    # inside-corner fillets: all four cross notches dark only with ra
    shf=D._odb_standard_symbol("cross10x6x2x1x0x0xsx0.4",u)
    sh0=D._odb_standard_symbol("cross10x6x2x1x0x0xs",u)
    for (x,y) in ((0.7,1.2),(0.7,-1.2),(-0.7,1.2),(-0.7,-1.2))
        @test D._artwork_contains(shf,x*u,y*u)
        @test !D._artwork_contains(sh0,x*u,y*u)
    end
    # dogbone notches open toward the mid-plane on both sides of the connector
    dbf=D._odb_standard_symbol("dogbone10x6x1.5x1x2xsx0.5",u)
    db0=D._odb_standard_symbol("dogbone10x6x1.5x1x2xs",u)
    for (x,y) in ((2.7,1.3),(2.7,-1.3),(1.3,1.3),(1.3,-1.3))
        @test D._artwork_contains(dbf,x*u,y*u)
        @test !D._artwork_contains(db0,x*u,y*u)
    end
end

@testset "ODB UPC39 barcode records" begin
    u=.0254
    r=D._odb_barcode_record("B -1.7 0.25 UPC39 standard P 8 0 E 0.008 0.2 Y N Y Y T 'Barcode 1234567890';0")
    @test r.orientation==8 && r.angle==0. && r.text_top && r.add_string && r.background && r.full_ascii
    @test !r.checksum && r.width≈0.008 && r.height≈0.2 && r.positive
    @test_throws ArgumentError D._odb_barcode_record("B 0 0 CODE128 standard P 0 E 0.008 0.2 Y N N N B 'X'")
    @test_throws ArgumentError D._odb_barcode_record("B 0 0 UPC39 standard Q 0 E 0.008 0.2 Y N N N B 'X'")
    @test_throws ArgumentError D._odb_barcode_record("B 0 0 UPC39 standard P 0 F 0.008 0.2 Y N N N B 'X'")
    @test_throws ArgumentError D._odb_barcode_record("B 0 0 UPC39 standard P 0 E 0.008 0.2 Y N N N X 'X'")
    @test D._odb_code39_encode("A1",false)=="A1"
    @test D._odb_code39_encode("a1",true)=="+A1"
    @test D._odb_code39_encode("!",true)=="/A"
    @test D._odb_code39_encode("{",true)=="%P"
    @test_throws ArgumentError D._odb_code39_encode("a",false)
    doc=_vendor_features("UNITS=INCH\nF 1\nB -1.7 0.25 UPC39 standard P 0 E 0.008 0.2 N N N N B 'H'\n")
    @test length(doc.objects)==1
    o=doc.objects[1];@test o.dark
    x0,y0=-1.7u,.25u;n,h=0.008u,0.2u
    # '*' = nwnnwnwnn: bar intervals (narrow units) [0,1] [4,5] [6,9] [10,13] [14,15]
    for (pos,inside) in ((0.5,true),(2.5,false),(4.5,true),(7.,true),(15.5,false))
        @test D._artwork_contains(o.shape,x0+pos*n,y0+0.5h)==inside
    end
    # 'H' = wnnnnwwnn: starts at 16n; bars [0,3] [4,5] [6,7] [10,13] [14,15]
    hx=x0+16n
    @test D._artwork_contains(o.shape,hx+1n,y0+0.5h)
    @test D._artwork_contains(o.shape,hx+6.5n,y0+0.5h)
    @test D._artwork_contains(o.shape,hx+11n,y0+0.5h)
    @test !D._artwork_contains(o.shape,hx+5.5n,y0+0.5h)
    @test !D._artwork_contains(o.shape,hx+8.5n,y0+0.5h)
    @test !D._artwork_contains(o.shape,hx+13.5n,y0+0.5h)
    # checksum appends '1' (value 1 mod 43) at [32,47] before the stop '*'
    doc=_vendor_features("UNITS=INCH\nF 1\nB 0 0 UPC39 standard P 0 E 0.01 0.1 N Y N N B '1'\n")
    o=doc.objects[1];n2=0.01u
    @test D._artwork_contains(o.shape,33.5n2,0.05u)   # checksum '1' e1 wide bar
    @test D._artwork_contains(o.shape,45.5n2,0.05u)   # checksum '1' e9 wide bar
    doc=_vendor_features("UNITS=INCH\nF 1\nB 0 0 UPC39 standard P 0 E 0.01 0.1 N N N N B '1'\n")
    o=doc.objects[1]
    @test !D._artwork_contains(o.shape,33.5n2,0.05u)  # stop '*' space instead
    @test !D._artwork_contains(o.shape,45.5n2,0.05u)
    # inverted background: quiet zone clear, bars still dark
    doc=_vendor_features("UNITS=INCH\nF 1\nB 0 0 UPC39 standard P 0 E 0.01 0.1 N N Y N B '1'\n")
    o=doc.objects[1]
    @test !D._artwork_contains(o.shape,-5n2,0.05u)
    @test D._artwork_contains(o.shape,0.5n2,0.05u)
    # astr text adds extent above the bars
    doc=_vendor_features("UNITS=INCH\nF 1\nB 0 0 UPC39 standard P 0 E 0.01 0.1 N N N Y T 'AB'\n";font=_B_FONT)
    @test D._artwork_bounds(doc.objects[1].shape)[4]>0.1u
    # 90-degree CW rotation maps the first '*' bar [0,n]x[0,h] to x[0,h] y[-n,0]
    doc=_vendor_features("UNITS=INCH\nF 1\nB 0 0 UPC39 standard P 8 90 E 0.01 0.1 N N N N B 'H'\n")
    o=doc.objects[1]
    @test D._artwork_contains(o.shape,0.05u,-0.5n2)
    @test !D._artwork_contains(o.shape,0.05u,0.5n2)
end
