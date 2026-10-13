# HFSS rebuild of the DXF combline filter (dxf_filter_pec.son)
# Stack: PEC cover | 0.1in air | Layer1 metal | 0.01in RO3010 | PEC ground
# Units: meters. 15 sheet rects at z=0.000254 + 11 vias to ground.
import ScriptEnv, os, traceback, sys

def _die(e):
    print("FATAL: " + str(e))
    traceback.print_exc()
    sys.stdout.flush()
    raise SystemExit(1)

ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.NewProject()
oProject.InsertDesign("HFSS", "combline", "DrivenModal", "")
oDesign = oProject.SetActiveDesign("combline")
oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits(["NAME:Units Parameter", "Units:=", "meter", "Rescale:=", False])

BOX = (0.01143, 0.00762, 0.002794)
SUB_H = 0.000254
MET_T = "1.778e-05"
SIGMA = "58000000"

oDM = oProject.GetDefinitionManager()
if not oDM.DoesMaterialExist("RO3010"):
    oDM.AddMaterial(["NAME:RO3010", "CoordinateSystemType:=", "Cartesian",
        "BulkOrSurfaceType:=", 1,
        ["NAME:PhysicsTypes", "set:=", ["Electromagnetic"]],
        "permittivity:=", "11.2", "dielectric_loss_tangent:=", "0.0022"])
print("materials ready")

def box(name, x0, y0, z0, dx, dy, dz, mat):
    oEditor.CreateBox(
        ["NAME:BoxParameters", "XPosition:=", str(x0), "YPosition:=", str(y0),
         "ZPosition:=", str(z0), "XSize:=", str(dx), "YSize:=", str(dy),
         "ZSize:=", str(dz)],
        ["NAME:Attributes", "Name:=", name, "Flags:=", "", "Color:=", "(132 132 193)",
         "Transparency:=", 0.75, "PartCoordinateSystem:=", "Global", "UDMId:=", "",
         "MaterialValue:=", '"' + mat + '"', "SurfaceMaterialValue:=", '""',
         "SolveInside:=", True, "ShellElement:=", False, "ShellElementThickness:=", "0mm",
         "IsMaterialEditable:=", True, "UseMaterialAppearance:=", False,
         "IsLightweight:=", False])

def rect_sheet(name, x0, y0, x1, y1, z):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    points = ["NAME:PolylinePoints"]
    for (px, py) in pts:
        points.append(["NAME:PLPoint", "X:=", repr(px), "Y:=", repr(py), "Z:=", repr(z)])
    segs = ["NAME:PolylineSegments"]
    for i in range(4):
        segs.append(["NAME:PLSegment", "SegmentType:=", "Line",
                     "StartIndex:=", i, "NoOfPoints:=", 2])
    oEditor.CreatePolyline(
        ["NAME:PolylineParameters",
         "IsPolylineCovered:=", True, "IsPolylineClosed:=", True,
         points, segs,
         ["NAME:PolylineXSection",
          "XSectionType:=", "None", "XSectionOrient:=", "Auto",
          "XSectionWidth:=", "0", "XSectionTopWidth:=", "0",
          "XSectionHeight:=", "0", "XSectionNumSegments:=", "0",
          "XSectionBendType:=", "Corner"]],
        ["NAME:Attributes", "Name:=", name, "Flags:=", "", "Color:=", "(143 175 143)",
         "Transparency:=", 0.0, "PartCoordinateSystem:=", "Global", "UDMId:=", "",
         "MaterialValue:=", '""', "SurfaceMaterialValue:=", '""',
         "SolveInside:=", False, "ShellElement:=", False, "ShellElementThickness:=", "0mm",
         "IsMaterialEditable:=", True, "UseMaterialAppearance:=", False,
         "IsLightweight:=", False])

def sheet2d(name, x0, y0, x1, y1, z):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    points = ["NAME:PolylinePoints"]
    for (px, py) in pts:
        points.append(["NAME:PLPoint", "X:=", repr(px), "Y:=", repr(py), "Z:=", repr(z)])
    segs = ["NAME:PolylineSegments"]
    for i in range(4):
        segs.append(["NAME:PLSegment", "SegmentType:=", "Line",
                     "StartIndex:=", i, "NoOfPoints:=", 2])
    oEditor.CreatePolyline(
        ["NAME:PolylineParameters",
         "IsPolylineCovered:=", True, "IsPolylineClosed:=", True,
         points, segs,
         ["NAME:PolylineXSection",
          "XSectionType:=", "None", "XSectionOrient:=", "Auto",
          "XSectionWidth:=", "0", "XSectionTopWidth:=", "0",
          "XSectionHeight:=", "0", "XSectionNumSegments:=", "0",
          "XSectionBendType:=", "Corner"]],
        ["NAME:Attributes", "Name:=", name, "Flags:=", "", "Color:=", "(175 143 143)",
         "Transparency:=", 0.0, "PartCoordinateSystem:=", "Global", "UDMId:=", "",
         "MaterialValue:=", '""', "SurfaceMaterialValue:=", '""',
         "SolveInside:=", False, "ShellElement:=", False, "ShellElementThickness:=", "0mm",
         "IsMaterialEditable:=", True, "UseMaterialAppearance:=", False,
         "IsLightweight:=", False])

def wall_sheet(name, xp):
    pts = [(0, 0), (BOX[1], 0), (BOX[1], BOX[2]), (0, BOX[2])]
    points = ["NAME:PolylinePoints"]
    for (py, pz) in pts:
        points.append(["NAME:PLPoint", "X:=", repr(xp), "Y:=", repr(py), "Z:=", repr(pz)])
    segs = ["NAME:PolylineSegments"]
    for i in range(4):
        segs.append(["NAME:PLSegment", "SegmentType:=", "Line",
                     "StartIndex:=", i, "NoOfPoints:=", 2])
    oEditor.CreatePolyline(
        ["NAME:PolylineParameters",
         "IsPolylineCovered:=", True, "IsPolylineClosed:=", True,
         points, segs,
         ["NAME:PolylineXSection",
          "XSectionType:=", "None", "XSectionOrient:=", "Auto",
          "XSectionWidth:=", "0", "XSectionTopWidth:=", "0",
          "XSectionHeight:=", "0", "XSectionNumSegments:=", "0",
          "XSectionBendType:=", "Corner"]],
        ["NAME:Attributes", "Name:=", name, "Flags:=", "", "Color:=", "(0 0 132)",
         "Transparency:=", 0.5, "PartCoordinateSystem:=", "Global", "UDMId:=", "",
         "MaterialValue:=", '""', "SurfaceMaterialValue:=", '""',
         "SolveInside:=", False, "ShellElement:=", False, "ShellElementThickness:=", "0mm",
         "IsMaterialEditable:=", True, "UseMaterialAppearance:=", False,
         "IsLightweight:=", False])

try:
    box("substrate", 0, 0, 0, BOX[0], BOX[1], SUB_H, "RO3010")
    box("air", 0, 0, SUB_H, BOX[0], BOX[1], BOX[2] - SUB_H, "vacuum")
    print("boxes done")

    sheets = [
        ("m159", 0.0, 0.00472439976378, 0.0010160000541019998, 0.004953),
        ("m157", 0.001015999972568, 0.0047244, 0.00198119994158, 0.0049530000838199995),
        ("m158", 0.00944879955804, 0.0047244, 0.010414, 0.0049530000838199995),
        ("m161", 0.010414, 0.0047244, 0.01143, 0.004953),
        ("m163", 0.0062991999873, 0.001016000243078, 0.0066547999517400005, 0.00685800053848),
        ("m164", 0.00703579958598, 0.000736600055372, 0.00739139955042, 0.00657860035052),
        ("m165", 0.00403859994666, 0.000736600055372, 0.0043941999111, 0.00657860035052),
        ("m166", 0.009093199593599999, 0.000889000364236, 0.00944879955804, 0.00685800053848),
        ("m167", 0.00477519992634, 0.001016000243078, 0.005130799890779999, 0.00685800053848),
        ("m168", 0.00198119994158, 0.000889000364236, 0.0023367999052579997, 0.00685800053848),
        ("m169", 0.0033274000203199995, 0.001016000243078, 0.00368299998476, 0.00685800053848),
        ("m170", 0.007747000271779999, 0.001016000243078, 0.00810260023622, 0.00685800053848),
        ("m171", 0.0026415999542799998, 0.000736600055372, 0.00299719991872, 0.00657860035052),
        ("m172", 0.0055371999568199996, 0.000736600055372, 0.0058927999212599995, 0.00657860035052),
        ("m173", 0.00843279976886, 0.000736600055372, 0.0087883997333, 0.00657860035052),
    ]
    names = [s[0] for s in sheets]
    for (nm, x0, y0, x1, y1) in sheets:
        sheet2d(nm, x0, y0, x1, y1, SUB_H)
    print("sheets done")

    oBnd = oDesign.GetModule("BoundarySetup")
    # Sonnet NOR thin metal: 0.5oz copper, sigma=5.8e7 S/m, t=0.0178mm.
    # Finite-conductivity boundary reproduces Sonnet's surface-impedance
    # sheet model without volume-meshing copper.
    fincond_args = ["NAME:finsheet", "Objects:=", names,
        "UseMaterial:=", False, "Conductivity:=", "58000000",
        "Permeability:=", "1", "UseThickness:=", True,
        "Thickness:=", "0.0178mm", "Roughness:=", "0",
        "InfGroundPlane:=", False]
    try:
        oBnd.AssignFiniteCond(fincond_args)
    except Exception:
        oBnd.AssignFiniteConductivity(fincond_args)
    print("finite conductivity assigned")

    vias = [
        ("v175", 0.002158999923292, 0.006477000144779999, 0.00012700006807200002),
        ("v176", 0.00571499993777, 0.001117600448945, 0.00012699987934999997),
        ("v177", 0.004952999908559999, 0.006477000144779999, 0.00012699987808000024),
        ("v178", 0.00647699996825, 0.006477000144779999, 0.00012699987934999997),
        ("v179", 0.0035052000012699997, 0.006477000144779999, 0.00012699987934999997),
        ("v180", 0.00792480025273, 0.006477000144779999, 0.00012699987934999997),
        ("v181", 0.00421639993015, 0.001117600448945, 0.00012699987934999997),
        ("v182", 0.00721359956947, 0.001117600448945, 0.00012699987934999997),
        ("v183", 0.00861059975108, 0.001117600448945, 0.00012699987807999937),
        ("v184", 0.0028193999365, 0.001117600448945, 0.0001270000685799999),
        ("v185", 0.00927099957455, 0.006477000144779999, 0.00012699987934999997),
    ]
    for (nm, cx, cy, r) in vias:
        oEditor.CreateCylinder(
            ["NAME:CylinderParameters", "XCenter:=", repr(cx), "YCenter:=", repr(cy),
             "ZCenter:=", "0.0", "Radius:=", repr(r),
             "Height:=", repr(SUB_H), "WhichAxis:=", "Z", "NumSides:=", 0],
            ["NAME:Attributes", "Name:=", nm, "Flags:=", "", "Color:=", "(143 175 143)",
             "Transparency:=", 0.0, "PartCoordinateSystem:=", "Global", "UDMId:=", "",
             "MaterialValue:=", '"copper"', "SurfaceMaterialValue:=", '""',
             "SolveInside:=", False, "ShellElement:=", False, "ShellElementThickness:=", "0mm",
             "IsMaterialEditable:=", True, "UseMaterialAppearance:=", False,
             "IsLightweight:=", False])
    via_names = [v[0] for v in vias]
    # Vias (Sonnet VOL metal) live inside the substrate only; subtract them
    # from the substrate solid, keeping the via bodies (surface-solved).
    oEditor.Subtract(
        ["NAME:Selections", "Blank Parts:=", "substrate",
         "Tool Parts:=", ",".join(via_names)],
        ["NAME:SubtractParameters", "KeepOriginals:=", True])
    print("vias done")

    yc = 0.0048387
    for (pn, xp) in (("p1", 0.0), ("p2", BOX[0])):
        wall_sheet(pn, xp)
        oBnd.AssignWavePort(
            ["NAME:" + pn, "Objects:=", [pn], "NumModes:=", 1,
             "UseLineModeAlignment:=", False,
             ["NAME:Modes",
              ["NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True,
               ["NAME:IntLine",
                "Start:=", [repr(xp), repr(yc), repr(SUB_H)],
                "End:=", [repr(xp), repr(yc), "0.0"]],
               "CharImp:=", "Zpi", "RenormImp:=", "50ohm"]],
             "DoDeembed:=", False])
    print("ports done")

    # All remaining outer faces take HFSS's default outer boundary (PEC);
    # only the two wave-port walls are non-PEC.
    print("boundaries done")

    # Seed a fine surface mesh on the copper body for the resonant edges.
    oMesh = oDesign.GetModule("MeshSetup")
    oMesh.AssignLengthOp(
        ["NAME:cuseed", "RefineInside:=", False, "Enabled:=", True,
         "Objects:=", names + via_names, "RestrictElem:=", False,
         "NumMaxElem:=", "100000", "RestrictLength:=", True,
         "MaxLength:=", "0.25mm"])
    oMesh.AssignLengthOp(
        ["NAME:subseed", "RefineInside:=", True, "Enabled:=", True,
         "Objects:=", ["substrate"], "RestrictElem:=", False,
         "NumMaxElem:=", "150000", "RestrictLength:=", True,
         "MaxLength:=", "0.7mm"])
    print("mesh seed done")

    oAnalysis = oDesign.GetModule("AnalysisSetup")
    oAnalysis.InsertSetup("HfssDriven",
        ["NAME:Setup1", "SolveType:=", "Single", "Frequency:=", "5GHz",
         "MaxDeltaS:=", 0.03, "MaximumPasses:=", 25, "MinimumPasses:=", 2,
         "MinimumConvergedPasses:=", 1, "PercentRefinement:=", 30,
         "IsEnabled:=", True, "BasisOrder:=", 1, "UseIterativeSolver:=", False])
    oAnalysis.InsertFrequencySweep("Setup1",
        ["NAME:Sweep", "IsEnabled:=", True, "RangeType:=", "LinearStep",
         "RangeStart:=", "4.5GHz", "RangeEnd:=", "5.5GHz", "RangeStep:=", "50MHz",
         "Type:=", "Interpolating", "SaveFields:=", False, "SaveRadFields:=", False])
    print("setup done; analyzing")

    oDesign.Analyze("Setup1")
    print("solve done")

    out = os.path.dirname(os.path.abspath(__file__))
    oRep = oDesign.GetModule("ReportSetup")
    oRep.CreateReport("S Table", "Modal Solution Data", "Data Table",
        "Setup1 : Sweep", ["Domain:=", "Sweep"], ["Freq:=", ["All"]],
        ["X Component:=", "Freq",
         "Y Component:=", ["re(S(1,1))", "im(S(1,1))", "re(S(2,1))", "im(S(2,1))",
                           "re(S(1,2))", "im(S(1,2))", "re(S(2,2))", "im(S(2,2))"]])
    oRep.ExportToFile("S Table", os.path.join(out, "hfss_s.csv"))
    try:
        oRep.CreateReport("Zo", "Modal Solution Data", "Data Table",
            "Setup1 : LastAdaptive", ["Domain:=", "Sweep"], ["Freq:=", ["All"]],
            ["X Component:=", "Freq",
             "Y Component:=", ["re(PortZo(1:1))", "re(PortZo(2:1))"]])
        oRep.ExportToFile("Zo", os.path.join(out, "hfss_zo.csv"))
        print("zo export done")
    except Exception as e:
        print("zo export failed:", e)
    try:
        oRep.CreateReport("S Single", "Modal Solution Data", "Data Table",
            "Setup1 : LastAdaptive", ["Domain:=", "Sweep"], ["Freq:=", ["All"]],
            ["X Component:=", "Freq",
             "Y Component:=", ["re(S(1,1))", "im(S(1,1))", "re(S(2,1))", "im(S(2,1))",
                               "re(S(1,2))", "im(S(1,2))", "re(S(2,2))", "im(S(2,2))"]])
        oRep.ExportToFile("S Single", os.path.join(out, "hfss_s_last.csv"))
        print("single export done")
    except Exception as e:
        print("single export failed:", e)
    oProject.SaveAs(os.path.join(out, "combline.aedt"), True)
    print("saved")
except Exception as e:
    _die(e)
