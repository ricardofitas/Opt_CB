import argparse
import os
import vtk


# Evita z-fighting entre linhas e faces
if hasattr(vtk.vtkMapper, "SetResolveCoincidentTopologyToPolygonOffset"):
    vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
    vtk.vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)
    # em algumas versões existe também line offset:
    if hasattr(vtk.vtkMapper, "SetResolveCoincidentTopologyLineOffsetParameters"):
        vtk.vtkMapper.SetResolveCoincidentTopologyLineOffsetParameters(-1.0, -1.0)

def load_stl(path: str) -> vtk.vtkPolyData:
    r = vtk.vtkSTLReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def make_feature_edges(poly: vtk.vtkPolyData, feature_angle: float) -> vtk.vtkPolyData:
    """
    Extrai arestas "CAD-like":
      - boundary edges (bordos)
      - feature edges (arestas com ângulo > feature_angle)
    """
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(poly)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOn()
    fe.ManifoldEdgesOff()
    fe.NonManifoldEdgesOn()
    fe.SetFeatureAngle(feature_angle)
    fe.Update()
    return fe.GetOutput()


def build_renderer(poly: vtk.vtkPolyData, edges: vtk.vtkPolyData) -> tuple[vtk.vtkRenderer, vtk.vtkRenderWindow]:
    # Actor sólido (serve sobretudo para o z-buffer / ocultação)
    solid_mapper = vtk.vtkPolyDataMapper()
    solid_mapper.SetInputData(poly)

    solid_actor = vtk.vtkActor()
    solid_actor.SetMapper(solid_mapper)
    solid_actor.GetProperty().LightingOff()
    solid_actor.GetProperty().SetColor(1, 1, 1)  # fundo branco, “invisível” na prática
    solid_actor.GetProperty().SetOpacity(1.0)

    # Actor de linhas (arestas)
    line_mapper = vtk.vtkPolyDataMapper()
    line_mapper.SetInputData(edges)

    line_actor = vtk.vtkActor()
    line_actor.SetMapper(line_mapper)
    line_actor.GetProperty().LightingOff()
    line_actor.GetProperty().SetColor(0, 0, 0)
    line_actor.GetProperty().SetLineWidth(1.2)

    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    ren.AddActor(solid_actor)
    ren.AddActor(line_actor)

    rw = vtk.vtkRenderWindow()
    rw.AddRenderer(ren)
    rw.SetSize(2000, 2000)
    rw.SetMultiSamples(0)  # linhas mais “crisp”

    # Isto é o “segredo” do estilo desenho técnico (linhas ocultas removidas)
    ren.UseHiddenLineRemovalOn()

    return ren, rw


def setup_ortho_camera(ren: vtk.vtkRenderer, poly: vtk.vtkPolyData, view: str):
    b = poly.GetBounds()  # xmin,xmax,ymin,ymax,zmin,zmax
    cx = (b[0] + b[1]) / 2.0
    cy = (b[2] + b[3]) / 2.0
    cz = (b[4] + b[5]) / 2.0
    dx = (b[1] - b[0])
    dy = (b[3] - b[2])
    dz = (b[5] - b[4])
    maxdim = max(dx, dy, dz) if max(dx, dy, dz) > 0 else 1.0
    dist = maxdim * 3.0

    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()

    if view == "xy":  # olhar ao longo de +Z
        cam.SetPosition(cx, cy, cz + dist)
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetViewUp(0, 1, 0)
        scale = max(dx, dy) / 2.0

    elif view == "xz":  # olhar ao longo de +Y
        cam.SetPosition(cx, cy + dist, cz)
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetViewUp(0, 0, 1)
        scale = max(dx, dz) / 2.0

    elif view == "yz":  # olhar ao longo de +X
        cam.SetPosition(cx + dist, cy, cz)
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetViewUp(0, 0, 1)
        scale = max(dy, dz) / 2.0

    else:
        raise ValueError("view tem de ser: xy, xz, yz")

    cam.SetParallelScale(scale * 1.05 if scale > 0 else 1.0)
    ren.ResetCameraClippingRange()


def export_vector(rw: vtk.vtkRenderWindow, out_path: str):
    """
    Exporta via GL2PS (vetorial). Funciona muito bem para “desenho”.
    Formatos comuns: svg, pdf, eps/ps (dependendo do build do VTK).
    """
    ext = os.path.splitext(out_path)[1].lower().lstrip(".")
    prefix = os.path.splitext(out_path)[0]

    exp = vtk.vtkGL2PSExporter()
    exp.SetRenderWindow(rw)
    exp.SetFilePrefix(prefix)
    exp.CompressOff()
    exp.SetSortToBSP()  # geralmente melhor para linhas

    # escolher formato
    if ext == "svg" and hasattr(exp, "SetFileFormatToSVG"):
        exp.SetFileFormatToSVG()
    elif ext == "pdf" and hasattr(exp, "SetFileFormatToPDF"):
        exp.SetFileFormatToPDF()
    elif ext in ("eps", "ps"):
        if hasattr(exp, "SetFileFormatToEPS") and ext == "eps":
            exp.SetFileFormatToEPS()
        else:
            exp.SetFileFormatToPS()
    else:
        raise ValueError(f"Formato vetorial não suportado aqui: .{ext} (tenta svg/pdf/eps/ps)")

    exp.Write()


def export_png(rw: vtk.vtkRenderWindow, out_path: str):
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetScale(1)
    w2i.SetInputBufferTypeToRGBA()
    w2i.ReadFrontBufferOff()
    w2i.Update()

    wr = vtk.vtkPNGWriter()
    wr.SetFileName(out_path)
    wr.SetInputConnection(w2i.GetOutputPort())
    wr.Write()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl", help="caminho para .stl")
    ap.add_argument("--out", default="drawing", help="prefixo do output (sem extensão)")
    ap.add_argument("--views", nargs="+", default=["xy", "xz", "yz"], choices=["xy", "xz", "yz"])
    ap.add_argument("--format", default="svg", choices=["svg", "pdf", "eps", "ps", "png"])
    ap.add_argument("--feature-angle", type=float, default=30.0, help="ângulo p/ 'feature edges' (graus)")
    ap.add_argument("--offscreen", action="store_true", help="render sem janela (útil em servidor)")
    args = ap.parse_args()

    poly = load_stl(args.stl)
    edges = make_feature_edges(poly, args.feature_angle)

    for v in args.views:
        ren, rw = build_renderer(poly, edges)
        if args.offscreen:
            rw.SetOffScreenRendering(1)

        setup_ortho_camera(ren, poly, v)

        rw.Render()  # importante antes de exportar

        out_path = f"{args.out}_{v}.{args.format}"
        if args.format == "png":
            export_png(rw, out_path)
        else:
            export_vector(rw, out_path)

        print("Gerado:", out_path)


if __name__ == "__main__":
    main()
