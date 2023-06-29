from data import *
import os.path as osp
import open3d as o3d 

def make_point_cloud(pt3d: Array, colors: Optional[Array]=None, save_path:str=None):
    pt3d = convert_numpy(pt3d)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt3d)
    if colors is not None:
        colors = convert_numpy(colors)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if save_path: o3d.io.write_point_cloud(save_path, pcd)
    return pcd

def make_mesh(vertices:Array, triangles:Array, vertex_colors:Optional[Array]=None, save_path:str=None):
    vertices, triangles = convert_numpy(vertices),convert_numpy(triangles)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    if vertex_colors is not None:
        mesh.vertex_colors = convert_numpy(vertex_colors)
    if save_path: o3d.io.write_triangle_mesh(save_path, mesh)
    return mesh

if __name__ == "__main__":
    val = torch.rand(10)
    print(val)
    inds = torch.randint(10,size=(10,))
    print(inds)
    val_selected = torch.index_select(val,0,inds)
    print(val_selected)
    