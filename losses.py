from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing
import torch

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	loss = torch.binary_cross_entropy_with_logits(voxel_src, voxel_tgt)
	return loss.mean()

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	src_to_tgt = knn_points(point_cloud_src, point_cloud_tgt, K=1, norm=2)
	tgt_to_src = knn_points(point_cloud_tgt, point_cloud_src, K=1, norm=2)

	return src_to_tgt.dists.mean() + tgt_to_src.dists.mean()

def smoothness_loss(mesh_src):
	# implement laplacian smoothening loss
    return mesh_laplacian_smoothing(mesh_src)
