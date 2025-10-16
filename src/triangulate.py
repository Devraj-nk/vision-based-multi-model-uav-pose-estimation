import ast, numpy as np, cv2

def parse_projection_matrix(cell):
    # cell is string like '[[...], [...], ...]'
    mat = np.array(ast.literal_eval(cell), dtype=float)
    return mat  # 4x4 or whatever is stored

def cam_extrinsic_from_pos_quat(pos, quat):
    # pos: (x,y,z), quat: (x,y,z,w)
    import scipy.spatial.transform as sst
    r = sst.Rotation.from_quat(quat)  # [x,y,z,w]
    R = r.as_matrix()  # 3x3
    t = -R @ np.array(pos).reshape(3)
    # extrinsic [R|t]
    extr = np.hstack([R, t.reshape(3,1)])
    return extr

def triangulate_two_views(pt1_px, pt2_px, P1, P2):
    # pt*_px: (u,v) in pixels, P*: 3x4 projection matrices
    pts4 = cv2.triangulatePoints(P1, P2, np.array(pt1_px).reshape(2,1), np.array(pt2_px).reshape(2,1))
    pts3 = (pts4[:3] / pts4[3]).reshape(3)
    return pts3