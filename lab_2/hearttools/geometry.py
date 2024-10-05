from functools import reduce

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from . import solver


def build_view_geometry(sid:float, sod:float, dp:float, alpha:float, beta:float, im_size:tuple[float]) -> dict[str, np.ndarray]:
	"""Builds the view geometry parameters from DICOM parameters

	Args:
	    sid(float): distance from source to detector (image plane) in mm
	    sod(float): distance from source to object (patient) in mm
	    dp(float): horizontal or vertical size of a pixel in mm
	    alpha(float): primary angle in degrees
	    beta(float): secondary angle in degrees
	    im_size(tuple[float]): array of 2 integers giving image size in pixels
	    sid:float: 
	    sod:float: 
	    dp:float: 
	    alpha:float: 
	    beta:float: 
	    im_size:tuple[float]: 

	Returns:
	    geometry(dict): dict with two keys ('source', 'detector'). The 'source' key contains a dictionary with parameters used to build the view, such as u0, v0, f, K, R, T, P, world_position, sid, sod and angles.

	
	"""
	t_matrix = np.array(
		[
			[1., 0., 0., 0.],
			[0., 1., 0., 0.],
			[0., 0., 1., -sod],
			[0., 0., 0., 1.]
		]
	)
	r_matrix = np.eye(4)
	r_matrix[:3, :3] = R.from_euler('xyz', [-beta, alpha, 0.], degrees=True).as_matrix()
	rc_matrix = r_matrix @ t_matrix
	ext_mat = np.linalg.inv(rc_matrix)[:-1, :]
	# Intrinsic parameters :
	u0 = im_size[0]/2 # Principal point (horiz. coord.)
	v0 = im_size[1]/2 # Principal point (vert. coord.)
	f = sid / dp # Focal length (in pixels)
	K = np.array([[f, 0., u0],[0., f, v0], [0., 0., 1.]]) # Intrinsic matrix
	P = K @ ext_mat

	# Detector parameters
	c_mat = np.array(
		[
			[1., 0., 0., 0.],
			[0., 1., 0., 0.],
			[0., 0., 1., sid-sod],
			[0., 0., 0., 1.]
		]
	)
	d_rc_matrix =    r_matrix @ c_mat
	out = {'source':
		     {
			     'u0':u0,
			     'v0':v0,
			     'f':f,
			     'T':ext_mat[:, [-1]],
			     'R':ext_mat[:3, :3],
			     'K':K,
			     'P':P,
			     'world_position':rc_matrix[:3, -1],
			     'sid':sid,
			     'sod':sod,
			     'angles':{
				     'alpha':alpha,
				     'beta':beta
			     }
		     },
		     'detector':
		     {
			     'world_position':d_rc_matrix[:3, -1]
		     }}
	return out

def to_homogeneous(points:np.ndarray) -> np.ndarray:
	"""Turns points to homogeneous coordinates

	Args:
	    points(np.ndarray): Array of points, shape (N, D)

	Returns:
	    points(np.ndarray): Homogeneous array of points, shape (N, D+1)
	
	"""
	return np.concatenate([points, np.ones((points.shape[0], 1))], 1)




def compute_8points_algorithm(im_1_points:np.ndarray, im_2_points:np.ndarray) -> np.ndarray:
	"""8-points algorithm, modification of https://github.com/Smelton01/8-point-algorithm

	Args:
	    im_1_points(np.ndarray): Points from image 1
	    im_2_points(np.ndarray): Corresponding points from image 2

	Returns:
	    F(np.ndarray): Fundamental matrix
	
	"""
	def __normalize(points):
		# T1 acts on x,y to give x_hat
		# T2 acts on x'y' to give x'_hat 
		n = len(points)
		img1_pts, img2_pts = [], []
		for a,b,c,d in points:
			img2_pts.append([a,b])
			img1_pts.append([c,d])
		sum1 = reduce(lambda x, y:    (x[0]+y[0], x[1]+y[1]), img1_pts)
		sum2 = reduce(lambda x, y:    (x[0]+y[0], x[1]+y[1]), img2_pts)
		
		mean1 = [val/n for val in sum1]
		mean2 = [val/n for val in sum2]

		s1 = (n*2)**0.5/(sum([((x-mean1[0])**2 + (y-mean1[1])**2)**0.5 for x,y in img1_pts]))
		s2 = (2*n)**0.5/(sum([((x-mean2[0])**2 + (y-mean2[1])**2)**0.5 for x,y in img2_pts]))

		T1 = np.array([[s1, 0, -mean1[0]*s1], [0, s1, -mean1[1]*s1], [0, 0, 1]])
		T2 = np.array([[s2, 0, -mean2[0]*s2], [0, s2, -mean2[1]*s2], [0, 0, 1]])

		points = [[T1 @ [c, d, 1], T2 @ [a,b,1]] for a,b,c,d in points]
		points = [[l[0], l[1], r[0], r[1]] for l,r in points]
		return points, T1, T2
	
	uv_mat = np.concatenate([im_2_points, im_1_points], 1)
	uv_mat, T1, T2 = __normalize(uv_mat)
	A = np.zeros((len(uv_mat),9))
	# img1 x' y' x y im2
	for i in range(len(uv_mat)):
		A[i][0] = uv_mat[i][0]*uv_mat[i][2]
		A[i][1] = uv_mat[i][1]*uv_mat[i][2]
		A[i][2] = uv_mat[i][2]
		A[i][3] = uv_mat[i][0]*uv_mat[i][3]
		A[i][4] = uv_mat[i][1]*uv_mat[i][3]
		A[i][5] = uv_mat[i][3] 
		A[i][6] = uv_mat[i][0]
		A[i][7] = uv_mat[i][1]
		A[i][8] = 1.0    
	
	_,_,v = np.linalg.svd(A)
	# print("v", v)
	f_vec = v.transpose()[:,8]
	# print("f_vec = ", f_vec)
	f_hat = np.reshape(f_vec, (3,3))
	# print("Fmat = ", f_hat)

	# Enforce rank(F) = 2 
	s,v,d = np.linalg.svd(f_hat)
	f_hat = s @ np.diag([*v[:2], 0]) @ d
	f_hat = T2.transpose() @ f_hat @ T1
	return f_hat

def refine_cam_param(x, X, K, R, t, rerr=2**-52, iter=30):
	"""Refines the camera parameters for 3D object-based calibration using the
	Levenberg-Marquardt nonlinear least squares algorithm.

	Args:
	    x(ndarray): nx2 array of image points (where n is the number of points).
	    X(ndarray): nx3 array of object points (where n is the number of points).
	    K(ndarray): Initial 3x3 camera intrinsic parameters matrix.
	    R(ndarray): Initial 3x3 rotation matrix.
	    t(ndarray): Initial 3x1 or 1x3 translation vector.
	    rerr(float, optional, optional): Relative error between the last and preceding iteration. Default is 2**-52.
	    iter(int, optional, optional): The number of maximum iterations. Default is 30.

	Returns:
	    K(ndarray): Refined 3x3 camera intrinsic parameters matrix.
	    R(ndarray): Refined 3x3 rotation matrix.
	    t(ndarray): Refined 3x1 translation vector.

	"""
	def reprojection_error(params, x, X, K):
		R = params[:9].reshape(3, 3)
		t = params[9:12].reshape(3, 1)
		
		# Project 3D points to 2D
		X_proj = (R @ X.T + t).T
		x_proj = (K @ X_proj.T).T
		x_proj = x_proj[:, :2] / x_proj[:, 2, np.newaxis]     
		return (np.linalg.norm(x - x_proj, axis=1)).ravel()
	
	# Initial parameters
	params = np.hstack((R.ravel(), t.ravel()))
	
	# Optimize
	result = least_squares(reprojection_error, params, args=(x, X, K), xtol=rerr, max_nfev=iter)
	
	# Extract refined parameters
	R_refined = result.x[:9].reshape(3, 3)
	t_refined = result.x[9:12].reshape(3, 1)
	
	return K, R_refined, t_refined

def calculate_3d_point(u0:float, v0:float, u1:float, v1:float, L0:np.ndarray, L1:np.ndarray) -> np.ndarray:
    """
    Calculates a 3D point from 2 2D coordinates pairs and L0 & L1 DLT calibration coefficients

    Parameters
    ----------
    u0 : float
        point 1 x-coordinate
    v0 : float
        point 1 y-coordinate
    u1 : float
        point 2 x-coordinate
    v1 : float
        point 2 y-coordinate
    L0 : np.ndarray
        DLT coefficients for view 1
    L1 : np.ndarray
        DLT coefficients for view 2
    """
    A = []

    A.append([u0*L0[8]-L0[0], u0*L0[9]-L0[1], L0[10]*u0-L0[2], -L0[3] + u0])
    A.append([v0*L0[8]-L0[4], v0*L0[9]-L0[5], L0[10]*v0-L0[6], -L0[7] + v0])
    A.append([u1*L1[8]-L1[0], u1*L1[9]-L1[1], L1[10]*u1-L1[2], -L1[3]+u1])
    A.append([v1*L1[8]-L1[4], v1*L1[9]-L1[5], L1[10]*v1-L1[6], -L1[7]+v1])
    
    return solver.dlt(np.array(A))

def __art(P):
    Q = np.linalg.inv(P[:3, :3])
    U, B = np.linalg.qr(Q)
    R = np.linalg.inv(U)
    t = np.dot(B, P[:3, 3])
    A = np.linalg.inv(B)
    A = A / A[2, 2]
    return A, R, t

def rectify(po1: np.ndarray, po2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rectify two projection matrices.

    Args:
        po1 (np.ndarray): First projection matrix.
        po2 (np.ndarray): Second projection matrix.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - Transformation matrix for the first projection matrix.
            - Transformation matrix for the second projection matrix.
            - New projection matrix for the first camera.
            - New projection matrix for the second camera.
    """
    a1, r1, _ = __art(po1)
    a2, _, _ = __art(po2)

    c1 = -np.linalg.inv(po1[:, :3]) @ po1[:, 3]
    c2 = -np.linalg.inv(po2[:, :3]) @ po2[:, 3]

    v1 = c1 - c2
    v2 = np.cross(r1[2, :], v1)
    v3 = np.cross(v1, v2)

    r = np.vstack([v1 / np.linalg.norm(v1),
                   v2 / np.linalg.norm(v2),
                   v3 / np.linalg.norm(v3)])

    a = (a1 + a2) / 2
    a[0, 1] = 0

    pn1 = a @ np.hstack([r, -r @ c1.reshape(-1, 1)])
    pn2 = a @ np.hstack([r, -r @ c2.reshape(-1, 1)])

    t1 = pn1[:3, :3] @ np.linalg.inv(po1[:3, :3])
    t2 = pn2[:3, :3] @ np.linalg.inv(po2[:3, :3])

    return t1, t2, pn1, pn2


def get_dicom_from_calib(k, r, t, dp):
    """
    Retrieve the DICOM parameters describing the view from the calibration parameters.

    Args:
        K (array_like): Matrix of intrinsic parameters (3x3).
        R (array_like): Rotation matrix (extrinsic) (3x3).
        T (array_like): Translation vector (extrinsic) (3x1 or 1x3).
        DP (float): Horizontal or vertical size of a pixel in mm.

    Returns:
        float: Distance from source to detector (image plane) in mm.
        float: Distance from source to object (patient) in mm.
        float: Primary angle in degrees.
        float: Secondary angle in degrees.

    Notes:
        - The parameter DP is assumed to be fixed and provided as input.
        - Coordinate systems are assumed to be the same as in BuildViewGeom().
        - The skew is assumed to be null (K[0, 1] == 0).
    """
    # Force R matrix to be orthogonal
    U, _, Vt = np.linalg.svd(r)
    R = np.dot(U, Vt)
    
    # Focal length (in pixels)
    f = np.mean([k[0, 0], k[1, 1]])
    
    # Source-detector distance (in mm)
    SID = f * dp
    
    # Source-object distance (in mm)
    SOD = np.linalg.norm(t)
    
    # Invert (transpose) R and decompose it into yaw, pitch, roll angles
    R = R.T
    
    # Assume that abs(b) <= PI/2 thus cos(b) >= 0
    cb = np.sqrt(np.sum(np.concatenate((R[0:2, 0], R[2, 1:3]))**2 / 2))
    sb = -R[2, 0]
    
    # Angle around Y == b == Primary angle
    alpha = np.degrees(np.arctan2(sb, cb))
    
    # Angle around X == c == Secondary angle
    if abs(alpha) == 90:
        # Assume here that angle a (around Z) is ~= 0
        beta = -np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
    else:
        beta = -np.degrees(np.arctan2(R[2, 1] / np.cos(np.radians(alpha)), R[2, 2] / np.cos(np.radians(alpha))))
    
    # TEST: angle around Z axis should be small (even negligible)
    if abs(alpha) == 90:
        delta = np.degrees(np.arctan2(R[1, 2], R[0, 2]))  # gives (a - c)
        a = delta + beta
    else:
        a = np.degrees(np.arctan2(R[1, 0] / np.cos(np.radians(alpha)), R[0, 0] / np.cos(np.radians(alpha))))
    
    return SID, SOD, alpha, beta