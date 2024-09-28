import numpy as np
from scipy.interpolate import CubicSpline

def interpolate(X:np.ndarray, n:int) -> np.ndarray:
    """
    Uses cubic spline interpolation to increase the number of points.

    Args:
        X (np.ndarray): Original points.
        n (int): Target number of points.

    Returns:
        np.ndarray: New interpolated points.
    """
    # Create the spline
    x = np.arange(1, len(X) + 1)
    cs = CubicSpline(x, X, axis=0)
    
    # Generate n points on the spline
    x_new = np.linspace(1, len(X), n)
    Y = cs(x_new)
    
    return Y

def sample_n_points(X:np.ndarray, n:int) -> np.ndarray:
    """
    Samples evenly spaced points from an array.

    Args:
        X (np.ndarray): Reference points, shape (N, D).
        n (int): Number of points to sample.

    Returns:
        np.ndarray: Sampled array.
    """
    idx = np.linspace(0, X.shape[0], endpoint=False, num=n, dtype = np.int64)
    return X[idx]

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