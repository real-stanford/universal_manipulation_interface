import numpy as np
import scipy.spatial.transform as st
from umi.common.pose_util import rot6d_to_mat, mat_to_rot6d

# converting full name to scipy Rotation name
scipy_rep_map = {
    'axis_angle': 'rotvec',
    'quaternion': 'quat',
    'matrix': 'matrix',
    'rotation_6d': None
}

def transform_rotation(x, from_rep, to_rep):
    from_rep = scipy_rep_map[from_rep]
    to_rep = scipy_rep_map[to_rep]
    if from_rep is not None and to_rep is not None:
        # scipy rotation transform
        rot = getattr(st.Rotation, f'from_{from_rep}')(x)
        out = getattr(rot, f'as_{to_rep}')()
        return out
    else:
        mat = None
        if from_rep is None:
            mat = rot6d_to_mat(x)
        else:
            mat = getattr(st.Rotation, f'from_{from_rep}')(x).as_matrix()
        
        if to_rep is None:
            out = mat_to_rot6d(mat)
        else:
            out = getattr(st.Rotation.from_matrix(mat), f'as_{to_rep}')()
        return out
            

class RotationTransformer:
    # for legacy compatibility
    def __init__(self, 
            from_rep='axis_angle', 
            to_rep='rotation_6d'):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        self.from_rep = from_rep
        self.to_rep = to_rep
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return transform_rotation(x, from_rep=self.from_rep, to_rep=self.to_rep)
    
    def inverse(self, x: np.ndarray) -> np.ndarray:
        return transform_rotation(x, from_rep=self.to_rep, to_rep=self.from_rep)


def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2*np.pi,2*np.pi,size=(1000,3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation
    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-7

    tf = RotationTransformer('rotation_6d', 'matrix')
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotaiton_6d will be normalized to rotation matrix
