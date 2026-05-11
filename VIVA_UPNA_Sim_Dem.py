from pathlib import Path
import numpy as np
from skspatial.objects import Line, Sphere, LineSegment, Plane, Point
import matplotlib.pyplot as plt
import pyvista as pv
import json
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, least_squares

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"

# Auxiliar functions

def rot_mat_align_vectors(v1: list[float], v2: list[float]) -> np.ndarray:
    '''
    This function calculates the rotation matrix such that, when applied to v1, aligns it with v2

    Parameters
    ----------
    v1 : list[float]
        First vector
    v2 : list[float]
        Second vector

    Returns
    -------
    R : np.ndarray
        Rotation matrix that aligns v1 with v2
    '''
    # We calculate one rotation matrix that transforms v1 into v2
    # We suppose that v1 and v2 are unitary vectors
    # The axis of rotation is the cross product of v1 and v2. This is only valid if v1 or v2 are in the primary position
    v1 = np.array(v1) / np.linalg.norm(v1)
    v2 = np.array(v2) / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if s == 0:
        R = np.eye(3)
    elif c == -1:
        R = -np.eye(3)
    else:
        R = np.eye(3) + v_x + np.dot(v_x, v_x) * (1 - c) / (s ** 2)
    return R


# Core classes

class Eye:

    def __init__(self, lor: str):
        self.lor = lor
        self.r_sclera = 0.0                 # Radius of the sclera in mm
        self.r_retina = 0.0                 # Radius of the retina in mm
        self.r_cornea = 0.0                 # Radius of the cornea in mm
        self.horizontal_offset = 0.0        # Absolute value of the horizontal offset between optical and visual axis in radians when the eye
                                            # is in the primary position. Positive means optical axis is to the temporal side
        self.vertical_offset = 0.0          # Absolute value of the vertical offset between optical and visual axis in radians when the eye
                                            # is in the primary position. Positive means optical axis is inferior to the visual axis
        self.d_iris = 0.0                   # Distance from the center of the eye to the iris in mm
        self.h_cornea = 0.0                 # Distance between cornea and sclera centers
        self.r_iris = 0.0                   # Radius of the iris in mm
        self.r_pupil = 0.0                  # Radius of the pupil in mm
        self.location = [0.0, 0.0, 0.0]     # Location of the eye in head coordinates in mm 
        self.visual_axis = [0.0, 0.0, 1.0]  # Visual axis of the eye in head coordinates
        self.optical_axis = [0.0, 0.0, 1.0] # Optical axis of the eye in head coordinates
        self.LP_phi = 0.0                   # Listing's plane normal vector phi angle in radians
        self.LP_theta = 0.0                 # Listing's plane normal vector theta angle in radians
        self.VP_vergence_var = 0.5          # Velocity plane vergence variation factor
        self.last_rotation_vector = None    # Last rotation vector of the eye in head coordinates (magnitude in rad/s)
        self.n_cornea = 1.3375              # Refractive index of the cornea
        return

    def update_d_iris(self) -> None:
        '''
        This function updates the distance between the center of the sclera and the center of the iris, calculated based on the radii
        of the sclera and the iris

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.d_iris = np.sqrt(self.r_sclera**2 - self.r_iris**2)
        return

    def update_h_cornea(self) -> None:
        '''
        This function updates the height of the cornea, calculated based on the radii of the cornea and the iris, and the distance between
        the center of the sclera and the iris

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.update_d_iris()
        self.h_cornea = self.d_iris - np.sqrt(self.r_cornea**2 - self.r_iris**2)
        return

    def eye_to_primary_pos(self) -> None:
        '''
        This function updates the optical axis and visual axis of the eye based on the eye parameters, placing the eye in the primary
        position

        Parameters
        ----------
        None

        # Returns
        -------
        None
        '''
        # We calculate the visual axis
        self.visual_axis = [np.sin(self.LP_phi)*np.cos(self.LP_theta), np.sin(self.LP_phi)*np.sin(self.LP_theta), np.cos(self.LP_phi)]
        
        # We calculate the projection of the visual axis onto the horizontal plane and the rotation matrix that aligns the Z axis with this projection
        n_proj = np.array([self.visual_axis[0], 0, self.visual_axis[2]]) / np.linalg.norm(np.array([self.visual_axis[0], 0, self.visual_axis[2]]))
        rot_n = rot_mat_align_vectors([0, 0, 1], n_proj)

        # We calculate the optical axis
        if self.lor == 'left':
            rot_mat1 = R.from_rotvec([0, -self.horizontal_offset, 0])
            rot_mat2 = R.from_rotvec(R.from_matrix(rot_n).apply([-self.vertical_offset * np.cos(self.horizontal_offset), 0, -self.vertical_offset * np.sin(self.horizontal_offset)]))
        else:
            rot_mat1 = R.from_rotvec([0, self.horizontal_offset, 0])
            rot_mat2 = R.from_rotvec(R.from_matrix(rot_n).apply([-self.vertical_offset * np.cos(self.horizontal_offset), 0, self.vertical_offset * np.sin(self.horizontal_offset)]))
        optaxis = rot_mat2.apply(rot_mat1.apply(self.visual_axis))
        self.optical_axis = optaxis / np.linalg.norm(optaxis)
        return
    
    def get_visaxis_optaxis_primary_position(self) -> np.ndarray:
        '''
        This function calculates the optical axis of the eye when it is in the primary position, based on the eye parameters

        Parameters
        ----------
        None

        Returns
        -------
        visaxis_primary : np.ndarray
            Visual axis of the eye in the primary position
        optaxis_primary : np.ndarray
            Optical axis of the eye in the primary position
        '''
        # We calculate the visual axis in the primary position
        visaxis_primary = [np.sin(self.LP_phi)*np.cos(self.LP_theta), np.sin(self.LP_phi)*np.sin(self.LP_theta), np.cos(self.LP_phi)]
        
        # We calculate the projection of the visual axis onto the horizontal plane and the rotation matrix that aligns the Z axis with this projection
        n_proj = np.array([visaxis_primary[0], 0, visaxis_primary[2]]) / np.linalg.norm(np.array([visaxis_primary[0], 0, visaxis_primary[2]]))
        rot_n = rot_mat_align_vectors([0, 0, 1], n_proj)

        # We calculate the optical axis in the primary position
        if self.lor == 'left':
            rot_mat1 = R.from_rotvec([0, -self.horizontal_offset, 0])
            rot_mat2 = R.from_rotvec(R.from_matrix(rot_n).apply([-self.vertical_offset * np.cos(self.horizontal_offset), 0, -self.vertical_offset * np.sin(self.horizontal_offset)]))
        else:
            rot_mat1 = R.from_rotvec([0, self.horizontal_offset, 0])
            rot_mat2 = R.from_rotvec(R.from_matrix(rot_n).apply([-self.vertical_offset * np.cos(self.horizontal_offset), 0, self.vertical_offset * np.sin(self.horizontal_offset)]))
        optaxis_primary = rot_mat2.apply(rot_mat1.apply(visaxis_primary))
        optaxis_primary = optaxis_primary / np.linalg.norm(optaxis_primary)

        return visaxis_primary, optaxis_primary

    def show(self, pl_plotter: pv.Plotter, plot_va: bool = False) -> None:
        '''
        This function shows the eye in a pyvista plotter

        Parameters
        ----------
        pl_plotter : pyvista.Plotter
            Pyvista plotter where the eye will be shown
        plot_va : bool, optional
            Indicates whether the visual axis will be plotted. Default is False
        
        Returns
        -------
        None
        '''
        # We calculate the visual and optical axis in the primary position
        visaxis_primary, optaxis_primary = self.get_visaxis_optaxis_primary_position()

        # We calculate the rotation matrix to represent the rotation from the principal position to the current visual axis
        rot_visaxis = rot_mat_align_vectors(visaxis_primary, self.visual_axis)

        # We calculate the rotation matrix to represent the rotation from the [0, 0, 1] vector to the optical axis in the primary position
        rot_optaxis = rot_mat_align_vectors([0, 0, 1], optaxis_primary)

        # The eyeball rotation will be the composition of these two rotations
        try:
            my_rot = R.from_matrix(np.dot(rot_visaxis, rot_optaxis))
            valid_rotation = True
        except Exception:
            # We simulate a zero rotation
            my_rot = R.from_matrix(np.eye(3))
            valid_rotation = False

        # First, we show the sclera surface
        sclera_color = 'white'
        phi_ini_rad = np.arctan(self.r_iris/self.d_iris)
        phi_ini_deg = 180*phi_ini_rad/np.pi
        phi_end_deg = 115
        sclera_front_semi_sphere = pv.Sphere(radius=self.r_sclera, center=self.location,
                                  theta_resolution=180, phi_resolution=180,
                                  start_theta=270.00001, end_theta=270, start_phi=phi_ini_deg, end_phi=phi_end_deg)
        # Smooth vertex normals for gentler shading on the sclera
        try:
            sclera_front_semi_sphere.compute_normals(point_normals=True, cell_normals=False,
                                                     auto_orient_normals=True, inplace=True)
        except Exception:
            pass
       
        # Initialize the texture coordinates array
        sclera_texture = pv.read_texture(str(RESOURCES_DIR / 'sclera_texture_v2.png'))
        sclera_front_semi_sphere.active_texture_coordinates = np.zeros((sclera_front_semi_sphere.points.shape[0], 2))
        # Populate by manually calculating
        for i in range(sclera_front_semi_sphere.points.shape[0]):
            theta = np.arctan2(sclera_front_semi_sphere.points[i, 1] - self.location[1], sclera_front_semi_sphere.points[i, 0] - self.location[0])
            phi = np.arccos((sclera_front_semi_sphere.points[i, 2] - self.location[2])/self.r_sclera)
            len_norm = 0.75*phi/(np.pi/2)
            if len_norm > 1:
                len_norm = 1
            u = 0.5 + 0.5*len_norm*np.cos(theta)
            v = 0.5 + 0.5*len_norm*np.sin(theta)
            sclera_front_semi_sphere.active_texture_coordinates[i] = [u, v]

        # We rotate the sclera to match the visual axis
        if valid_rotation:
            sclera_front_semi_sphere.rotate(my_rot, point=self.location, inplace=True)

        pl_plotter.add_mesh(sclera_front_semi_sphere,
                    color=sclera_color, show_edges=False, opacity=1, texture=sclera_texture,
                    specular=0.2, specular_power=20, smooth_shading=True)

        sclera_back_semi_sphere = pv.Sphere(radius=self.r_sclera, center=self.location, direction=self.optical_axis,
                            theta_resolution=180, phi_resolution=180,
                            start_theta=270.00001, end_theta=270,
                            start_phi=phi_end_deg, end_phi=180)

        try:
            sclera_back_semi_sphere.compute_normals(point_normals=True, cell_normals=False,
                                                    auto_orient_normals=True, inplace=True)
        except Exception:
            pass

        pl_plotter.add_mesh(sclera_back_semi_sphere, color=(210, 37, 34), show_edges=False, opacity=1,
                    specular=0.2, specular_power=15, smooth_shading=True)
        
        # Now, we show the iris surface
        inc = 0
        # The representation has some problem when r_pupil is exactly 0. We change it exclusively for the visualization
        if self.r_pupil == 0:
            inc = 0.01
        iris_texture = pv.read_texture(str(RESOURCES_DIR / 'iris_texture.png'))
        iris_disc = pv.Disc(center=self.location + np.array([0, 0, 1])*self.d_iris, 
                    inner = self.r_pupil + inc, outer=self.r_iris,
                    normal=[0, 0, 1], c_res=120)

        # We initialize the texture coordinates array
        iris_disc.active_texture_coordinates = np.zeros((iris_disc.points.shape[0], 2))
        try:
            iris_disc.compute_normals(point_normals=True, cell_normals=False,
                                      auto_orient_normals=True, inplace=True)
        except Exception:
            pass
        # Populate by manually calculating
        r_norm_ext = 0.48
        r_norm_int = 0.16
        for i in range(iris_disc.points.shape[0]):
            theta = np.arctan2(iris_disc.points[i, 1] - self.location[1], iris_disc.points[i, 0] - self.location[0])
            rho = np.sqrt((iris_disc.points[i, 0] - self.location[0])**2 + (iris_disc.points[i, 1] - self.location[1])**2)
            m = (r_norm_ext - r_norm_int)/(self.r_iris - self.r_pupil + inc)
            # We normalize the distance to the iris radius
            rho_img = r_norm_int + m*(rho - self.r_pupil + inc)
            u = 0.5 + rho_img*np.cos(theta)
            v = 0.5 + rho_img*np.sin(theta)
            iris_disc.active_texture_coordinates[i] = [u, v]
        
        if valid_rotation:
            iris_disc.rotate(my_rot, point=self.location, inplace=True)
        pl_plotter.add_mesh(iris_disc,
                color='white', show_edges=False, opacity=1, texture=iris_texture,
                specular=0.15, specular_power=10, smooth_shading=True)
        
        # Now, we show the corneal surface
        phi_end = 180*np.arctan(self.r_iris/(self.d_iris - self.h_cornea))/np.pi
        cornea_sphere = pv.Sphere(radius=self.r_cornea, center=self.location + self.h_cornea*np.array(self.optical_axis), 
                                  direction=self.optical_axis,
                                  theta_resolution=180, phi_resolution=180,
                                  start_theta=0, end_theta=360, start_phi=0, end_phi=phi_end)
        try:
            cornea_sphere.compute_normals(point_normals=True, cell_normals=False,
                                          auto_orient_normals=True, inplace=True)
        except Exception:
            pass
        pl_plotter.add_mesh(cornea_sphere,
                          color=(1, 1, 1), show_edges=False, opacity=0.2,
                          specular=0.6, specular_power=40, smooth_shading=True)       

        # Plot the arrow of the optical axis
        optical_axis_arrow = pv.Arrow(start=np.array(self.location), direction=30*np.array(self.optical_axis),
                                      tip_length=0.05,      # Controls the length of the tip
                                      tip_radius=0.01,      # Controls the thickness of the tip
                                      shaft_radius=0.005,   # Controls the thickness of the shaft
                                      shaft_resolution=60,  # Defines the resolution
                                      scale='auto')      
        pl_plotter.add_mesh(optical_axis_arrow, color='b', opacity=1, smooth_shading=True)

        if plot_va:
            # Plot the arrow of the visual axis
            visual_axis_arrow = pv.Arrow(start=np.array(self.location), direction=30*np.array(self.visual_axis),
                                        tip_length=0.05,      # Controls the length of the tip
                                        tip_radius=0.01,      # Controls the thickness of the tip
                                        shaft_radius=0.005,   # Controls the thickness of the shaft
                                        shaft_resolution=60,  # Defines the resolution
                                        scale='auto')
            pl_plotter.add_mesh(visual_axis_arrow, color='r', opacity=1, smooth_shading=True)
        return

    def to_dict(self) -> dict[str, any]:
        # Convert numpy arrays or Point objects to lists
        location_list = list(self.location) if hasattr(self.location, '__iter__') else self.location
        visual_axis_list = list(self.visual_axis) if hasattr(self.visual_axis, '__iter__') else self.visual_axis
        optical_axis_list = list(self.optical_axis) if hasattr(self.optical_axis, '__iter__') else self.optical_axis
        
        return {'lor':self.lor,
                'r_sclera':self.r_sclera,
                'r_retina':self.r_retina,
                'r_cornea':self.r_cornea,
                'horizontal_offset':self.horizontal_offset,
                'vertical_offset':self.vertical_offset,
                'd_iris':self.d_iris,
                'h_cornea':self.h_cornea,
                'r_iris':self.r_iris,
                'r_pupil':self.r_pupil,
                'location':location_list,
                'visual_axis':visual_axis_list,
                'optical_axis':optical_axis_list,
                'n_cornea': self.n_cornea}
    
    def from_dict(self, eye_dict: dict[str, any]) -> None:
        if 'lor' in eye_dict: self.lor = eye_dict['lor']
        if 'r_sclera' in eye_dict: self.r_sclera = eye_dict['r_sclera']
        if 'r_retina' in eye_dict: self.r_retina = eye_dict['r_retina']
        if 'r_cornea' in eye_dict: self.r_cornea = eye_dict['r_cornea']
        if 'horizontal_offset' in eye_dict: self.horizontal_offset = eye_dict['horizontal_offset']
        if 'vertical_offset' in eye_dict: self.vertical_offset = eye_dict['vertical_offset']
        if 'd_iris' in eye_dict: self.d_iris = eye_dict['d_iris']
        if 'h_cornea' in eye_dict: self.h_cornea = eye_dict['h_cornea']
        if 'r_iris' in eye_dict: self.r_iris = eye_dict['r_iris']
        if 'r_pupil' in eye_dict: self.r_pupil = eye_dict['r_pupil']
        if 'location' in eye_dict: self.location = eye_dict['location']
        if 'n_cornea' in eye_dict: self.n_cornea = eye_dict['n_cornea']
        self.update_d_iris()
        self.update_h_cornea()
        if 'visual_axis' in eye_dict and 'optical_axis' in eye_dict:
            self.visual_axis = eye_dict['visual_axis']
            self.optical_axis = eye_dict['optical_axis']
        else:
            self.eye_to_primary_pos()
        return

    def look_at_this_point(self, point3D: list[float], vergence: float = None) -> None:
        '''
        This function updates the visual axis and the optical axis of the eye to point towards the indicated 3D point

        Parameters
        ----------
        point3D : list[float]
            3D point the eye should look at, in the eye's reference frame
        vergence : float, optional
            Vergence angle in radians. If provided, the velocity plane will be calculated and the visual and optical axes will be updated
            following Listing's law. Default is None

        Returns
        -------
        None
        '''
        self.last_rotation_vector = None # Reset the last rotation vector
        if vergence is None:
            # Direct rotation is assumed, no taking into account Listing's law
            # We calculate the rotation that must be applied to the visual axis
            R1 = rot_mat_align_vectors(self.visual_axis, np.array(point3D) - np.array(self.location))
            # We calculate the rotation matrix
            rot_mat = R.from_matrix(R1)

            # We apply the rotation to the visual and optical axes
            self.visual_axis = rot_mat.apply(self.visual_axis)
            self.optical_axis = rot_mat.apply(self.optical_axis)
        else:
            # We take into account Listing's law
            # We calculate the velocity plane position according to the vergence
            LP = [np.sin(self.LP_phi)*np.cos(self.LP_theta), np.sin(self.LP_phi)*np.sin(self.LP_theta), np.cos(self.LP_phi)]
            if self.lor == 'left':
                rot_VP = np.array([0, -vergence * self.VP_vergence_var, 0])
            else:
                rot_VP = np.array([0, vergence * self.VP_vergence_var, 0])
            VP = R.from_rotvec(rot_VP).apply(LP)

            # We calculate the rotations to the primary position and to the final position
            R1 = rot_mat_align_vectors(self.visual_axis, VP)
            R2 = rot_mat_align_vectors(VP, np.array(point3D) - np.array(self.location))
            # We calculate the total rotation
            rot_mat = R.from_matrix(np.matmul(R2, R1))

            # We apply the rotation to the visual and optical axes
            self.visual_axis = rot_mat.apply(self.visual_axis)
            self.visual_axis = self.visual_axis / np.linalg.norm(self.visual_axis)
            self.optical_axis = rot_mat.apply(self.optical_axis)
            self.optical_axis = self.optical_axis / np.linalg.norm(self.optical_axis)
        return

    def look_in_this_direction(self, target_axis: list[float], vergence: float = None) -> None:
        '''
        This function updates the visual axis and the optical axis of the eye to point in the indicated direction

        Parameters
        ----------
        target_axis : list[float]
            Direction the eye should look at, in the eye's reference frame
        vergence : float, optional
            Vergence angle in radians. If provided, the velocity plane will be calculated and the visual and optical axes will be updated
            following Listing's law. Default is None

        Returns
        -------
        None
        '''
        self.last_rotation_vector = None # Reset the last rotation vector
        if vergence is None:
            # Direct rotation is assumed, no taking into account Listing's law
            # We calculate the rotation that must be applied to the visual axis
            R1 = rot_mat_align_vectors(self.visual_axis, target_axis)
            # We calculate the rotation matrix
            rot_mat = R.from_matrix(R1)
            # We apply the rotation to the visual and optical axes
            self.visual_axis = rot_mat.apply(self.visual_axis)
            self.optical_axis = rot_mat.apply(self.optical_axis)
        else:
            # We take into account Listing's law
            # We calculate the velocity plane position according to the vergence
            LP = [np.sin(self.LP_phi)*np.cos(self.LP_theta), np.sin(self.LP_phi)*np.sin(self.LP_theta), np.cos(self.LP_phi)]
            if self.lor == 'left':
                rot_VP = np.array([0, -vergence * self.VP_vergence_var, 0])
            else:
                rot_VP = np.array([0, vergence * self.VP_vergence_var, 0])
            VP = R.from_rotvec(rot_VP).apply(LP)

            # We calculate the rotations to the primary position and to the final position
            R1 = rot_mat_align_vectors(self.visual_axis, VP)
            R2 = rot_mat_align_vectors(VP, target_axis)
            # We calculate the total rotation
            rot_mat = R.from_matrix(np.dot(R2, R1))

            # We apply the rotation to the visual and optical axes
            self.visual_axis = rot_mat.apply(self.visual_axis)
            self.optical_axis = rot_mat.apply(self.optical_axis)
        return
    
    def rotate_eyeball(self, rot_vector: list[float]) -> None:
        '''
        This function rotates the eye around its center according to the indicated rotation vector

        Parameters
        ----------
        rot_vector : list[float]
            Rotation vector in radians, in the eye's reference frame

        Returns
        -------
        None
        '''
        # We create the rotation matrix
        rot_mat = R.from_rotvec(np.array(rot_vector))
        # We apply the rotation to the visual and optical axes
        self.visual_axis = rot_mat.apply(self.visual_axis)
        self.optical_axis = rot_mat.apply(self.optical_axis)
        self.last_rotation_vector = np.array(rot_vector) # We save the last rotation vector
        return

class User:

    def __init__(self):
        self.eyes = {'left':Eye('left'),'right':Eye('right')}   # Dictionary containing the left and right eyes
        self.vergence = 0                                       # Vergence angle in radians. This is the angle between the two visual axes
                                                                # of the eyes
        return
    
    def update_vergence(self) -> None:
        '''
        This function updates the user's vergence angle, which is the angle between the visual axes of both eyes

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.vergence = np.arccos(np.dot(np.array(self.eyes['left'].visual_axis), np.array(self.eyes['right'].visual_axis)))
        return
    
    def to_dict(self) -> dict[str, any]:
        return {'eyes':{key:value.to_dict() for key,value in self.eyes.items()}}
    
    def from_dict(self, user_dict: dict[str, any]) -> None:
        self.eyes['left'].from_dict(user_dict['eyes']['left'])
        self.eyes['right'].from_dict(user_dict['eyes']['right'])
        self.update_vergence()
        return
    
class Sensor:

    def __init__(self):
        self.ID = ''                                                        # Sensor ID number
        self.origin = [0.0, 0.0, 0.0]                                       # Sensor origin in the glasses' reference frame
        self.direction = [0.0, 0.0, 0.0]                                    # Sensor direction in the glasses' reference frame
        self.lor = ''                                                       # Sensor laterality ('left' or 'right')
        self.scaling = 1.0                                                  # Scaling factor for the sensor's distance measurements
        self.distance_std_noise = 0.1                                       # Standard deviation of the noise in the distance measurements
        self.velocity_std_noise = 3 * (np.pi/180) * 12 * np.cos(np.pi/4)    # Standard deviation of the noise in the velocity measurements

        # Now, we create the interpolator error functions
        # Samples from the real noise characterization of the sensors
        distance_values = np.linspace(20, 30, 11)
        velocity_values = np.linspace(0, 500, 11)
        std_values = np.array([
            # [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # 0
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], # 50 
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], # 100
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], # 150
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], # 200
            [0.7, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], # 250
            [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.6], # 300
            [2.3, 2.0, 1.8, 1.7, 1.6, 1.5, 1.3, 1.2, 1.1, 1.0, 0.7], # 350
            [2.7, 2.2, 2.0, 1.7, 1.6, 1.5, 1.4, 1.4, 1.3, 1.3, 1.1], # 400
            [3.5, 2.75, 2.5, 2.3, 2.1, 1.8, 1.6, 1.4, 1.4, 1.3, 1.2], # 450
            [4.5, 3.5, 2.75, 2.4, 2.2, 1.8, 1.7, 1.5, 1.4, 1.3, 1.2]  # 500
            ])
        std_values = std_values.T
        # Create the interpolator
        self.std_noise_distance_interpolator = RegularGridInterpolator(     # Interpolator for the distance noise
                            (distance_values, velocity_values), 
                            std_values,
                            bounds_error=False,
                            fill_value=0.1)
        
        # Now for noise in the velocity measurement
        std_values = np.array([
            # [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.4, 0.4], # 0
            [0.6, 1.2, 0.4, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2], # 50 
            [1.3, 1.3, 1.3, 1.6, 1.4, 1.4, 1.3, 1.6, 1.4, 1.3, 1.3], # 100
            [3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6], # 150
            [2.0, 1.5, 1.5, 1.4, 2.0, 1.6, 1.7, 1.4, 2.0, 1.4, 1.4], # 200
            [2.0, 2.0, 2.0, 1.8, 1.3, 1.2, 1.6, 1.3, 2.0, 1.3, 1.2], # 250
            [2.6, 2.2, 2.1, 2.1, 2.1, 2.1, 2.0, 1.6, 1.9, 2.0, 1.8], # 300
            [3.6, 3.3, 3.1, 2.8, 2.8, 2.8, 2.0, 2.8, 2.8, 2.8, 2.2], # 350
            [5.2, 4.4, 4.4, 3.2, 3.6, 2.8, 2.8, 2.8, 2.8, 2.8, 2.0], # 400
            [11.2, 8.6, 7.0, 4.8, 4.0, 3.4, 3.2, 3.1, 2.8, 2.4, 2.2], # 450
            [15.6, 14.6, 13.8, 10.8, 8.0, 5.6, 4.8, 3.8, 3.6, 3.6, 2.8]  # 500
            ])
        std_values = std_values.T
        # Create the interpolator
        self.std_noise_velocity_interpolator = RegularGridInterpolator(     # Interpolator for the velocity noise
                            (distance_values, velocity_values), 
                            std_values,
                            bounds_error=False,
                            fill_value=0.1)
        
        return
    
    def set_location(self, origin: list[float], direction: list[float]) -> None: # Respect glasses coordinate system in mm and degrees
        '''
        This function sets the location of the sensor in the glasses coordinate system

        Parameters
        ----------
        origin : list[float]
            Origin of the sensor in the glasses reference frame [mm]
        direction : list[float]
            Direction of the sensor in the glasses reference frame

        Returns
        -------
        None
        '''
        self.origin = origin 
        self.direction = direction
        return
    
    def to_dict(self) -> dict[str, any]:
        # Convert numpy arrays or Point objects to lists
        origin_list = list(self.origin) if hasattr(self.origin, '__iter__') else self.origin
        direction_list = list(self.direction) if hasattr(self.direction, '__iter__') else self.direction
        
        return {'ID':self.ID,
                'origin':origin_list,
                'direction':direction_list,
                'lor':self.lor,
                'distance_std_noise':self.distance_std_noise,
                'velocity_std_noise':self.velocity_std_noise}
    
    def from_dict(self, sensor_dict: dict[str, any]) -> None:
        if 'ID' in sensor_dict: self.ID = sensor_dict['ID']
        if 'origin' in sensor_dict: self.origin = sensor_dict['origin']
        if 'direction' in sensor_dict: self.direction = sensor_dict['direction']
        if 'lor' in sensor_dict: self.lor = sensor_dict['lor']
        if 'scaling' in sensor_dict: self.scaling = sensor_dict['scaling']
        if 'distance_std_noise' in sensor_dict: self.distance_std_noise = sensor_dict['distance_std_noise']
        if 'velocity_std_noise' in sensor_dict: self.velocity_std_noise = sensor_dict['velocity_std_noise']
        return
    
    def get_distance_noise(self, distance_query: float = 0, velocity_query: float = 0) -> float:
        '''
        This function returns the noise in the sensor's distance measurement, in mm

        Parameters
        ----------
        distance_query : float, optional
            Distance that models the noise [mm]. Default is 0
        velocity_query : float, optional
            Velocity that models the noise [mm/s]. Default is 0

        Returns
        -------
        float
            Noise in the sensor's distance measurement [mm]
        '''
        # We obtain the distance noise from the last measurement
        std_interpolated = self.std_noise_distance_interpolator([[distance_query, velocity_query]])[0]

        return np.random.normal(0,std_interpolated)
    
    def get_velocity_noise(self, distance_query: float = 0, velocity_query: float = 0) -> float:
        '''
        This function returns the noise in the sensor's velocity measurement, in mm/s

        Parameters
        ----------
        distance_query : float, optional
            Distance that models the noise [mm]. Default is 0
        velocity_query : float, optional
            Velocity that models the noise [mm/s]. Default is 0

        Returns
        -------
        float
            Noise in the sensor's velocity measurement [mm/s]
        '''
        # We obtain the velocity noise from the last measurement
        std_interpolated = self.std_noise_velocity_interpolator([[distance_query, velocity_query]])[0]
        # We make the conversion from º/s to mm/s
        std_interpolated = std_interpolated * (np.pi/180) * 12 * np.cos(np.pi/4)

        return np.random.normal(0,std_interpolated)

class Glasses:

    def __init__(self, sensors_list: list[Sensor] = None):
        self.sensors_list = []
        self.noise_sensor_model = 'global_constant'                             # Noise model for the sensors. Options: 'no_noise',
                                                                                # 'global_constant', 'ind_constant' or 'ind_variable'
        self.global_distance_std_noise = 0.1                                    # Standard deviation of the noise in the distance measurements
        self.global_velocity_std_noise = 3 * (np.pi/180) * 12 * np.cos(np.pi/4) # Standard deviation of the noise in the velocity measurements
        self.location_R = [[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]     # Rotation matrix of the glasses respect to the HCS
        self.location_T = [0.0, 0.0, 0.0]                                       # Translation vector of the glasses respect to the HCS
        self.freq = 1000                                                        # Frequency of the sensors in Hz
        # If sensors_list is not empty, add the sensors in the list
        if sensors_list is not None:
            for sensor_i in sensors_list:
                self.add_sensor(sensor_i)
        return
    
    def to_dict(self) -> dict[str, any]:
        # Convert numpy arrays to lists if needed
        location_R_list = [list(row) if hasattr(row, '__iter__') else row for row in self.location_R] if hasattr(self.location_R, '__iter__') else self.location_R
        location_T_list = list(self.location_T) if hasattr(self.location_T, '__iter__') else self.location_T
        
        return {'sensors_list':[sensor.to_dict() for sensor in self.sensors_list],
                'noise_sensor_model':self.noise_sensor_model,
                'global_distance_std_noise':self.global_distance_std_noise,
                'global_velocity_std_noise':self.global_velocity_std_noise,
                'location_R':location_R_list,
                'location_T':location_T_list,
                'freq':self.freq}
    
    def from_dict(self, glasses_dict: dict[str, any]) -> None:
        if 'sensors_list' in glasses_dict:
            self.sensors_list = [Sensor() for sensor in glasses_dict['sensors_list']]
            for i,sensor_dict in enumerate(glasses_dict['sensors_list']):
                self.sensors_list[i].from_dict(sensor_dict)
        if 'noise_sensor_model' in glasses_dict: self.noise_sensor_model = glasses_dict['noise_sensor_model']
        if 'global_distance_std_noise' in glasses_dict: self.global_distance_std_noise = glasses_dict['global_distance_std_noise']
        if 'global_velocity_std_noise' in glasses_dict: self.global_velocity_std_noise = glasses_dict['global_velocity_std_noise']
        if 'location_R' in glasses_dict: self.location_R = glasses_dict['location_R']
        if 'location_T' in glasses_dict: self.location_T = glasses_dict['location_T']
        if 'freq' in glasses_dict: self.freq = glasses_dict['freq']
        return
    
    def add_sensor(self, sensor: Sensor) -> None:
        '''
        This function adds a sensor to the glasses' sensor list

        Parameters
        ----------
        sensor : Sensor
            Sensor to be added to the glasses' sensor list

        Returns
        -------
        None
        '''    
        self.sensors_list.append(sensor)
        return

class Measurement:

    def __init__(self):
        self.ID_sensor = ''     # Sensor ID
        self.lor = ''           # 'left' or 'right'
        self.surface = ''       # 'sclera', 'iris', 'retina' or 'none'
        self.distance = 0.0     # Measured distance in mm
        self.velocity = 0.0     # Measured velocity in mm/s
        self.path_points = []   # Points of the path in HCS
        return
    
    def to_dict(self) -> dict[str, any]:
        return {'ID_sensor': self.ID_sensor,
                'lor': self.lor,
                'surface': self.surface,
                'distance': self.distance,
                'velocity': self.velocity,
                'path_points': [list(p) if hasattr(p, '__iter__') else p for p in self.path_points]}
    
    def from_dict(self, meas_dict: dict[str, any]) -> None:
        if 'ID_sensor' in meas_dict: self.ID_sensor = meas_dict['ID_sensor']
        if 'lor' in meas_dict: self.lor = meas_dict['lor']
        if 'surface' in meas_dict: self.surface = meas_dict['surface']
        if 'distance' in meas_dict: self.distance = meas_dict['distance']
        if 'velocity' in meas_dict: self.velocity = meas_dict['velocity']
        if 'path_points' in meas_dict: self.path_points = [np.array(p) for p in meas_dict['path_points']]
        return

class Gaze_trajectory: # Parameters obtained in previous timestamps

    def __init__(self):
        self.optical_axis_list = {'left': [], 'right': []}  # Dictionary containing the list of optical axes for each eye
        self.visual_axis_list = {'left': [], 'right': []}   # Dictionary containing the list of visual axes for each eye
        return
    
    def to_dict(self) -> dict[str, any]:
        return {'optical_axis_list':{key:[i.tolist() for i in value] for key,value in self.optical_axis_list.items()},
                'visual_axis_list':{key:[i.tolist() for i in value] for key,value in self.visual_axis_list.items()}}

    def from_dict(self, gaze_traj_dict: dict[str, any]) -> None:
        if 'optical_axis_list' in gaze_traj_dict:
            self.optical_axis_list = {key:[np.array(i) for i in value] for key,value in gaze_traj_dict['optical_axis_list'].items()}
        if 'visual_axis_list' in gaze_traj_dict:
            self.visual_axis_list = {key:[np.array(i) for i in value] for key,value in gaze_traj_dict['visual_axis_list'].items()}
        return

    def add_optical_axis_sample(self, lor: str, optical_axis: list[float]) -> None:
        '''
        This function adds the optical axis at a given time to the list of optical axes of the corresponding eye

        Parameters
        ----------
        lor : str
            Side of the eye ('left' or 'right')
        optical_axis : list[float]
            Optical axis of the eye

        Returns
        -------
        None
        '''
        self.optical_axis_list[lor].append(np.array(optical_axis))
        return
    
    def add_visual_axis_sample(self, lor: str, visual_axis: list[float]) -> None:
        '''
        This function adds the visual axis at a given time to the list of visual axes of the corresponding eye

        Parameters
        ----------
        lor : str
            Side of the eye ('left' or 'right')
        visual_axis : list[float]
            Visual axis of the eye

        Returns
        -------
        None
        '''
        self.visual_axis_list[lor].append(np.array(visual_axis))
        return

class System_simulator:

    def __init__(self):
        self.glasses = Glasses()                                # Glasses object
        self.user = User()                                      # User object
        self.last_measurement = dict[str, Measurement]()        # Dict with the last measurement from each sensor
        self.meas_case = {'left_case': 0, 'right_case': 0}      # Dict with the current surface hit case for left and right eye
        self.is_noisy = True                                    # Boolean indicating if the simulation is noisy or not
        self.gaze_trajectory = Gaze_trajectory()                # Gaze trajectory object
        return

    def to_dict(self) -> dict[str, any]:
        return {'glasses':self.glasses.to_dict(),
                'user':self.user.to_dict(),
                'last_measurement':{key: value.to_dict() for key, value in self.last_measurement.items()},
                'meas_case': self.meas_case,
                'is_noisy':self.is_noisy,
                'gaze_trajectory':self.gaze_trajectory.to_dict()}
    
    def from_dict(self, system_dict: dict[str, any]) -> None:
        self.glasses.from_dict(system_dict['glasses'])
        self.user.from_dict(system_dict['user'])
        if 'is_noisy' in system_dict: self.is_noisy = system_dict['is_noisy']
        if 'last_measurement' in system_dict:
            self.last_measurement.clear()
            for key, value in system_dict['last_measurement'].items():
                meas = Measurement()
                meas.from_dict(value)
                self.last_measurement[key] = meas
        if 'gaze_trajectory' in system_dict:
            self.gaze_trajectory.from_dict(system_dict['gaze_trajectory'])
        return

    def set_configuration(self, config_file: str | None) -> None:
        if config_file is not None:
            with open(config_file, 'r') as f:
                system_dict = json.load(f)
            self.from_dict(system_dict)
            # Assure the coherence of the configuration of iris parameters
            self.user.eyes['left'].update_d_iris() 
            self.user.eyes['left'].update_h_cornea()
            self.user.eyes['right'].update_d_iris()
            self.user.eyes['right'].update_h_cornea()
        else:
            raise ValueError('No configuration file provided')
        return

    def transform_sensor_to_HCS(self, sensor: Sensor) -> tuple[np.ndarray, np.ndarray]:
        '''
        This function transforms the sensor coordinates from the glasses' reference frame to the head coordinate system (HCS)

        Parameters
        ----------
        sensor : Sensor
            Sensor to be transformed

        Returns
        -------
        new_origin : numpy.ndarray
            Origin of the sensor in the head coordinate system (HCS) [mm]
        new_direction : numpy.ndarray
            Direction of the sensor in the head coordinate system (HCS)
        '''
        # First, we transform the sensors to the head coordinate system
        new_origin = np.dot(np.array(self.glasses.location_R), np.array(sensor.origin)) + \
                    np.array(self.glasses.location_T)
        new_direction = np.dot(np.array(self.glasses.location_R), np.array(sensor.direction))
        # Normalize the direction vector
        new_direction = new_direction / np.linalg.norm(new_direction)
        return new_origin, new_direction
    
    def get_meas_case(self, measurements_dict: dict[str, Measurement] = None) -> tuple[int, int]:
        '''
        This function obtains the number of hits on each type of surface for both eyes

        Parameters
        ----------
        measurements_dict : dict[str, Measurement], optional
            Dictionary with the measurements. If None, the function will use the last measurements stored in the system. Default is None

        Returns
        -------
        left_case : int
            Number of hits per surface in the left eye
        right_case : int
            Number of hits per surface in the right eye
        '''
        # We load the measurements dictionary. If it is None, we use the last measurements stored in the system
        if measurements_dict is None:
            measurements_dict = self.last_measurement

        # We count the number of hits on each surface for both eyes
        counting_left = {'retina':0, 'iris':0, 'sclera':0}
        counting_right = {'retina':0, 'iris':0, 'sclera':0}
        for meas_i in measurements_dict.values():
            if meas_i.lor == 'left': 
                counting_left[meas_i.surface] += 1
            else:
                counting_right[meas_i.surface] += 1
        left_case = 100 * counting_left['retina'] + 10 * counting_left['iris'] + counting_left['sclera']
        right_case = 100 * counting_right['retina'] + 10 * counting_right['iris'] + counting_right['sclera']
        return left_case, right_case
    
    def simulate_velocity_measurement(self, sensor_dir: np.ndarray, impact_p: np.ndarray, lor: str) -> tuple[float, float]:
        '''
        Important: only makes sense when the impact point is on the sclera
        -------------------------------------------------------------

        This function simulates the velocity detected by a sensor based on the change in gaze direction or last known rotation vector

        Parameters
        ----------
        sensor_dir : numpy.ndarray
            Direction of the sensor in the head coordinate system (HCS)
        impact_p : numpy.ndarray
            Impact point in the head coordinate system (HCS)
        lor : str
            Indicates whether the studied eye is the left or right eye ('left' or 'right')

        Returns
        -------
        sensor_v : float
            Velocity detected by the sensor [mm/s]
        angular_vel : float
            Angular velocity of the eye [º/s]
        '''
        
        # We obtain some parameters needed for the calculations
        impact_p_corr = impact_p - np.array(self.user.eyes[lor].location)
        # If a known rotation has occurred, we obtain it directly:
        if (self.user.eyes[lor].last_rotation_vector is not None):
            angular_vel_vec = self.user.eyes[lor].last_rotation_vector
        else:
            # We calculate the rotation based on the change in visual axis
            visaxis_prev = self.gaze_trajectory.visual_axis_list[lor][-2]
            visaxis_new = self.gaze_trajectory.visual_axis_list[lor][-1]

            # We calculate the rotation axis
            rotation_axis = np.cross(visaxis_prev, visaxis_new)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            # We calculate the rotation angle
            excursion_angle = np.arccos(np.dot(visaxis_prev, visaxis_new))
            # We calculate the angular velocity
            angular_vel = excursion_angle * self.glasses.freq  # Obtained in rad/s
            # We calculate the angular velocity vector
            angular_vel_vec = angular_vel * rotation_axis  # Obtained in rad/s
        # # We calculate the linear velocity
        lineal_v = np.cross(angular_vel_vec, impact_p_corr)
        # We calculate the velocity detected by the sensor
        sensor_v = np.dot(lineal_v, sensor_dir)  # Obtained in mm/s

        # We convert to º/s for noise interpolation
        angular_vel = np.linalg.norm(angular_vel_vec) * 180 / np.pi
        return sensor_v, angular_vel
    
    def get_last_measurements(self, **kwargs) -> None:
        '''
        This function obtains the latest measurements from the glasses' sensors and stores them in the self.last_measurement object

        Parameters
        ----------
        None

        **kwargs : dict, optional
            Further define the behavior of the function. Valid keys are:\n
            - 'all_sclera': bool: Indicates if all measurements are to be obtained as if the sensors hit on the sclera. Default is False
            - 'static': bool: Indicates if all measurements are to be obtained as if the user were static, i.e., there were no eye movement.
            Default is False
            - 'reset_rotation': bool: Indicates if the eye rotation information is to be cleared (True) or not (False). Default is True

        Returns
        -------
        None
        '''
        # We obtain the parameters from kwargs
        all_sclera = kwargs.get('all_sclera', False)
        static = kwargs.get('static', False)
        reset_rotation = kwargs.get('reset_rotation', True)

        # We obtain the measurements from the sensors and store them in a dictionary
        measurements_dict = {}
        na = 1.0 # Refractive index of air
        for sensor_i in self.glasses.sensors_list:
            # We create the measurement object
            meas_i = Measurement()
            # We obtain the refractive index of the corresponding eye
            nc = self.user.eyes[sensor_i.lor].n_cornea
            # Transform the sensor to the head coordinate system
            new_origin, new_direction = self.transform_sensor_to_HCS(sensor_i)

            # Create the sphere and plane that represent the eye
            this_eye = self.user.eyes[sensor_i.lor]
            sclera_sphere = Sphere(this_eye.location, this_eye.r_sclera) 
            retina_sphere = Sphere(this_eye.location, this_eye.r_retina)
            cornea_center = np.array(this_eye.location) + np.array(this_eye.optical_axis)/np.linalg.norm(this_eye.optical_axis) * this_eye.h_cornea
            cornea_sphere = Sphere(cornea_center, this_eye.r_cornea)
            iris_plane = Plane(np.array(this_eye.location) + np.array(this_eye.optical_axis)/np.linalg.norm(this_eye.optical_axis) * this_eye.d_iris, this_eye.optical_axis)
            optical_line = Line(this_eye.location, this_eye.optical_axis)
            r_pupil = this_eye.r_pupil

            # Now, we calculate the intersection points with the sclera
            point_a, point_b = sclera_sphere.intersect_line(Line(new_origin, new_direction))
            # We choose the closest point to the sensor
            if np.linalg.norm(point_a-new_origin) < np.linalg.norm(point_b-new_origin):
                scleral_point = point_a
            else:
                scleral_point = point_b
            scleral_distance = np.linalg.norm(scleral_point-new_origin)

            # Now, we calculate the intersection points with the cornea, if it would be
            try:
                point_c, point_d = cornea_sphere.intersect_line(Line(new_origin, new_direction))
                if np.linalg.norm(point_c-new_origin) < np.linalg.norm(point_d-new_origin):
                    corneal_point = point_c
                else:
                    corneal_point = point_d
                cornea_distance = np.linalg.norm(corneal_point-new_origin)
            except:
                corneal_point = []
                cornea_distance = 1000000 # A very big number     

            if (scleral_distance < cornea_distance or all_sclera):
                meas_i.surface = 'sclera'
                final_point = scleral_point
                path_points = [final_point]
                d_i = na*np.linalg.norm(final_point - new_origin)
            else:
                # The beam hit on the cornea surface so we have to calculate the refracted ray
                # We calculate the normal vector of the cornea
                cornea_normal = (corneal_point - cornea_center) / np.linalg.norm(corneal_point - cornea_center) # Outward normal vector
                # We apply Snell's law to calculate the refracted ray in its vectorial form
                mu = na / nc
                incident_cosine = - np.dot(new_direction, cornea_normal)
                refracted_direction = mu * new_direction + (mu * incident_cosine - np.sqrt(1 - mu**2 * (1 - incident_cosine**2))) * cornea_normal
                new_direction = refracted_direction / np.linalg.norm(refracted_direction)

                # Now, we calculate the intersection with the iris plane
                iris_point = iris_plane.intersect_line(Line(corneal_point, new_direction))
                d_inter_iris = optical_line.distance_point(iris_point)
                if r_pupil < d_inter_iris:
                    final_point = iris_point
                    meas_i.surface = 'iris'
                    path_points = [corneal_point, final_point]
                else:
                    meas_i.surface = 'retina'
                    # This is only meaningful if the sensor hits on the pupil   
                    point_e, point_f = retina_sphere.intersect_line(Line(corneal_point, new_direction))
                    if np.linalg.norm(point_e-new_origin) > np.linalg.norm(point_f-new_origin):
                        retinal_point = point_e
                    else:
                        retinal_point = point_f
                    final_point = retinal_point
                    path_points = [corneal_point, retinal_point]
                    
                d_i = na*np.linalg.norm(corneal_point-new_origin) + nc*np.linalg.norm(final_point-corneal_point)

            meas_i.path_points = path_points
            meas_i.ID_sensor = sensor_i.ID

            # We obtain the velocity measurement
            if not static:
                v_i, vel_angular = self.simulate_velocity_measurement(new_direction, np.array(final_point), sensor_i.lor)
            else:
                v_i = 0
                vel_angular = 0

            # We introduce the noise
            if self.is_noisy:
                match self.glasses.noise_sensor_model:
                    case 'global_constant':
                        meas_i.distance = d_i + np.random.normal(0, self.glasses.global_distance_std_noise)
                        meas_i.velocity = v_i + np.random.normal(0, self.glasses.global_velocity_std_noise)
                    case 'ind_constant':
                        meas_i.distance = d_i + np.random.normal(0, sensor_i.distance_std_noise)
                        meas_i.velocity = v_i + np.random.normal(0, sensor_i.velocity_std_noise)
                    case 'ind_variable':
                        meas_i.distance = d_i + sensor_i.get_distance_noise(d_i, vel_angular)
                        meas_i.velocity = v_i + sensor_i.get_velocity_noise(d_i, vel_angular)
                    case 'no_noise': # Redundant case
                        meas_i.distance = d_i
                        meas_i.velocity = v_i
                    case _:
                        meas_i.distance = d_i
                        meas_i.velocity = v_i
            else:
                meas_i.distance = d_i
                meas_i.velocity = v_i

            meas_i.lor = sensor_i.lor
            # We create a dictionary with the measurements and the sensor ID
            measurements_dict[sensor_i.ID] = meas_i
        if reset_rotation:
            self.user.eyes[sensor_i.lor].last_rotation_vector = None # Reset the last rotation vector
        # We update the case of the measurements
        self.meas_case['left_case'], self.meas_case['right_case'] = self.get_meas_case(measurements_dict)

        self.last_measurement.clear()
        self.last_measurement.update(measurements_dict)
        return

    def set_last_measurement(self, last_measurement: dict[str, Measurement]) -> None:
        '''
        This function set the last_measurement object of the system and updates the case of the measurements

        Parameters
        ----------
        last_measurement : dict[str, Measurement]
            last_measurement-like object to be set in the system

        Returns
        -------
        None
        '''
        # We update the measurement without breaking a possible link with the demonstrator's last_measurement object
        self.last_measurement.clear()
        self.last_measurement.update(last_measurement)

        # We update the case of the measurements
        self.meas_case['left_case'], self.meas_case['right_case'] = self.get_meas_case(last_measurement)
        return

    def show(self, lor: list[str] = ['left','right'], **kwargs) -> pv.Plotter:
        '''
        This function shows the system in a 3D plot using PyVista. Output can be displayed via its own show method

        Parameters
        ----------
        lor : list[str], optional
            Sides of the eyes to show (e.g., ['left', 'right']). Default is ['left','right']
        **kwargs : dict, optional
            Further define the behavior of the function. Valid keys are:\n
            - 'off_screen': bool: Indicates if the rendering is off-screen (True) or interactive (False). Default is False
            - 'add_plots': list[str]: List of additional plots to be shown. Possible values are 'visual_axis', 'velocity_plane' and
            'reference_systems'. Default is []

        Returns
        -------
        pl : pyvista.Plotter
            PyVista Plotter object that shows the system
        '''
        # We obtain the parameters from kwargs
        off_screen = kwargs.get('off_screen', False)
        add_plots = kwargs.get('add_plots', [])

        if off_screen:
            pl = pv.Plotter(lighting='none', off_screen=True, window_size=[3840, 2160])
        else:
            pl = pv.Plotter(lighting='none', off_screen=False, window_size=[1920, 1080])
        # Anti-aliasing and EDL help reduce jagged edges and patchiness
        pl.enable_anti_aliasing('ssaa')
        try:
            pl.enable_eye_dome_lighting()
        except Exception:
            # EDL may not be available in older VTK/PyVista versions; safely ignore
            pass

        # Draw the eyes
        for eye_key in lor:
            self.user.eyes[eye_key].show(pl, plot_va='visual_axis' in add_plots)

        # Draw the sensors and their paths
        for sensor_i in self.glasses.sensors_list:
            if sensor_i.lor in lor:
                new_origin, new_direction = self.transform_sensor_to_HCS(sensor_i)
                previous_point = new_origin
                if self.last_measurement[sensor_i.ID].surface == 'sclera':
                    sensor_color = 'b'
                elif self.last_measurement[sensor_i.ID].surface == 'iris':
                    sensor_color = 'g'
                else:
                    sensor_color = 'r'

                for point_i in self.last_measurement[sensor_i.ID].path_points:
                    pl.add_mesh(pv.Line(previous_point, point_i), color=sensor_color, line_width=5, opacity=1)
                    previous_point = point_i
        
        # We make the camera point to the left or right eye
        if lor == ['left']:
            cam_pos = -30
        elif lor == ['right']:
            cam_pos = 30
        else:
            cam_pos = 0

        pl.camera_position = [(-72.8311066083788, -55.21310042329851, 82.45184895111981),       # Camera location
                              (-3.9235209929515946, 8.510522292048218, 3.442119797480064),      # Focal point
                              (0.30183130460034047, -0.8533735747946016, -0.4250310640477221)]  # View-up direction

        # Key light
        UFO = pv.Light(position=(cam_pos, 0, 100), focal_point=(cam_pos, 0, 0), color='white')
        UFO.positional = True
        UFO.cone_angle = 80
        UFO.exponent = 4  # softer falloff
        UFO.intensity = 0.9
        pl.add_light(UFO)

        # Fill light to lower harsh contrast and reduce "patchy" look
        fill = pv.Light(position=(cam_pos, -120, 60), focal_point=(cam_pos, 0, 0), color='white')
        fill.positional = True
        fill.cone_angle = 100
        fill.exponent = 2
        fill.intensity = 0.5
        pl.add_light(fill)

        # Disable hard shadows (can create blocky patches depending on GPU/driver)
        # pl.enable_shadows()
        
        # Captures image with off-screen rendering, then shows interactively
        # pl.screenshot('figure.png', window_size=[3840, 2160])
        # pl.show(screenshot='figure.png', window_size=[3840, 2160])
        # pl.show()
        
        if 'reference_systems' in add_plots:
            # We show a coordinate system for reference (head coordinate system)
            # Create axes with larger lines manually
            origin = np.array([0, 0, 0])
            axis_length = 8  # Length of each axis in mm
            arrow_length = 2  # Length of the arrowhead
            arrow_radius = 0.5  # Radius of the arrowhead
            line_width = 10  # Width of the axis lines
            
            # X axis (red)
            x_end = origin + np.array([axis_length, 0, 0])
            pl.add_mesh(pv.Line(origin, x_end), color='r', line_width=line_width)
            arrow_x = pv.Cone(direction=[1, 0, 0], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_x = arrow_x.translate(x_end)
            pl.add_mesh(arrow_x, color='r')
            
            # Y axis (green)
            y_end = origin + np.array([0, axis_length, 0])
            pl.add_mesh(pv.Line(origin, y_end), color='g', line_width=line_width)
            arrow_y = pv.Cone(direction=[0, 1, 0], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_y = arrow_y.translate(y_end)
            pl.add_mesh(arrow_y, color='g')
            
            # Z axis (blue)
            z_end = origin + np.array([0, 0, axis_length])
            pl.add_mesh(pv.Line(origin, z_end), color='b', line_width=line_width)
            arrow_z = pv.Cone(direction=[0, 0, 1], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_z = arrow_z.translate(z_end)
            pl.add_mesh(arrow_z, color='b')

            # We add the glasses' coordinate system
            glasses_origin = self.glasses.location_T
            axis_length = 8  # Same length for consistency
            
            # X axis (red)
            glasses_x_end = glasses_origin + np.array([axis_length, 0, 0])
            pl.add_mesh(pv.Line(glasses_origin, glasses_x_end), color='r', line_width=line_width)
            arrow_gx = pv.Cone(direction=[1, 0, 0], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_gx = arrow_gx.translate(glasses_x_end)
            pl.add_mesh(arrow_gx, color='r')
            
            # Y axis (green)
            glasses_y_end = glasses_origin + np.array([0, axis_length, 0])
            pl.add_mesh(pv.Line(glasses_origin, glasses_y_end), color='g', line_width=line_width)
            arrow_gy = pv.Cone(direction=[0, 1, 0], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_gy = arrow_gy.translate(glasses_y_end)
            pl.add_mesh(arrow_gy, color='g')
            
            # Z axis (blue)
            glasses_z_end = glasses_origin + np.array([0, 0, axis_length])
            pl.add_mesh(pv.Line(glasses_origin, glasses_z_end), color='b', line_width=line_width)
            arrow_gz = pv.Cone(direction=[0, 0, 1], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_gz = arrow_gz.translate(glasses_z_end)
            pl.add_mesh(arrow_gz, color='b')

        if 'velocity_plane' in add_plots:
            # We show the current velocity plane on both eyes
            for lor_i in lor:
                # We obtain the vector perpendicular to the velocity plane
                VP = self.get_current_VP(lor_i, verbose=False)
                # We draw a plane perpendicular to the VP vector, centered on the eye
                eye_center = np.array(self.user.eyes[lor_i].location)
                plane_size = 22
                plane_points = np.array([
                    [-plane_size, -plane_size, 0.0],
                    [-plane_size,  plane_size, 0.0],
                    [ plane_size,  plane_size, 0.0],
                    [ plane_size, -plane_size, 0.0]
                ])
                # We rotate the plane points to be perpendicular to the VP vector
                VP_normalized = VP / np.linalg.norm(VP)
                # We calculate the rotation axis and angle to align the plane with the VP vector
                rotation_axis = np.cross([0, 0, 1], VP_normalized)
                rotation_angle = np.arccos(np.clip(np.dot([0, 0, 1], VP_normalized), -1.0, 1.0))
                if np.linalg.norm(rotation_axis) < 1e-12:
                    rotation = R.from_rotvec([0.0, 0.0, 0.0])
                else:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    rotation = R.from_rotvec(rotation_axis * rotation_angle)
                plane_points_rotated = rotation.apply(plane_points) + eye_center
                # We add the rotated plane to the plot
                faces = np.array([4, 0, 1, 2, 3])
                plane_mesh = pv.PolyData(plane_points_rotated, faces=faces)
                pl.add_mesh(plane_mesh, color='navy', opacity=0.5)

        return pl

    def look_at_this_point(self, point3D: list[float], single_eye: bool = False, **kwargs) -> None:
        '''
        This function makes the eye look at a given 3D point and stores the optical and visual axes in the historic list

        Parameters
        ----------
        point3D : list[float]
            3D point to look at
        single_eye : bool, optional
            Indicates whether both eyes must look at the point (False) or a single eye (left or right) is enough (True). Default is False
            (both eyes)
        **kwargs : dict, optional
            Further define the behavior of the function. Valid keys are:\n
            - 'lor': str: Indicates whether the eye to use is the left or right ('left' or 'right'). It is taken into account only if
            single_eye is True. Default is 'left'
            - 'complete_mov': bool: Indicates whether, in the movement to point3D, a complex rotation taking into account Listing's law should
            be considered (True) or if it is a simple movement to point3D (False). Default is False (simple movement)
            - 'update_vergence': bool: Indicates whether to update the user's vergence after looking at the 3D point. Default is True

        Returns
        -------
        None
        '''
        # We obtain the parameters from kwargs
        lor = kwargs.get('lor', 'left')
        complete_mov = kwargs.get('complete_mov', False)
        update_vergence = kwargs.get('update_vergence', True)

        # We make the eye(s) look at the point and we store the new optical and visual axes in the historic list
        if (single_eye):
            self.user.eyes[lor].look_at_this_point(point3D)
            self.gaze_trajectory.add_optical_axis_sample(lor, self.user.eyes[lor].optical_axis)
            self.gaze_trajectory.add_visual_axis_sample(lor, self.user.eyes[lor].visual_axis)
        else:
            if complete_mov:
                # We produce the rotation of the visual axis taking into account Listing's law
                self.user.eyes['left'].look_at_this_point(point3D, vergence=self.user.vergence)
                self.user.eyes['right'].look_at_this_point(point3D, vergence=self.user.vergence)
            else:
                # We produce the rotation of the visual axis in a simple movement
                self.user.eyes['left'].look_at_this_point(point3D)
                self.user.eyes['right'].look_at_this_point(point3D)
            if update_vergence:
                self.user.update_vergence()
            # We store the new optical and visual axes in the historic list
            self.gaze_trajectory.add_optical_axis_sample('left', self.user.eyes['left'].optical_axis)
            self.gaze_trajectory.add_optical_axis_sample('right', self.user.eyes['right'].optical_axis)
            self.gaze_trajectory.add_visual_axis_sample('left', self.user.eyes['left'].visual_axis)
            self.gaze_trajectory.add_visual_axis_sample('right', self.user.eyes['right'].visual_axis)
        return
    
    def look_in_this_direction(self, target_axis: list[float], lor: str, **kwargs) -> None:
        '''
        This function makes the eye look in a given direction and stores the optical and visual axes in the historic list

        Parameters
        ----------
        target_axis : numpy array
            Optical axis to look at
        lor : str
            Indicates whether the eye to use is the left or right ('left' or 'right')
        **kwargs : dict, optional
            Further define the behavior of the function. Valid keys are:\n
            - 'complete_mov': bool: Indicates whether, in the movement to point3D, a complex rotation taking into account Listing's law should
            be considered (True) or if it is a simple movement to point3D (False). Default is False (simple movement)
            - 'update_vergence': bool: Indicates whether to update the user's vergence after looking at the 3D point. Default is True

        Returns
        -------
        None
        '''
        # We obtain the parameters from kwargs
        complete_mov = kwargs.get('complete_mov', False)
        update_vergence = kwargs.get('update_vergence', True)

        # We make the eye look in the direction of the target axis and we store the new optical and visual axes in the historic list
        if complete_mov:
            # We produce the rotation of the visual axis taking into account Listing's law
            self.user.eyes[lor].look_in_this_direction(target_axis, vergence=self.user.vergence)
        else:
            # We produce the rotation of the visual axis in a simple movement
            self.user.eyes[lor].look_in_this_direction(target_axis)
        if update_vergence:
            self.user.update_vergence()
        # We store the new optical and visual axes in the historic list
        self.gaze_trajectory.add_optical_axis_sample(lor, self.user.eyes[lor].optical_axis)
        self.gaze_trajectory.add_visual_axis_sample(lor, self.user.eyes[lor].visual_axis)
        return
    
    def rotate_eyeball(self, rot_vector: list[float], lor: str) -> None:
        '''
        This function rotates the eyeball in a given direction and stores the optical and visual axes in the historic list

        Parameters
        ----------
        rot_vector : list[float]
            Rotation vector of the eyeball [rad]
        lor : str
            Indicates whether the eye to use is the left or right ('left' or 'right')

        Returns
        -------
        None
        '''
        self.user.eyes[lor].rotate_eyeball(rot_vector)
        # We store the new optical and visual axes in the historic list
        self.gaze_trajectory.add_optical_axis_sample(lor, self.user.eyes[lor].optical_axis)
        self.gaze_trajectory.add_visual_axis_sample(lor, self.user.eyes[lor].visual_axis)
        return
    
    def iterable_produce_angular_velocity(self, lor: str, initial_point: list[float], second_point: list[float], angular_velocity: float, velocity_model: str, n_samples: int, version: str = 'angular_velocity') -> iter:
        '''
        This iterable produces a movement of an eye at a given angular velocity, in the direction of the gaze
        indicated by the input parameters. Unlike previous versions, this version produces the rotation of the eye by itself;
        and it is not necessary to call the functions look_at_this_point, look_in_this_direction, etc.

        Parameters
        ----------
        lor : string
            Indicates whether the eye being studied is the left or right
        initial_point : numpy array
            Initial point of the trajectory
        second_point : numpy array
            Second point of the trajectory
        angular_velocity : float
            Angular velocity of the movement [rad/s]. It is taken into account in the 'angular_velocity' version
        velocity_model : string
            Angular velocity model ('constant')
        n_samples : int
            Number of samples to take during the movement. It is taken into account in the 'n_samples' version
        version : str, optional
            Indicates the version of the iterable to use ('angular_velocity', 'n_samples'). Default is 'angular_velocity'

            In angular_velocity, eye rotates for n_samples instants at a constant angular velocity equal to that indicated by angular_velocity

            In n_samples, eye rotates a constant angle at each instant, so that it looks at second_point at the end of the movement

        Returns
        -------
        iter
            Yields True if the movement is complete, False otherwise
        '''
        # We obtain the optical axes at the beginning and at the end of the movement
        visaxis_prev = np.array(initial_point) - np.array(self.user.eyes[lor].location)
        visaxis_prev = visaxis_prev / np.linalg.norm(visaxis_prev)
        visaxis_new = np.array(second_point) - np.array(self.user.eyes[lor].location)
        visaxis_new = visaxis_new / np.linalg.norm(visaxis_new)

        # We calculate the velocity plane position taking vergence into account
        LP = [np.sin(self.user.eyes[lor].LP_phi)*np.cos(self.user.eyes[lor].LP_theta), np.sin(self.user.eyes[lor].LP_phi)*np.sin(self.user.eyes[lor].LP_theta), np.cos(self.user.eyes[lor].LP_phi)]

        if lor == 'left':
            rot_VP = np.array([0, -self.user.vergence * self.user.eyes[lor].VP_vergence_var, 0])
        else:
            rot_VP = np.array([0, self.user.vergence * self.user.eyes[lor].VP_vergence_var, 0])
        VP = R.from_rotvec(rot_VP).apply(LP)
        VP = VP / np.linalg.norm(VP)

        # We calculate the rotations up to the primary position and up to the final position
        R1 = rot_mat_align_vectors(visaxis_prev, VP)
        R2 = rot_mat_align_vectors(VP, visaxis_new)
        # We calculate the total rotation
        comp_axis = R.from_matrix(np.matmul(R2, R1)).as_rotvec()

        # We calculate the angle increment and the number of samples
        match version:
            case 'angular_velocity':
                angle_inc = angular_velocity / self.glasses.freq  # rad per sample
                n_samples = int(np.floor(np.linalg.norm(comp_axis) / angle_inc))
            case 'n_samples':
                angle_inc = np.linalg.norm(comp_axis) / n_samples # rad per sample
        
        # We normalize the rotation axis
        comp_axis = comp_axis / np.linalg.norm(comp_axis)

        match velocity_model:
            case 'constant':
                # We produce the rotation needed in each sample
                rot_vector = comp_axis * angle_inc

                complete_mov = False

                for i in range(n_samples):
                    # We rotate the eyeball
                    self.rotate_eyeball(rot_vector, lor)

                    if (i == n_samples - 1) and version == 'n_samples':
                        complete_mov = True

                    yield complete_mov
                    
                if (version == 'angular_velocity'):
                    # Small step to properly adjust the trajectory of the visual axis
                    self.look_at_this_point(second_point, single_eye=True, lor=lor, update_vergence=False)
                    complete_mov = True
                    yield complete_mov
    
    def eye_to_primary_pos(self) -> None:
        '''
        This function updates the optical and visual axes of the user's eyes based on their parameters, returning to a standard position

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.user.eyes['left'].eye_to_primary_pos()
        self.user.eyes['right'].eye_to_primary_pos()
        return

    def get_current_VP(self, lor: str, **kwargs) -> np.ndarray:
        '''
        This function returns the vector perpendicular to the velocity plane of the chosen eye in 3D space

        Parameters
        ----------
        lor : string
            Indicates whether the studied eye is the left or right one
        **kwargs : dict, optional
            Further define the behavior of the function. Valid keys are:\n
            - 'verbose': bool: Indicates whether to print the VP coordinates (True) or not (False). Default is False
        
        Returns
        -------
        VP : np.ndarray
            Vector perpendicular to the velocity plane of the chosen eye in 3D space
        '''
        # We obtain the parameters from kwargs
        verbose = kwargs.get('verbose', False)
        
        # We calculate the Listing plane position
        LP = [np.sin(self.user.eyes[lor].LP_phi)*np.cos(self.user.eyes[lor].LP_theta), np.sin(self.user.eyes[lor].LP_phi)*np.sin(self.user.eyes[lor].LP_theta), np.cos(self.user.eyes[lor].LP_phi)]

        # We calculate the rotation due to vergence
        if lor == 'left':
            rot_VP = np.array([0, -self.user.vergence * self.user.eyes[lor].VP_vergence_var, 0])
        else:
            rot_VP = np.array([0, self.user.vergence * self.user.eyes[lor].VP_vergence_var, 0])
        VP = R.from_rotvec(rot_VP).apply(LP)
        VP = VP / np.linalg.norm(VP)

        if verbose:
            # We calculate the phi and theta coordinates
            phi_VP = np.arccos(VP[2]) * 180/np.pi
            theta_VP = np.atan2(VP[1], VP[0]) * 180/np.pi
            if lor == 'left':
                theta_VP = 180 - theta_VP
            print('VP:', lor, 'theta:', str(round(theta_VP, 3)).ljust(7), 'phi:', str(round(phi_VP, 3)).ljust(7))
        return VP

    def adapt_point_to_isovergence(self, point: list[float], reference_point: list[float] = None) -> np.ndarray:
        '''
        This function adapts a 3D point to the vergence of the visual system

        Parameters
        ----------
        point : list[float]
            The 3D point to adapt
        reference_point : list[float], optional
            A reference point to calculate the vergence. If not provided, the current vergence of the system is used. Default is None

        Returns
        -------
        final_point : np.ndarray
            The point, adapted to the vergence of the visual system
        '''
        # We obtain the vergence to use
        if reference_point is None:
            vergence = self.user.vergence
        else:
            vec_left = np.array(reference_point) - np.array(self.user.eyes['left'].location)
            vec_left = vec_left / np.linalg.norm(vec_left)
            vec_right = np.array(reference_point) - np.array(self.user.eyes['right'].location)
            vec_right = vec_right / np.linalg.norm(vec_right)
            vergence = np.arccos(np.dot(vec_left, vec_right))
        
        # We obtain the parameters of the spindle torus of same vergence
        R_sphere = np.linalg.norm(np.array(self.user.eyes['left'].location) - np.array(self.user.eyes['right'].location)) / (2 * np.sin(vergence))
        d_axis = R_sphere * np.cos(vergence)

        # We obtain the generating sphere of the torus aligned with the given point
        middle_point = (np.array(self.user.eyes['right'].location) + np.array(self.user.eyes['left'].location)) / 2
        middle_2_point_path = np.array(point) - middle_point
        middle_2_point_path[0] = 0 # We project the path onto the YZ plane
        middle_2_point_path = middle_2_point_path / np.linalg.norm(middle_2_point_path)
        sphere_center = middle_point + middle_2_point_path * d_axis
        sphere = Sphere(sphere_center, R_sphere)

        # We obtain the intersection of the sphere with the line between the center of the eyes and the given point
        intersection = sphere.intersect_line(Line(middle_point, (np.array(point) - middle_point) / np.linalg.norm(np.array(point) - middle_point)))
        if intersection[0][2] > 0:
            final_point = intersection[0]
        else:
            final_point = intersection[1]

        return final_point
    
    def simulate_tertiary_movement(self, point1: list[float], point2: list[float], mov_type: str, mode: str, **kwargs) -> tuple[list[dict[str, Measurement]], dict[str, dict[str, Measurement]]]:
        '''
        This function simulates the eye movement between two tertiary positions given by the points point1 and point2

        Parameters
        ----------
        point1 : list[float]
            Initial point of the trajectory
        point2 : list[float]
            Final point of the trajectory
        mov_type : str
            Angular velocity model ('constant')
        mode : str
            Indicates the version of the iterable to use ('angular_velocity', 'n_samples'). Default is 'angular_velocity'

            In angular_velocity, eye rotates at a constant angular velocity equal to that indicated by angular_velocity

            In n_samples, eye rotates a constant angle at each instant, so that it looks at second_point after n_samples samples
        **kwargs : dict, optional
            Further define the behavior of the function. Valid keys are:\n
            - 'n_samples': int: Duration in number of samples of the movement. It is taken into account in the 'n_samples' version
            - 'angular_velocity': float: Angular velocity of the movement [rad/s]. It is taken into account in the 'angular_velocity' version
            - 'obtain_measurements': bool: Indicates whether to obtain the measurements of an instant where a certain condition is met (see
            meas_out_cond). By default it is False
            - 'meas_out_cond': tuple[int, str]: Condition to consider for obtaining the output measurement. It is a tuple where the first
            element is the minimum number of sensors and the second element is the surface name. By default it is (4, 'sclera')

        Returns
        -------
        measurements_array : list[dict[str, Measurement]]
            List of measurements taken during the movement
        measurements_out : dict[str, dict[str, Measurement]]
            Dictionary with the first measurement where at least 4 sensors hit the sclera for each eye (only if obtain_measurements is True).
            If obtain_measurements is False, this output is returned as an empty dictionary
        '''
        # We obtain the parameters from kwargs
        n_samples = kwargs.get('n_samples', None)
        angular_velocity = kwargs.get('angular_velocity', None)
        obtain_measurements = kwargs.get('obtain_measurements', False)
        meas_out_cond = kwargs.get('meas_out_cond', (4, 'sclera'))

        # We check that the necessary parameters are provided for the selected mode
        if (mode == 'angular_velocity' and angular_velocity is None) or (mode == 'n_samples' and n_samples is None):
            raise ValueError("If mode is 'angular_velocity', angular_velocity must be provided. If mode is 'n_samples', n_samples must be provided.")

        # We make the movement of the eyes

        self.look_at_this_point(point1, complete_mov=True)

        # We create the iterables for both eyes
        iter_left = self.iterable_produce_angular_velocity('left', point1, point2, angular_velocity, mov_type, n_samples, mode)
        iter_right = self.iterable_produce_angular_velocity('right', point1, point2, angular_velocity, mov_type, n_samples, mode)

        # We simulate the movement
        measurements_array = []
        measurements_out = {'left': [], 'right': []}
        measurements_recorded = {'left': False, 'right': False}
        complete_mov_left = False
        complete_mov_right = False
        while (not complete_mov_left) or (not complete_mov_right):
            # We make the next step in the movement of each eye if they have not completed it yet
            if not complete_mov_left:
                complete_mov_left = next(iter_left)
            if not complete_mov_right:
                complete_mov_right = next(iter_right)

            # We obtain the measurements at this instant
            self.get_last_measurements()
            if obtain_measurements and (not measurements_recorded['left'] or not measurements_recorded['right']):
                # We check if the output measurement condition is met
                for lor in ['left', 'right']:
                    cont_surface = 0
                    for sensor in self.glasses.sensors_list:
                        if sensor.lor == lor and self.last_measurement[sensor.ID].surface == meas_out_cond[1]:
                            cont_surface += 1
                    if cont_surface >= meas_out_cond[0] and not measurements_recorded[lor]:
                        # We store the measurement
                        measurements_out[lor] = self.last_measurement.copy()
                        measurements_recorded[lor] = True

            # We store the measurement of this instant
            measurements_array.append(self.last_measurement.copy())
        
        # We update the vergence of the user at the end of the movement
        self.user.update_vergence()

        return measurements_array, measurements_out

class System_demonstrator:

    def __init__(self, embedded_sys: bool = False, **kwargs) -> None:
        self.embedded_sys = embedded_sys                        # Indicates whether the system is embedded within an umbrella system or not
        self.approx_gaze_trajectory = Gaze_trajectory()         # Gaze trajectory object
        self.estimations = User()                               # User object with the estimations of the system
        
        if embedded_sys:
            self.glasses = kwargs.get('glasses', Glasses())                             # Glasses object
            self.ground_truth = kwargs.get('user', self.estimations)                    # Ground truth User object. Only makes sense in
                                                                                        # embedded systems
            self.last_measurement = kwargs.get('last_measurement', {})                  # Last measurement object
            self.meas_case = kwargs.get('meas_case', {'left_case': 0, 'right_case': 0}) # Measurement case dictionary
        else:
            self.glasses = Glasses()                            # Glasses object
            self.ground_truth = self.estimations                # Ground truth User object
            self.last_measurement = dict[str, Measurement]()    # Last measurement object
            self.meas_case = {'left_case': 0, 'right_case': 0}  # Measurement case dictionary
        return

    def to_dict(self) -> dict[str, any]:
        if self.embedded_sys:
            return {'approx_gaze_trajectory':self.approx_gaze_trajectory.to_dict(),
                    'estimations':self.estimations.to_dict()}
        else:
            return {'approx_gaze_trajectory':self.approx_gaze_trajectory.to_dict(),
                    'estimations':self.estimations.to_dict(),
                    'glasses':self.glasses.to_dict(),
                    'ground_truth':self.ground_truth.to_dict(),
                    'last_measurement':{key: value.to_dict() for key, value in self.last_measurement.items()},
                    'meas_case':self.meas_case}
    
    def from_dict(self, system_dict: dict[str, any]) -> None:
        if not self.embedded_sys:
            if 'glasses' in system_dict:
                self.glasses.from_dict(system_dict['glasses'])
            if 'ground_truth' in system_dict:
                self.ground_truth.from_dict(system_dict['ground_truth'])
            if 'last_measurement' in system_dict:
                self.last_measurement.clear()
                for key, value in system_dict['last_measurement'].items():
                    meas = Measurement()
                    meas.from_dict(value)
                    self.last_measurement[key] = meas
            if 'meas_case' in system_dict:
                self.meas_case = system_dict['meas_case']

        if 'approx_gaze_trajectory' in system_dict:
            self.approx_gaze_trajectory.from_dict(system_dict['approx_gaze_trajectory'])
        if 'estimations' in system_dict:
            self.estimations.from_dict(system_dict['estimations'])
        return

    def set_configuration(self, config_file: str | None) -> None:
        if self.embedded_sys:
            raise ValueError('The configuration of the system cannot be set in an embedded system demonstrator. Use the System_umbrella class instead')
        if config_file is not None:
            with open(config_file, 'r') as f:
                system_dict = json.load(f)
            self.from_dict(system_dict)
            # Assure the coherence of the configuration of iris parameters
            self.estimations.eyes['left'].update_d_iris() 
            self.estimations.eyes['left'].update_h_cornea()
            self.estimations.eyes['right'].update_d_iris()
            self.estimations.eyes['right'].update_h_cornea()
        else:
            raise ValueError('No configuration file provided')
        return

    def obtain_subsets(self, original_set: list[object]) -> list[list[object]]:
        '''
        This function obtains all the length-3-subsets of the original list, taking the elements 3 by 3, without repetition and not taking
        into account the order of the elements

        Parameters
        ----------
        original_set : list[object]
            Original list of elements. It is assumed to be of length greater than or equal to 3

        Returns
        -------
        all_subsets : list[list[object]]
            List of possible lists obtained from the original set
        '''
        all_subsets = []

        for i in range(len(original_set)):
            for j in range(i + 1, len(original_set)):
                for k in range(j + 1, len(original_set)):
                    all_subsets.append([original_set[i], original_set[j], original_set[k]])

        return all_subsets

    def line_point_distance(self, p: np.ndarray, d: float) -> list[np.ndarray]:
        '''
        This function estimates the directional vector of the line that passes through point p and is at a distance d from the origin.
        For this function, we are working in 2D

        Parameters
        ----------
        p : np.ndarray
            Point in 2D space
        d : float
            Distance from the origin to the line

        Returns
        -------
        out : list[np.ndarray]
            List with the directional vector of the line that passes through point p and is at a distance d from the origin
        '''
        # For this function, we are working in 2D.
        px = p[0]
        py = p[1]
        mod_p = np.linalg.norm(p)
        if mod_p == 0: # The point is the origin. Not probable, but we have to consider it
            out = []
        elif np.abs(mod_p-d) < 1e-12: # The point is in the circle of radius d
            v = np.array([py, -px])
            v = v / np.linalg.norm(v)
            out = [v]
        elif mod_p > d: # The point is outside the circle of radius d
            vx1 = -(d*py**2-px*np.sqrt((px**2+py**2-d**2)*py**2))/(py*px**2+py**3)
            vy1 = (d*px+np.sqrt((px**2+py**2-d**2)*py**2))/(px**2+py**2)
            vx2 = -(d*py**2+px*np.sqrt((px**2+py**2-d**2)*py**2))/(py*px**2+py**3)
            vy2 = (d*px-np.sqrt((px**2+py**2-d**2)*py**2))/(px**2+py**2)
            v1 = np.array([vx1, vy1])
            v2 = np.array([vx2, vy2])
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            out = [v1, v2]
        else: # The point is inside the circle of radius d
            v = np.array([py, -px])
            v = v / np.linalg.norm(v)
            out = [v]
        return out
        
    def planes_two_points_distance_center(self, p1: np.ndarray, p2: np.ndarray, center: np.ndarray, d: float) -> list[np.ndarray]:
        '''
        This function estimates the normal vectors of the planes that pass through points p1 and p2 and are at a distance d from the center

        Parameters
        ----------
        p1 : np.ndarray
            First point in 3D space
        p2 : np.ndarray
            Second point in 3D space
        center : np.ndarray
            Center at distance d from the output planes
        d : float
            Distance from the center to the plane

        Returns
        -------
        normals : list[np.ndarray]
            List with the normal vectors of the planes
        '''
        v = p2 - p1
        R = rot_mat_align_vectors(v, [0,0,1])
        p1_r = np.matmul(R, p1)
        p2_r = np.matmul(R, p2)
        center_r = np.matmul(R, center)
        p1_r_t = p1_r - center_r
        p2_r_t = p2_r - center_r

        # We simplify the problem to 2D
        p1_r_t_2D = p1_r_t[:2]
        p2_r_t_2D = p2_r_t[:2] # It must be equal to p1_t_r_2D
        directional_vector_2D_list = self.line_point_distance(p1_r_t_2D, d)
        normals = []
        R_inv = rot_mat_align_vectors([0,0,1], v)
        if directional_vector_2D_list is not None:
            for directional_vector_2D in directional_vector_2D_list:
                normal_vector = np.array([-directional_vector_2D[1], directional_vector_2D[0], 0])
                normal_vector = np.matmul(R_inv, normal_vector)  
                normal_vector = normal_vector / np.linalg.norm(normal_vector)
                if normal_vector[2] < 0:
                    normal_vector = -normal_vector # Always the estimation vector to front (z>0)
                normals.append(normal_vector)
        return normals

    def estimate_plane(self, points_list: list[np.ndarray], lor: str, ideal_center_Q: bool = True, invert_z_axis: bool = False) -> tuple[np.ndarray, float]:
        '''
        This function estimates the normal vector of the plane that best fits the given points and its distance to the center of the eye

        Parameters
        ----------
        points_list : list[np.ndarray]
            List of points in 3D space representing the hitting points on the iris
        lor : str
            Indicates whether the studied eye is the left or right
        ideal_center_Q : bool, optional
            Indicates whether the calculations should be made with the nominal eye parameters (True) or with the estimated ones (False).
            Default is True
        invert_z_axis : bool, optional
            Indicates whether the desired optical axis should have a negative z component (True) or positive (False).
            Default is False

        Returns
        -------
        normal_vector : np.ndarray
            Normal vector of the estimated plane
        distance : float
            Distance from the estimated plane to the center of the eye
        '''
        normal_vector = None
        distance = None
        if len(points_list) >=3:
            # We calculate the plane that best fits the points
            plane = Plane.best_fit(points_list) 
            if ideal_center_Q:
                distance = np.abs(plane.distance_point_signed(self.ground_truth.eyes[lor].location))
            else:
                distance = np.abs(plane.distance_point_signed(self.estimations.eyes[lor].location))
            normal_vector = plane.normal/np.linalg.norm(plane.normal)
            # We check the direction of the normal vector, inverting it if necessary
            if invert_z_axis:
                if normal_vector[2] > 0:
                    normal_vector = -normal_vector
            else:
                if normal_vector[2] < 0:
                    normal_vector = -normal_vector
        elif len(points_list) == 2:
            # We have two points, so we have to use the estimation of the d_iris parameter
            # We have find the family of planes that contains these two points and among them the (two) planes that has a distance d_iris from the estimated eye center
            if ideal_center_Q:
                normals = self.planes_two_points_distance_center(points_list[0], points_list[1], 
                                                        np.array(self.ground_truth.eyes[lor].location), self.ground_truth.eyes[lor].d_iris)
            else:
                normals = self.planes_two_points_distance_center(points_list[0], points_list[1], 
                                                        np.array(self.estimations.eyes[lor].location), self.estimations.eyes[lor].d_iris)
            # We choose the normal vector that is in the direction of the optical axis
            if len(normals) == 2:
                if ideal_center_Q:
                    dot_0 = np.abs(np.dot(normals[0], np.array(self.ground_truth.eyes[lor].optical_axis)))
                    dot_1 = np.abs(np.dot(normals[1], np.array(self.ground_truth.eyes[lor].optical_axis)))
                else:
                    dot_0 = np.abs(np.dot(normals[0], np.array(self.estimations.eyes[lor].optical_axis)))
                    dot_1 = np.abs(np.dot(normals[1], np.array(self.estimations.eyes[lor].optical_axis)))
                if dot_0 > dot_1:
                    normal_vector = normals[0]
                else:
                    normal_vector = normals[1]
            elif len(normals) == 1:
                normal_vector = normals[0]
            # We calculate the distance
            if normal_vector is None:
                distance = None
            else:
                if ideal_center_Q:
                    distance = np.dot(points_list[0]-np.array(self.ground_truth.eyes[lor].location), normal_vector)
                else:
                    distance = np.dot(points_list[0]-np.array(self.estimations.eyes[lor].location), normal_vector)
        return normal_vector, distance

    def estimate_iris_planes(self, **kwargs) -> tuple[np.ndarray, float, np.ndarray, float]:
        '''
        This function estimates the left and right iris planes from the hitting points obtained by the sensors

        Parameters
        ----------
        None

        **kwargs : dict, optional
            Define the method for the parameter estimation. Valid keys are:\n
            'ideal_center_Q': bool: Indicates whether the calculations should be made with the nominal eye center (True) or with the
            estimated one (False). Default is False
            'custom_measurement': dict[str, Measurement]: Custom measurement to use instead of the last_measurement attribute. If not provided,
            the last_measurement attribute is used

        Returns
        -------
        n_plane_left : np.ndarray
            Normal vector of the estimated left iris plane
        d_plane_left : float
            Distance from the estimated left iris plane to the center of the eye
        n_plane_right : np.ndarray
            Normal vector of the estimated right iris plane
        d_plane_right : float
            Distance from the estimated right iris plane to the center of the eye
        '''
        # We obtain the parameters from kwargs
        ideal_center_Q = kwargs.get('ideal_center_Q', False)
        local_measurement = kwargs.get('custom_measurement', self.last_measurement)
        
        # We obtain the hitting points on the iris for each eye
        p_right_list = []
        p_left_list = []
        for sensor_i in self.glasses.sensors_list:
            # We obtain the intersection point from the measured distance
            direction_norm = np.array(sensor_i.direction) / np.linalg.norm(sensor_i.direction)
            p_i = np.array(sensor_i.origin) + direction_norm * local_measurement[sensor_i.ID].distance
            if sensor_i.lor == 'left':
                if local_measurement[sensor_i.ID].surface == 'iris':
                    p_left_list.append(p_i)
            elif local_measurement[sensor_i.ID].surface == 'iris':
                p_right_list.append(p_i)

        # We estimate the planes that best fit the points
        n_plane_left, d_plane_left = self.estimate_plane(p_left_list, 'left', ideal_center_Q)
        n_plane_right, d_plane_right = self.estimate_plane(p_right_list, 'right', ideal_center_Q)

        return n_plane_left, d_plane_left, n_plane_right, d_plane_right
    
    def adjust_sphere(self, points_list: list[list[float]], method: str = 'free', fixed_radius: float = 12, initial_guess: list[float] = None) -> Sphere:
        '''
        This function estimates the center and radius of a sphere that best fits the given points

        Parameters
        ----------
        points_list : list[list[float]]
            List of points in 3D space representing the hitting points on the sclera
        method : str, optional
            Method to use for the adjustment ('free', 'fixed', 'guided'). Default is 'free'
        fixed_radius : float, optional
            Fixed radius of the sphere to fit (only used if method is 'fixed'). Default is 12.0 mm
        initial_guess : list[float], optional
            Initial guess for the radius (only if method is 'guided') and center of the sphere.
            If not provided, the mean of the points and fixed_radius is used. Default is None

        Returns
        -------
        sphere : Sphere
            Sphere object with the estimated center and radius
        '''
        if method == 'free':
            sphere = Sphere.best_fit(points_list)
            return sphere
        elif method == 'fixed' or method == 'guided':
            # We define the cost function to minimize
            def cost_function(center_radius: list[float]) -> float:
                center = center_radius[:3]
                if method == 'fixed':
                    radius = fixed_radius
                elif method == 'guided':
                    radius = center_radius[3]

                total_error = 0.0
                for point in points_list:
                    distance = np.linalg.norm(np.array(point) - np.array(center))
                    error = (distance - radius)**2
                    total_error += error
                return total_error
            
            # Initial guess for the center and radius
            if initial_guess is not None and hasattr(initial_guess, '__len__') and len(initial_guess) >= 4:
                x0 = np.array(initial_guess[:4], dtype=float)
            else:
                points_list = np.array([np.array(p) for p in points_list])
                # We calculate the mean of the points as the initial guess for the center, moving the estimation a bit to the negative direction
                # of the last estimated optical axis to avoid local minima in the optimization
                center0 = np.mean(points_list, axis=0) - 0.5 * np.array(self.estimations.eyes['left'].optical_axis)
                # We calculate the initial guess for the radius
                radius0 = fixed_radius if (method == 'fixed' and fixed_radius is not None) else np.mean([np.linalg.norm(p - center0) for p in points_list])
                x0 = np.hstack([center0, radius0])
            
            # Bounds: allow center free, restrict radius to [11, 13] if method is 'guided'
            if method == 'guided':
                bounds = [(None, None), (None, None), (None, None), (11.0, 13.0)]
            else:
                bounds = [(None, None), (None, None), (None, None), (None, None)]

            # Minimize with a method that supports bounds (L-BFGS-B or SLSQP). L-BFGS-B typically works well.
            res = minimize(cost_function, x0, method='L-BFGS-B', bounds=bounds, options={'ftol':1e-9, 'maxiter':10000})

            # Build result sphere
            center_est = res.x[:3]
            if method == 'fixed':
                radius_est = fixed_radius
            elif method == 'guided':
                radius_est = res.x[3]

            sphere = Sphere(center_est, radius_est)

            return sphere
        else:
            raise ValueError("Invalid method. Choose 'free', 'fixed', or 'guided'.")
    
    def stat_estimate_eye_centers(self, lor: str = 'left', hist_method: str = 'simple', est_method: str = 'free', save_estimation: bool = True, **kwargs) -> tuple[np.ndarray, float]:
        '''
        This function estimates the center and radius of the left or right scleral sphere from the hitting points obtained by the sensors

        Parameters
        ----------
        lor : str, optional
            Indicates whether the eye to estimate is the left or right ('left' or 'right'). Default is 'left'
        hist_method : str, optional
            Method to use for averaging the points ('simple', 'all_points', 'avg'). 'simple' uses only the last measurement,
            'all_points' uses all points from all measurements without averaging per sensor, and 'avg' averages the points per sensor.
            Default is 'simple'
        est_method : str, optional
            Method to restraint radius in the sphere adjustment ('free', 'fixed', 'guided'). Default is 'free'
        save_estimation : bool, optional
            Indicates whether to save the estimation in the estimations object (True) or not (False). Default is True
        **kwargs : dict, optional
            Define the method for the parameter estimation. Valid keys are:\n
            - 'measurements': list[dict[str, Measurement]]: List of measurement objects containing the sensor data. If not provided,
            the last_measurement object is used. Only used if hist_method is 'all_points' or 'avg'
            - 'fixed_radius': float: Fixed radius of the sphere to fit. Only used if est_method is 'fixed'. Default is 12.0 mm

        Returns
        -------
        center : np.ndarray
            Mean center of the estimated scleral sphere
        radius : float
            Mean radius of the estimated scleral sphere
        '''
        p_list_avg = []
        p_list = {}
        
        if hist_method == 'simple':
            # We just use the last measurement
            for sensor_i in self.glasses.sensors_list:
                if sensor_i.lor == lor:
                    # We check if p_list has a key for the sensor ID; if not, we create it
                    if sensor_i.ID not in p_list:
                        p_list[sensor_i.ID] = np.empty((0, 3))
                    # We obtain the intersection point, from the measured distance
                    direction_norm = np.array(sensor_i.direction) / np.linalg.norm(sensor_i.direction)
                    p_i = np.array(sensor_i.origin) + direction_norm * self.last_measurement[sensor_i.ID].distance
                    if self.last_measurement[sensor_i.ID].surface == 'sclera':
                        p_list[sensor_i.ID] = np.vstack([p_list[sensor_i.ID], p_i])
        else:
            # We use all the measurements
            measurements = kwargs.get('measurements', [self.last_measurement])
            for i in range(len(measurements)):
                for sensor_i in self.glasses.sensors_list:
                    if sensor_i.lor == lor:
                        # We check if p_list has a key for the sensor ID; if not, we create it
                        if sensor_i.ID not in p_list:
                            p_list[sensor_i.ID] = np.empty((0, 3))
                        # We obtain the intersection point, from the measured distance
                        direction_norm = np.array(sensor_i.direction) / np.linalg.norm(sensor_i.direction)
                        p_i = np.array(sensor_i.origin) + direction_norm * measurements[i][sensor_i.ID].distance
                        if measurements[i][sensor_i.ID].surface == 'sclera':
                            p_list[sensor_i.ID] = np.vstack([p_list[sensor_i.ID], p_i])

        # We average the points
        if hist_method == 'all_points':
            # We use all the points from all sensors, without averaging per sensor
            for key in p_list:
                for point in p_list[key]:
                    p_list_avg.append(point)
            p_list_avg = np.array(p_list_avg)
        else:
            # We average the points per sensor
            for key in p_list:
                p_list_avg.append(np.mean(p_list[key], axis=0))
            p_list_avg = np.array(p_list_avg)
        
        # We clean the points, removing the nan points
        p_list_avg = p_list_avg[~np.isnan(p_list_avg).any(axis=1)]
        
        # We estimate the sphere that best fits the points
        fixed_radius = kwargs.get('fixed_radius', 12.0)
        aux_sphere = self.adjust_sphere(p_list_avg, method=est_method, fixed_radius=fixed_radius, initial_guess=self.estimations.eyes[lor].location)
        
        # Extract the center and the radius
        center = aux_sphere.point
        radius = aux_sphere.radius
        if save_estimation:
            self.estimations.eyes[lor].location = center
            self.estimations.eyes[lor].r_sclera = radius
            self.estimations.eyes[lor].update_d_iris()
                
        return center, radius
        
    def stat_estimate_eye_centers_v2(planes_by_fix: dict, plane_method: str = 'all', initial_guess: list = None, **kwargs) -> dict[str, any]:
        '''
        This function finds a 3D consensus point that is as equidistant as possible to all the planes
        
        Parameters
        ----------
        planes_by_fix : dict
            Dictionary of planes by fixation, where each key is a fixation index and each value is a dictionary with keys 'all', 'centroids'
            and 'per_sample' containing the corresponding plane estimations for that fixation. Each plane estimation is a dictionary with keys
            'normal', 'point', 'd' and optionally 'pts_used' (the points used to estimate the plane, if available). The structure is as
            follows:\n
            {fix_idx: {'all': {...}, 'centroids': {...}, 'per_sample': {...} } }
        plane_method : str, optional
            Indicates which plane information to use for each fixation ('all', 'centroids' or 'per_sample'). Default is 'all'
        initial_guess : array-like, shape (3,), optional
            Initial guess for the optimization. If None, uses the centroid of all points. Default is None
        **kwargs : dict, optional
            Define the method for the parameter estimation. Valid keys are:\n
            - 'target_distance': float: Target distance from the consensus point to the planes. Default is 10.0
            - verbose: bool: Indicates whether to print optimization information. Default is False
        
        Returns
        -------
        result_dict : dict
            Dictionary with optimization results, including the following keys:\n
                'x': 3D consensus point (numpy array shape (3,))\n
                'distances': array with distances from the consensus point to each plane\n
                'distance_std': standard deviation of distances\n
                'distance_mean': mean of distances\n
                'within_tolerance': number of planes within ±tol (tol=1.5 mm)\n
                'success': whether the optimization converged\n
                'fun': final value of the target function\n
                'stats': dictionary with additional statistics
        '''
        # We extract the parameters from kwargs
        target_distance = kwargs.get('target_distance', 10.0)
        verbose = kwargs.get('verbose', False)

        # We extract the valid planes
        valid_planes = []
        valid_indices = []
        
        for idx, plane_dict in sorted(planes_by_fix.items()):
            if not plane_dict:
                continue
            
            # We extract the plane according to the method
            if isinstance(plane_dict, dict) and plane_method in plane_dict:
                plane = plane_dict[plane_method]
            else:
                plane = plane_dict
            
            if plane is not None:
                valid_planes.append(plane)
                valid_indices.append(idx)
        
        if not valid_planes:
            raise ValueError("No valid planes found")
        
        n_planes = len(valid_planes)
        
        # Function to calculate the distance from a point to each plane
        def distances_to_planes(point):
            """Returns array of signed distances to each plane"""
            point = np.asarray(point, dtype=float)
            dists = np.zeros(n_planes)
            for i, plane in enumerate(valid_planes):
                normal = np.asarray(plane['normal'], dtype=float)
                plane_point = np.asarray(plane['point'], dtype=float)
                d = plane['d']
                # Distance with sign: (point - plane_point) . normal + d
                dists[i] = np.dot(point - plane_point, normal) + d
            return dists
        
        # Target function: minimize variance of distances + penalty for mean distance deviating from target_distance
        def objective(point):
            dists = np.abs(distances_to_planes(point))  # absolute distances
            
            # Penalties:
            # 1. Minimize variance of distances (equidistance)
            variance = np.var(dists)
            
            # 2. Penalize deviation from target distance
            mean_dist = np.mean(dists)
            target_penalty = (mean_dist - target_distance) ** 2
            
            # 3. Penalize outliers (distances much farther than target_distance)
            outlier_threshold = target_distance + 5.0  # 5 mm tolerance for outliers
            outliers = np.sum((dists > outlier_threshold) * (dists - outlier_threshold) ** 2)
            
            # Weighted target function
            obj = variance + 0.5 * target_penalty + 0.5 * outliers
            
            return obj
        
        # Initial guess: centroid of all points from the planes
        if initial_guess is None:
            all_points = []
            for plane in valid_planes:
                if 'pts_used' in plane and plane['pts_used'] is not None:
                    all_points.append(plane['pts_used'])
                else:
                    # If there are no pts_used, we can use the point on the plane as a proxy for the centroid of the points that defined it
                    all_points.append(np.asarray(plane['point']).reshape(1, 3))
            all_points_arr = np.vstack(all_points)
            initial_guess = all_points_arr.mean(axis=0)
        else:
            initial_guess = np.asarray(initial_guess, dtype=float)
        
        if verbose:
            print(f"Number of planes: {n_planes}")
            print(f"Initial guess: {initial_guess}")
            print(f"Initial target value: {objective(initial_guess):.6f}")
        
        # Optimize
        result = minimize(objective, initial_guess, method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        # Extract results
        consensus_point = result.x
        dists_final = np.abs(distances_to_planes(consensus_point))
        distance_std = np.std(dists_final)
        distance_mean = np.mean(dists_final)
        
        # Count how many planes are within tolerance (±1.5 mm from the target distance)
        tolerance = 1.5
        within_tolerance = np.sum(np.abs(dists_final - target_distance) <= tolerance)
        
        if verbose:
            print(f"\n✓ Optimization converged: {result.success}")
            print(f"✓ Consensus point: {consensus_point}")
            print(f"✓ Mean distance: {distance_mean:.4f} mm")
            print(f"✓ Standard deviation: {distance_std:.4f} mm")
            print(f"✓ Planes within tolerance (±{tolerance}mm): {within_tolerance}/{n_planes}")
            print(f"\nDistances per fixation:")
            for idx, dist in zip(valid_indices, dists_final):
                status = "✓" if abs(dist - target_distance) <= tolerance else "✗"
                print(f"  Fix {idx}: {dist:.4f} mm {status}")
        
        result_dict = {
            'x': consensus_point,
            'distances': dists_final,
            'distance_std': distance_std,
            'distance_mean': distance_mean,
            'within_tolerance': within_tolerance,
            'within_mask': np.abs(dists_final - target_distance) <= tolerance,
            'success': result.success,
            'fun': result.fun,
            'valid_indices': valid_indices,
            'stats': {
                'n_planes': n_planes,
                'tolerance': tolerance,
                'target_distance': target_distance
            }
        }
        return result_dict
    
    def stat_estimate_r_retina(self, lor: str = 'left', hist_method: str = 'simple', save_estimation: bool = True, **kwargs) -> float:
        '''
        This function estimates the radius of the retina from the hitting points obtained by the sensors

        Parameters
        ----------
        lor : str, optional
            Indicates whether the eye to estimate is the left or right ('left' or 'right'). Default is 'left'
        hist_method : str, optional
            Method to use for averaging the points ('simple', 'all_points', 'avg'). 'simple' uses only the last measurement,
            'all_points' uses all points from all measurements without averaging per sensor, and 'avg' averages the points per sensor.
            Default is 'simple'
        save_estimation : bool, optional
            Indicates whether to save the estimation in the estimations attribute (True) or not (False). Default is True
        **kwargs : dict, optional
            Define the method for the parameter estimation. Valid keys are:\n
            - 'measurements': list[dict[str, Measurement]]: List of measurement objects containing the sensor data. If not provided,
            the last_measurement attribute is used. Only used if hist_method is 'all_points' or 'avg'

        Returns
        -------
        mean_r_retina : float
            Mean radius of the estimated retina
        '''
        p_list_avg = []
        p_list = {}
        
        if hist_method == 'simple':
            # We just use the last measurement
            for sensor_i in self.glasses.sensors_list:
                if sensor_i.lor == lor:
                    # We check if p_list has a key for the sensor ID; if not, we create it
                    if sensor_i.ID not in p_list:
                        p_list[sensor_i.ID] = np.empty((0, 3))
                    # We obtain the intersection point, from the measured distance
                    direction_norm = np.array(sensor_i.direction) / np.linalg.norm(sensor_i.direction)
                    p_i = np.array(sensor_i.origin) + direction_norm * self.last_measurement[sensor_i.ID].distance
                    if self.last_measurement[sensor_i.ID].surface == 'retina':
                        p_list[sensor_i.ID] = np.vstack([p_list[sensor_i.ID], p_i])
        else:
            # We use all the measurements
            measurements = kwargs.get('measurements', [self.last_measurement])
            for i in range(len(measurements)):
                for sensor_i in self.glasses.sensors_list:
                    if sensor_i.lor == lor:
                        # We check if p_list has a key for the sensor ID; if not, we create it
                        if sensor_i.ID not in p_list:
                            p_list[sensor_i.ID] = np.empty((0, 3))
                        # We obtain the intersection point, from the measured distance
                        direction_norm = np.array(sensor_i.direction) / np.linalg.norm(sensor_i.direction)
                        p_i = np.array(sensor_i.origin) + direction_norm * measurements[i][sensor_i.ID].distance
                        if measurements[i][sensor_i.ID].surface == 'retina':
                            p_list[sensor_i.ID] = np.vstack([p_list[sensor_i.ID], p_i])
        
        # We average the points
        if hist_method == 'all_points':
            # We use all the points from all sensors, without averaging per sensor
            for key in p_list:
                for point in p_list[key]:
                    p_list_avg.append(point)
            p_list_avg = np.array(p_list_avg)
        else:
            # We average the points per sensor
            for key in p_list:
                p_list_avg.append(np.mean(p_list[key], axis=0))
            p_list_avg = np.array(p_list_avg)
        
        # We estimate the retina radius
        mean_r_retina = np.mean(np.linalg.norm(p_list_avg-np.array(self.estimations.eyes[lor].location), axis=1))

        if save_estimation:
            self.estimations.eyes[lor].r_retina = mean_r_retina
            self.estimations.eyes[lor].update_d_iris()
                
        return mean_r_retina
    
    def stat_estimate_d_iris(self, lor: str ='left', hist_method: str = 'simple', save_estimation: bool = True, **kwargs) -> float:
        '''
        This function estimates the distance from the center of the eye to the iris plane from the hitting points obtained by the sensors
        
        Parameters
        ----------
        lor : str, optional
            Indicates whether the eye to estimate is the left or right ('left' or 'right'). Default is 'left'
        hist_method : str, optional
            Method to use for averaging the points ('simple', 'all_points', 'avg'). 'simple' uses only the last measurement,
            'all_points' uses all points from all measurements without averaging per sensor, and 'avg' averages the points per sensor.
            Default is 'simple'
        save_estimation : bool, optional
            Indicates whether to save the estimation in the estimations attribute (True) or not (False). Default is True
        **kwargs : dict, optional
            Define the method for the parameter estimation. Valid keys are:\n
            - 'measurements': list[dict[str, Measurement]]: List of measurement objects containing the sensor data. If not provided,
            the last_measurement attribute is used. Only used if hist_method is 'all_points' or 'avg'
            - 'ideal_center_Q': bool: Indicates whether the calculations should be made with the nominal eye parameters (True) or with
            the estimated ones (False). Default is False

        Returns
        -------
        mean_d_iris : float
            Mean distance from the center of the eye to the iris plane
        '''
        # We extract the ideal_center_Q parameter from kwargs, defaulting to False if not provided
        ideal_center_Q = kwargs.get('ideal_center_Q', False)

        # We extract the distances to the iris plane from the different measurements
        distances_vector = np.empty(((0,1)))
        if hist_method == 'simple':
            # We just use the last measurement
            n_plane_left, d_plane_left, n_plane_right, d_plane_right  = self.estimate_iris_planes(ideal_center_Q=ideal_center_Q)
            if lor == 'left':
                distances_vector = np.append(distances_vector, d_plane_left)
            else:
                distances_vector = np.append(distances_vector, d_plane_right)
        else:
            # We use all the measurements
            measurements = kwargs.get('measurements', [self.last_measurement])
            for i in range(len(measurements)):
                n_plane_left, d_plane_left, n_plane_right, d_plane_right  = self.estimate_iris_planes(ideal_center_Q=ideal_center_Q, custom_measurement=measurements[i])
                if lor == 'left':
                    distances_vector = np.append(distances_vector, d_plane_left)
                else:
                    distances_vector = np.append(distances_vector, d_plane_right)

        # We average the distances
        mean_d_iris = np.mean(distances_vector)
        mean_r_iris = np.sqrt(self.estimations.eyes[lor].r_sclera**2 - mean_d_iris**2)

        if save_estimation:
            self.estimations.eyes[lor].r_iris = mean_r_iris
            self.estimations.eyes[lor].update_d_iris()
        return mean_d_iris
    
    def estimate_fixation_point_from_vel_data(self, sensor_list: list[int], ideal_eye_Q: bool = False) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
        '''
        This function obtains the data from the selected sensors

        Parameters
        ----------
        sensor_list : list[int]
            List of indices of the selected sensors
        ideal_eye_Q : bool, optional
            Indicates whether the calculations should be made with the nominal eye parameters (True) or with the estimated ones (False).
            Default is True

        Returns
        -------
        v_list : list[float]
            List of velocities detected by the sensors
        x_list : list[np.ndarray]
            List of impact points referenced to the center of the eye
        p_list : list[np.ndarray]
            List of sensor directions
        '''
        # We obtain the velocities of the sensors, the hitting points and the direction of each sensor
        v_list = []
        x_list = []
        p_list = []

        for i in sensor_list:
            sensor_i = self.glasses.sensors_list[i]
            # We obtain the hitting point of the sensor in the new coordinate system
            direction_norm = np.array(sensor_i.direction) / np.linalg.norm(sensor_i.direction)
            x_i = np.array(sensor_i.origin) + direction_norm * self.last_measurement[sensor_i.ID].distance
            # We correct the hitting point of the sensor, centering the coordinate system on the corresponding eye
            if (ideal_eye_Q):
                x_i = x_i - np.array(self.ground_truth.eyes[sensor_i.lor].location)
            else:
                x_i = x_i - np.array(self.estimations.eyes[sensor_i.lor].location)
            
            # We add the values to the corresponding vectors
            v_list.append(self.last_measurement[sensor_i.ID].velocity)
            x_list.append(x_i)
            p_list.append(direction_norm)
        
        return v_list, x_list, p_list
    
    def estimate_fixation_point_from_vel_calc_omega(self, v_list: list[float], x_list: list[np.ndarray], p_list: list[np.ndarray]) -> np.ndarray:
        '''
        This function performs the necessary calculations to estimate the direction of gaze at the current instant

        Parameters
        ----------
        v_list : list[float]
            List of velocities detected by the sensors
        x_list : list[np.ndarray]
            List of impact points referenced to the center of the eye
        p_list : list[np.ndarray]
            List of sensor directions

        Returns
        -------
        vel_angular : np.ndarray
            Estimated angular velocity vector [rad/time unit]
        '''

        # We obtain the C matrix
        C = np.empty((0,3))
        for i in range(3):
            C = np.vstack([C, np.cross(x_list[i], p_list[i])])
        C = np.transpose(C)

        # We obtain the angular velocity
        vel_angular = np.dot(np.array(v_list), np.linalg.inv(C))

        return vel_angular
    
    def estimate_fixation_point_from_vel_simple(self, n_plane_prev: np.ndarray, sensor_list: list[int], ideal_eye_Q: bool = False) -> tuple[np.ndarray, bool]:
        '''
        This function estimates the fixation point of a single eye from the velocities detected by the indicated sensors

        Parameters
        ----------
        n_plane_prev : np.ndarray
            Direction of gaze at the previous instant
        sensor_list : list[int]
            List of indices of the selected sensors
        ideal_eye_Q : bool, optional
            Indicates whether the calculations should be made with the nominal eye parameters (True) or with the estimated ones (False).
            Default is True

        Returns
        -------
        n_plane_new : np.ndarray
            Direction of gaze at the current instant
        correct_estimation : bool
            Indicates whether the estimation has been performed correctly
        '''
        # We check that there are enough sensors
        correct_estimation = True
        if (len(sensor_list) < 3):
            correct_estimation = False
            return n_plane_prev, correct_estimation

        v_list, x_list, p_list = self.estimate_fixation_point_from_vel_data(sensor_list, ideal_eye_Q=ideal_eye_Q)

        vel_angular = self.estimate_fixation_point_from_vel_calc_omega(v_list, x_list, p_list)

        # We obtain the new direction of gaze
        # We calculate the rotation matrix from the calculated angular velocity and apply it
        r = R.from_rotvec(vel_angular/self.glasses.freq)
        n_plane_new = r.apply(n_plane_prev)

        return n_plane_new, correct_estimation
    
    def stat_estimate_optical_axis_dynamic(self, lor: str, **kwargs) -> tuple[np.ndarray, bool]:
        '''
        This function estimates the fixation point of a single eye from the velocities detected by the sensors and the direction
        of gaze at the previous instant

        Parameters
        ----------
        lor : str
            Indicates whether the eye to estimate is the left or right ('left' or 'right')
        **kwargs : dict, optional
            Define the method for the parameter estimation. Valid keys are:\n
            - 'ideal_eye_Q': bool: Indicates whether the calculations should be made with the nominal eye parameters (True) or with
            the estimated ones (False). Default is False

        Returns
        -------
        n_plane_new : np.ndarray
            Direction of gaze at the current instant
        correct_estimation : bool
            Indicates whether the estimation has been performed correctly
        '''
        # We extract the ideal_eye_Q parameter from kwargs, defaulting to False if not provided
        ideal_eye_Q = kwargs.get('ideal_eye_Q', False)

        # We obtain the indices of the sensors:
        indices = []
        for i in range(len(self.glasses.sensors_list)):
            sensor_i = self.glasses.sensors_list[i]
            if self.last_measurement[sensor_i.ID].surface == 'sclera' and sensor_i.lor == lor:
                indices.append(i)

        # We obtain the previous estimation of the direction of gaze
        n_plane_prev = self.approx_gaze_trajectory.optical_axis_list[lor][-1]

        # We check the number of sensors on each side
        sufficient_number = True
        if len(indices) < 3:
            sufficient_number = False
            return n_plane_prev, sufficient_number
        elif len(indices) == 3:
            n_plane_new, correct_estimation = self.estimate_fixation_point_from_vel_simple(n_plane_prev, indices, ideal_eye_Q)
            self.estimations.eyes[lor].optical_axis = n_plane_new
            return n_plane_new, correct_estimation
        else:
            all_subsets = self.obtain_subsets(indices)

            estimations = []
            for subset in all_subsets:
                # We make the estimation for each sensor subset
                n_plane_new, correct_estimation = self.estimate_fixation_point_from_vel_simple(n_plane_prev, subset, ideal_eye_Q)
                # If there is no error, it is added to the list of estimations
                if (correct_estimation):
                    estimations.append(n_plane_new)
            # The mean of the estimations for the different sensor subsets is computed:
            estimation_mean = np.average(estimations, axis=0)

            self.estimations.eyes[lor].optical_axis = estimation_mean

            return estimation_mean, correct_estimation
    
    def estimate_angular_vel(self, lor: str, **kwargs) -> np.ndarray:
        '''
        This function performs an estimation of the angular velocity of the eye based on the measurements from the sensors

        Parameters
        ----------
        lor : string
            Indicates whether the eye to estimate is the left or right ('left' or 'right')
        **kwargs : dict, optional
            Define the method for the parameter estimation. Valid keys are:\n
            - 'ideal_eye_Q': bool: Indicates whether the calculations should be made with the nominal eye parameters (True) or with
            the estimated ones (False). Default is False
            - 'normalize_Q': bool: Indicates whether the resulting angular velocity vector should be normalized (True) or not (False).
            Default is False

        Returns
        -------
        vel_angular : np.ndarray
            The normalized estimated angular velocity vector
        '''
        # We extract the parameters from kwargs, defaulting to False if not provided
        ideal_eye_Q = kwargs.get('ideal_eye_Q', False)
        normalize_Q = kwargs.get('normalize_Q', False)

        # We obtain the indices of the sensors that hit the sclera of the selected eye
        indices = []
        for i in range(len(self.glasses.sensors_list)):
            sensor_i = self.glasses.sensors_list[i]
            if self.last_measurement[sensor_i.ID].surface == 'sclera' and sensor_i.lor == lor:
                indices.append(i)

        # We calculate the rotational velocity depending on the number of available sensors
        if len(indices) == 3:
            v_list, x_list, p_list = self.estimate_fixation_point_from_vel_data(indices, ideal_eye_Q)
            vel_angular = self.estimate_fixation_point_from_vel_calc_omega(v_list, x_list, p_list)
            if normalize_Q:
                vel_angular = vel_angular / np.linalg.norm(vel_angular)
            return vel_angular
        elif len(indices) > 3:
            estimations = []
            all_subsets = self.obtain_subsets(indices)
            for subset in all_subsets:
                v_list, x_list, p_list = self.estimate_fixation_point_from_vel_data(subset, ideal_eye_Q)
                vel_angular = self.estimate_fixation_point_from_vel_calc_omega(v_list, x_list, p_list)
                if normalize_Q:
                    vel_angular = vel_angular / np.linalg.norm(vel_angular)
                # If there is no error, it is added to the list of estimations
                estimations.append(vel_angular)

            # The mean of the estimations for the different sensor subsets is computed:
            vel_angular = np.average(estimations, axis=0)
            if normalize_Q:
                vel_angular = vel_angular / np.linalg.norm(vel_angular)
            return vel_angular

    def stat_estimate_cornea_center(self, lor: str = 'left', initial_guess: list[float] = None, **kwargs) -> np.ndarray:
        '''
        This function estimates the center of the cornea from a minimization process of the cost function defined inside.
        It is based on the estimation of the optical distance from the sensors to the surface of the eye, comparing with the measurements
        obtained

        Parameters
        ----------
        lor : str, optional
            Indicates whether the eye to estimate is the left or right ('left' or 'right')
        initial_guess : list[float], optional
            Initial point for the estimation of the cornea center. If not specified, the center of the eye is taken as the initial point.
            Default is None
        **kwargs : dict, optional
            Define the method for the parameter estimation. Valid keys are:\n
            - 'invert_z_axis': bool: Indicates whether the desired cornea center should have a lower (True) or higher (False) z component than
            the eyeball center. Default is False
            Default is False
            - 'only_iris': bool: Indicates whether only the sensors that hit the iris (True) or all sensors (False) should be used for the
            estimation. Default is False
            - 'verbose': int: Level of verbosity for the optimization process. 0 for no output, 1 for final results, 2 for detailed
            optimization information. Default is 0

        Returns
        -------
        estimated_cornea_center : np.ndarray
            Estimation of the cornea center
        '''
        # We extract the parameters from kwargs, defaulting to False if not provided
        invert_z_axis = kwargs.get('invert_z_axis', False)
        only_iris = kwargs.get('only_iris', False)
        verbose = kwargs.get('verbose', 0)

        def cost_function_estimation_cornea_center_lm(theta_phi_array: np.ndarray, lor: str = 'left') -> np.ndarray:
            '''
            This function calculates the value of the cost function for the estimation of the cornea center

            Parameters
            ----------
            theta_phi_array : np.ndarray
                Array with the theta and phi angles of the candidate for the cornea center
            lor : str, optional
                Indicates whether the studied eye is the left or right. Default is 'left'

            Returns
            -------
            residual_vector : np.ndarray
                Vector of residuals between the measured and estimated optical distances
            '''
            # We start from a cornea center candidate. From there, we define an eye surface and calculate the intersection of the 
            # rays from the sensors with the eye surface. We record the distances. We define the opl_candidate_vector
            na = 1.0 # Refractive index of air 
            nc = self.estimations.eyes[lor].n_cornea # Refractive index of cornea

            opl_meas_vector = np.array([])
            candidate_cornea_center = np.array([np.cos(theta_phi_array[1]) * np.sin(theta_phi_array[0]),
                                                 np.sin(theta_phi_array[1]), 
                                                 np.cos(theta_phi_array[1]) * np.cos(theta_phi_array[0])]) * self.estimations.eyes[lor].h_cornea + self.estimations.eyes[lor].location
            opl_candidate_vector = np.array([])
            for sensor_i in self.glasses.sensors_list:
                if sensor_i.lor == lor:
                    if (only_iris and self.last_measurement[sensor_i.ID].surface != 'iris') or self.last_measurement[sensor_i.ID].distance == 'none':
                        continue
                    # Construct the measurement vector
                    if self.last_measurement[sensor_i.ID].distance:
                        opl_meas_vector = np.append(opl_meas_vector, self.last_measurement[sensor_i.ID].distance)
                    else:
                        opl_meas_vector = np.append(opl_meas_vector, 0)
                    # Transform the sensor variables to numpy arrays
                    origin = np.array(sensor_i.origin)
                    direction_norm = np.array(sensor_i.direction) / np.linalg.norm(np.array(sensor_i.direction))
                    # Create the sphere and plane that represent the eye
                    this_eye_estimation = self.estimations.eyes[sensor_i.lor]
                    sclera_sphere = Sphere(this_eye_estimation.location, this_eye_estimation.r_sclera) 
                    retina_sphere = Sphere(this_eye_estimation.location, this_eye_estimation.r_retina)
                    cornea_sphere = Sphere(candidate_cornea_center, this_eye_estimation.r_cornea)
                    candidate_optical_axis = np.array((candidate_cornea_center - np.array(this_eye_estimation.location))/np.linalg.norm(candidate_cornea_center - np.array(this_eye_estimation.location)))
                    iris_plane = Plane(np.array(this_eye_estimation.location) + candidate_optical_axis * this_eye_estimation.d_iris, candidate_optical_axis)
                    optical_line = Line(np.array(this_eye_estimation.location), candidate_optical_axis)
                    r_iris = this_eye_estimation.r_iris
                    r_pupil = this_eye_estimation.r_pupil

                    # Calculate the intersection of the ray with the eye surface
                    point_a, point_b = sclera_sphere.intersect_line(Line(origin, direction_norm))            
                    # We choose the closest point to the sensor
                    if np.linalg.norm(point_a-origin) < np.linalg.norm(point_b-origin):
                        scleral_point = point_a
                    else:
                        scleral_point = point_b
                    scleral_distance = np.linalg.norm(scleral_point-origin)

                    # Now, we calculate the intersection points with the cornea, if it would be
                    try:
                        point_c, point_d = cornea_sphere.intersect_line(Line(origin, direction_norm))
                        if np.linalg.norm(point_c-origin) < np.linalg.norm(point_d-origin):
                            corneal_point = point_c
                        else:
                            corneal_point = point_d
                        cornea_distance = np.linalg.norm(corneal_point-origin)
                    except:
                        corneal_point = Point([0, 0, 0])
                        cornea_distance = 1000000 # A very big number     

                    if (scleral_distance < cornea_distance):
                        d_i = na*np.linalg.norm(scleral_point - origin)
                    else:
                        # The beam hit on the cornea surface so we have to calculate the refracted ray
                        # We calculate the normal vector of the cornea
                        cornea_normal = (corneal_point - candidate_cornea_center) / np.linalg.norm(corneal_point - candidate_cornea_center) # Outward normal vector
                        # We apply Snell's law to calculate the refracted ray in its vectorial form
                        mu = na / nc
                        incident_cosine = - np.dot(direction_norm, cornea_normal)
                        refracted_direction = mu * direction_norm + (mu * incident_cosine - np.sqrt(1 - mu**2 * (1 - incident_cosine**2))) * cornea_normal
                        
                        iris_point = iris_plane.intersect_line(Line(corneal_point, refracted_direction))
                        d_inter_iris = optical_line.distance_point(iris_point)
                        if r_pupil < d_inter_iris:
                            final_point = iris_point
                        else:
                            # This is only meaningful if the sensor hits on the pupil   
                            point_e, point_f = retina_sphere.intersect_line(Line(corneal_point, refracted_direction))
                            if np.linalg.norm(point_e-origin) > np.linalg.norm(point_f-origin):
                                retinal_point = point_e
                            else:
                                retinal_point = point_f
                            final_point = retinal_point
                            
                        d_i = na*np.linalg.norm(corneal_point-origin) + nc*np.linalg.norm(final_point-corneal_point)
                    opl_candidate_vector = np.append(opl_candidate_vector, d_i)
            residual_vector = opl_candidate_vector - opl_meas_vector
            return residual_vector
    
        if (initial_guess is None):
            p_list = []
            for sensor_i in self.glasses.sensors_list:
                # We obtain the intersection point, from the measured distance
                direction_norm = np.array(sensor_i.direction) / np.linalg.norm(np.array(sensor_i.direction))
                p_i = np.array(sensor_i.origin) + direction_norm * self.last_measurement[sensor_i.ID].distance
                if sensor_i.lor == lor: 
                    if self.last_measurement[sensor_i.ID].surface == 'iris':
                        p_list.append(p_i)
            # We estimate the planes that best fit the points
            n_plane, d_plane = self.estimate_plane(p_list, lor, ideal_center_Q=True, invert_z_axis=invert_z_axis)
            if n_plane is None:
                initial_guess = [0, 0]
                if verbose > 0:
                    print('Impossible to determine iris plane without refraction. Using default values for theta/phi')  # Default initial guess if plane estimation fails
            else:
                theta = np.arctan2(n_plane[0], n_plane[2])
                phi = np.arcsin(n_plane[1]) # Assuming the normal vector is normalized
                if verbose > 0:
                    print('Estimated theta:', 180*theta/np.pi, 'phi:', 180*phi/np.pi)
                initial_guess = [theta, phi]
        res = least_squares(cost_function_estimation_cornea_center_lm, np.array(initial_guess), method='lm', args=[lor],
                            verbose=verbose, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)

        if verbose > 0:
            print('Optimization result:', 180*res['x']/np.pi)
        estimated_cornea_center = np.array([np.cos(res['x'][1]) * np.sin(res['x'][0]),
                                                 np.sin(res['x'][1]), 
                                                 np.cos(res['x'][1]) * np.cos(res['x'][0])]) * self.estimations.eyes[lor].h_cornea + np.array(self.estimations.eyes[lor].location)
        if verbose > 0:
            if res.success:
                print('Cornea center estimated successfully:', estimated_cornea_center)
            else:
                print('Cornea center estimation failed:', res.message)
        return estimated_cornea_center

    def check_coplanarity(self, lor: str = 'left', **kwargs) -> float:
        '''
        This function checks the coplanarity of the intersection points obtained from the sensors.
        It can work with a single measurement (the last one) or with a set of measurements

        Parameters
        ----------
        lor : str, optional
            Indicates whether the eye to use is the left or right ('left' or 'right'). Default is 'left'
        **kwargs : dict, optional
            Define the method for the parameter estimation. Valid keys are:\n
            - 'measurements': list[dict[str, Measurement]]: List of measurement dictionaries to use for the coplanarity check. If not provided,
            the last measurement is used. Default is None

        Returns
        -------
        angle : float
            Angle between the two planes defined by the four intersection points [degrees]
        '''
        # We extract the measurements from kwargs, using the last measurement if not provided
        measurements = kwargs.get('measurements', [self.last_measurement])

        # We obtain the intersection points
        p_list_avg = []
        p_list = {}
        for i in range(len(measurements)):
            for sensor_i in self.glasses.sensors_list:
                # We check if p_list has a key for the sensor ID; if not, we create it
                if sensor_i.ID not in p_list:
                    p_list[sensor_i.ID] = np.empty((0, 3))
                # We obtain the intersection point, from the measured distance
                direction_norm = np.array(sensor_i.direction) / np.linalg.norm(np.array(sensor_i.direction))
                p_i = np.array(sensor_i.origin) + direction_norm * measurements[i][sensor_i.ID].distance
                if sensor_i.lor == lor and measurements[i][sensor_i.ID].surface == 'sclera':
                    p_list[sensor_i.ID] = np.vstack([p_list[sensor_i.ID], p_i])

        # We average the points for each sensor
        for key in p_list:
            p_list_avg.append(np.mean(p_list[key], axis=0))
        points = np.array(p_list_avg)

        # We calculate the angle between the two planes defined by the four points
        N1 = np.cross(points[1] - points[0], points[2] - points[0])
        N2 = np.cross(points[3] - points[0], points[1] - points[0])
        angle = np.arccos(np.clip(np.dot(N1, N2) / (np.linalg.norm(N1) * np.linalg.norm(N2)), -1.0, 1.0))
        angle = angle * 180 / np.pi
        return angle
    
    def check_slippage(self) -> bool:
        '''
        This function checks whether there is slippage by estimating the eyeball center and comparing it with previous estimations.

        Parameters
        ----------
        None

        Returns
        -------
        slippage_detected : bool
            Indicates whether slippage is detected (True) or not (False)
        '''
        # We estimate the center of the eye using the last measurement
        center_left, radius_left = self.stat_estimate_eye_centers(lor='left', save_estimation=False)
        center_right, radius_right = self.stat_estimate_eye_centers(lor='right', save_estimation=False)

        # We compare the estimated centers with the previous ones
        if np.linalg.norm(center_left - np.array(self.estimations.eyes['left'].location)) + np.linalg.norm(center_right - np.array(self.estimations.eyes['right'].location)) > 1.0:
            slippage_detected = True
        else:
            slippage_detected = False
        return slippage_detected

    def show(self, lor: list[str] = ['left','right'], **kwargs) -> pv.Plotter:
        '''
        This function shows the system in a 3D plot using PyVista. Output can be displayed via its own show method

        Parameters
        ----------
        lor : list[str], optional
            Sides of the eyes to show (e.g., ['left', 'right']). Default is ['left','right']
        **kwargs : dict, optional
            Further define the behavior of the function. Valid keys are:\n
            - 'off_screen': bool: Indicates if the rendering is off-screen (True) or interactive (False). Default is False
            - 'add_plots': list[str]: List of additional plots to be shown. Possible values are 'visual_axis', 'velocity_plane' and
            'reference_systems'. Default is []

        Returns
        -------
        pl : pyvista.Plotter
            PyVista Plotter object that shows the system
        '''
        # We obtain the parameters from kwargs
        off_screen = kwargs.get('off_screen', False)
        add_plots = kwargs.get('add_plots', [])

        if off_screen:
            pl = pv.Plotter(lighting='none', off_screen=True, window_size=[3840, 2160])
        else:
            pl = pv.Plotter(lighting='none', off_screen=False, window_size=[1920, 1080])
        # Anti-aliasing and EDL help reduce jagged edges and patchiness
        pl.enable_anti_aliasing('ssaa')
        try:
            pl.enable_eye_dome_lighting()
        except Exception:
            # EDL may not be available in older VTK/PyVista versions; safely ignore
            pass

        # Draw the eyes
        for eye_key in lor:
            self.estimations.eyes[eye_key].show(pl, plot_va='visual_axis' in add_plots)

        # Draw the sensors and their paths
        for sensor_i in self.glasses.sensors_list:
            if sensor_i.lor in lor:
                previous_point = np.array(sensor_i.origin)
                if self.last_measurement[sensor_i.ID].surface == 'sclera':
                    sensor_color = 'b'
                elif self.last_measurement[sensor_i.ID].surface == 'iris':
                    sensor_color = 'g'
                else:
                    sensor_color = 'r'

                # for point_i in self.last_measurement[sensor_i.ID].path_points:
                #     pl.add_mesh(pv.Line(previous_point, point_i), color=sensor_color, line_width=5, opacity=1)
                #     previous_point = point_i
                final_point = np.array(sensor_i.origin) + np.array(sensor_i.direction) / np.linalg.norm(np.array(sensor_i.direction)) * self.last_measurement[sensor_i.ID].distance
                pl.add_mesh(pv.Line(previous_point, final_point), color=sensor_color, line_width=5, opacity=1)
        
        # We make the camera point to the left or right eye
        if lor == ['left']:
            cam_pos = -30
        elif lor == ['right']:
            cam_pos = 30
        else:
            cam_pos = 0

        pl.camera_position = [(-72.8311066083788, -55.21310042329851, 82.45184895111981),       # Camera location
                              (-3.9235209929515946, 8.510522292048218, 3.442119797480064),      # Focal point
                              (0.30183130460034047, -0.8533735747946016, -0.4250310640477221)]  # View-up direction

        # Key light
        UFO = pv.Light(position=(cam_pos, 0, 100), focal_point=(cam_pos, 0, 0), color='white')
        UFO.positional = True
        UFO.cone_angle = 80
        UFO.exponent = 4  # softer falloff
        UFO.intensity = 0.9
        pl.add_light(UFO)

        # Fill light to lower harsh contrast and reduce "patchy" look
        fill = pv.Light(position=(cam_pos, -120, 60), focal_point=(cam_pos, 0, 0), color='white')
        fill.positional = True
        fill.cone_angle = 100
        fill.exponent = 2
        fill.intensity = 0.5
        pl.add_light(fill)

        # Disable hard shadows (can create blocky patches depending on GPU/driver)
        # pl.enable_shadows()
        
        # Captures image with off-screen rendering, then shows interactively
        # pl.screenshot('figure.png', window_size=[3840, 2160])
        # pl.show(screenshot='figure.png', window_size=[3840, 2160])
        # pl.show()
        
        if 'reference_systems' in add_plots:
            # We show a coordinate system for reference (head coordinate system)
            # Create axes with larger lines manually
            origin = np.array([0, 0, 0])
            axis_length = 8  # Length of each axis in mm
            arrow_length = 2  # Length of the arrowhead
            arrow_radius = 0.5  # Radius of the arrowhead
            line_width = 10  # Width of the axis lines
            
            # X axis (red)
            x_end = origin + np.array([axis_length, 0, 0])
            pl.add_mesh(pv.Line(origin, x_end), color='r', line_width=line_width)
            arrow_x = pv.Cone(direction=[1, 0, 0], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_x = arrow_x.translate(x_end)
            pl.add_mesh(arrow_x, color='r')
            
            # Y axis (green)
            y_end = origin + np.array([0, axis_length, 0])
            pl.add_mesh(pv.Line(origin, y_end), color='g', line_width=line_width)
            arrow_y = pv.Cone(direction=[0, 1, 0], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_y = arrow_y.translate(y_end)
            pl.add_mesh(arrow_y, color='g')
            
            # Z axis (blue)
            z_end = origin + np.array([0, 0, axis_length])
            pl.add_mesh(pv.Line(origin, z_end), color='b', line_width=line_width)
            arrow_z = pv.Cone(direction=[0, 0, 1], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_z = arrow_z.translate(z_end)
            pl.add_mesh(arrow_z, color='b')

            # We add the glasses' coordinate system
            glasses_origin = self.glasses.location_T
            axis_length = 8  # Same length for consistency
            
            # X axis (red)
            glasses_x_end = glasses_origin + np.array([axis_length, 0, 0])
            pl.add_mesh(pv.Line(glasses_origin, glasses_x_end), color='r', line_width=line_width)
            arrow_gx = pv.Cone(direction=[1, 0, 0], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_gx = arrow_gx.translate(glasses_x_end)
            pl.add_mesh(arrow_gx, color='r')
            
            # Y axis (green)
            glasses_y_end = glasses_origin + np.array([0, axis_length, 0])
            pl.add_mesh(pv.Line(glasses_origin, glasses_y_end), color='g', line_width=line_width)
            arrow_gy = pv.Cone(direction=[0, 1, 0], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_gy = arrow_gy.translate(glasses_y_end)
            pl.add_mesh(arrow_gy, color='g')
            
            # Z axis (blue)
            glasses_z_end = glasses_origin + np.array([0, 0, axis_length])
            pl.add_mesh(pv.Line(glasses_origin, glasses_z_end), color='b', line_width=line_width)
            arrow_gz = pv.Cone(direction=[0, 0, 1], height=arrow_length, radius=arrow_radius, resolution=8)
            arrow_gz = arrow_gz.translate(glasses_z_end)
            pl.add_mesh(arrow_gz, color='b')

        if 'velocity_plane' in add_plots:
            raise ValueError('Velocity plane plot is not available in the demonstrator system')

        return pl
    
    def get_meas_case(self, measurements_dict: dict[str, Measurement] = None) -> tuple[int, int]:
        '''
        This function obtains the number of hits on each type of surface for both eyes

        Parameters
        ----------
        measurements_dict : dict[str, Measurement], optional
            Dictionary with the measurements. If None, the function will use the last measurements stored in the system. Default is None

        Returns
        -------
        left_case : int
            Number of hits per surface in the left eye
        right_case : int
            Number of hits per surface in the right eye
        '''
        # We load the measurements dictionary. If it is None, we use the last measurements stored in the system
        if measurements_dict is None:
            measurements_dict = self.last_measurement

        # We count the number of hits on each surface for both eyes
        counting_left = {'retina':0, 'iris':0, 'sclera':0}
        counting_right = {'retina':0, 'iris':0, 'sclera':0}
        for meas_i in measurements_dict.values():
            if meas_i.lor == 'left': 
                counting_left[meas_i.surface] += 1
            else:
                counting_right[meas_i.surface] += 1
        left_case = 100 * counting_left['retina'] + 10 * counting_left['iris'] + counting_left['sclera']
        right_case = 100 * counting_right['retina'] + 10 * counting_right['iris'] + counting_right['sclera']
        return left_case, right_case

    def set_last_measurement(self, last_measurement: dict[str, Measurement]) -> None:
        '''
        This function set the last_measurement object of the system and updates the case of the measurements

        Parameters
        ----------
        last_measurement : dict[str, Measurement]
            last_measurement-like object to be set in the system

        Returns
        -------
        None
        '''
        # We check that the system is not embedded in an umbrella system, to avoid breaking the link between the demonstrator
        # and the simulator
        if self.embedded_sys:
            raise ValueError('The last_measurement object cannot be set in the demonstrator if the system is embedded')
        else:
            # We update the measurement
            self.last_measurement = last_measurement.copy()

            # We update the case of the measurements
            self.meas_case['left_case'], self.meas_case['right_case'] = self.get_meas_case(last_measurement)
        return

class System_umbrella: # Class that contains both the simulator and the demonstrator and allows interaction between them

    def __init__(self):
        self.simulator = System_simulator()                                                         # Simulator object
        self.demonstrator = System_demonstrator(embedded_sys=True,                                  # Demonstrator object
                                                glasses=self.simulator.glasses,
                                                user=self.simulator.user,
                                                last_measurement=self.simulator.last_measurement)
        return

    def to_dict(self) -> dict[str, dict[str, any]]:
        return {'simulator':self.simulator.to_dict(),
                'demonstrator':self.demonstrator.to_dict()}
    
    def from_dict(self, system_dict: dict[str, dict[str, any]]) -> None:
        if 'simulator' in system_dict:
            self.simulator.from_dict(system_dict['simulator'])
        if 'demonstrator' in system_dict:
            self.demonstrator.from_dict(system_dict['demonstrator'])
        return
    
    def write_configuration(self, config_file: str) -> None:
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return

    def set_configuration(self, config_file: str) -> None:
        self.from_dict(json.load(open(config_file, 'r')))
        return

    def set_last_measurement(self, last_measurement: dict[str, Measurement]) -> None:
        '''
        This function set the last_measurement object of the simulator system and updates the case of the measurements

        Parameters
        ----------
        last_measurement : dict[str, Measurement]
            last_measurement-like object to be set in the system

        Returns
        -------
        None
        '''
        # We update the measurement without breaking a possible link with the demonstrator's last_measurement object
        self.simulator.last_measurement.clear()
        self.simulator.last_measurement.update(last_measurement)

        # We update the case of the measurements
        self.simulator.meas_case['left_case'], self.simulator.meas_case['right_case'] = self.simulator.get_meas_case(last_measurement)
        self.demonstrator.meas_case['left_case'], self.demonstrator.meas_case['right_case'] = self.demonstrator.get_meas_case(last_measurement)
        return

    def record_tertiary_movement(self, point1: list[float], point2: list[float], n_samples: int, mov_type: str, angular_velocity: float, mode: str, ideal_eye_Q: bool = True, obtain_measurements: bool = False, meas_out_cond: tuple[int, str] = (4, 'sclera')) -> tuple[np.ndarray, np.ndarray, dict[str, Measurement]]:
        '''
        This function simulates the eye movement between two tertiary positions given by the points point1 and point2

        Parameters
        ----------
        point1 : list[float]
            Initial point of the trajectory
        point2 : list[float]
            Second point of the trajectory
        n_samples : int
            Number of samples to take during the movement. It is taken into account in the 'n_samples' version
        mov_type : str
            Angular velocity model ('constant')
        angular_velocity : float
            Angular velocity of the movement [rad/s]. It is taken into account in the 'angular_velocity' version
        mode : str
            Indicates the version of the iterable to use ('angular_velocity', 'n_samples'). Default is 'angular_velocity'

            In angular_velocity, eye rotates for n_samples instants at a constant angular velocity equal to that indicated by angular_velocity

            In n_samples, eye rotates a constant angle at each instant, so that it looks at second_point at the end of the movement
        ideal_eye_Q : bool, optional
            Indicates whether the calculations should be made with the nominal eye parameters (True) or with the estimated ones (False).
            By default it is True
        obtain_measurements : bool, optional
            Indicates whether to obtain the measurements of an instant where 4 sensors hit on the sclera. By default it is False
        meas_out_cond : tuple[int, str], optional
            Condition to consider for obtaining the output measurement. It is a tuple where the first element is the minimum number of sensors
            and the second element is the surface name. By default it is (4, 'sclera')

        Returns
        -------
        estimation_out_total_left : np.ndarray
            Estimated angular velocity of the left eye during the movement
        estimation_out_total_right : np.ndarray
            Estimated angular velocity of the right eye during the movement
        measurement_out : dict[str, Measurement]
            Measurements of an instant where 4 sensors hit on the sclera obtained during the movement (only if obtain_measurements is True).
            If obtain_measurements is False, this output is returned as an empty dictionary
        '''
        # We simulate the movement and obtain the measurements at each instant
        measurement_array, measurement_out = self.simulator.simulate_tertiary_movement(point1, point2, n_samples, mov_type, angular_velocity, mode, obtain_measurements, meas_out_cond=meas_out_cond)
        
        # We save the last measurement to restore it later
        last_measurement_prev = self.simulator.last_measurement
        # We estimate the angular velocity at each instant
        estimations_out_left = []
        estimations_out_right = []
        for measurement in measurement_array:
            # We set the measurement at this instant
            self.simulator.last_measurement = measurement
            # We estimate the angular velocity for each eye
            estimation_angular_vel = self.demonstrator.estimate_angular_vel('left', ideal_eye_Q)
            estimations_out_left.append(estimation_angular_vel)
            estimation_angular_vel = self.demonstrator.estimate_angular_vel('right', ideal_eye_Q)
            estimations_out_right.append(estimation_angular_vel)

        # We sum the estimations at each instant
        estimation_out_total_left = np.sum(estimations_out_left, axis=0)
        estimation_out_total_right = np.sum(estimations_out_right, axis=0)

        # We restore the last measurement as it was before the calculations
        self.simulator.last_measurement = last_measurement_prev

        return estimation_out_total_left, estimation_out_total_right, measurement_out

    