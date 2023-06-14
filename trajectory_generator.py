import numpy as np
import shapely.geometry
import math
from scipy.special import binom

class TrajectoryGenerator:
    def __init__(
            self,
            max_speed: float = 17.0,
            min_traj_len: float = None
        ) -> None:
        self.max_speed = max_speed
        
        self.min_traj_len = 0 if min_traj_len is None else min_traj_len

    def generate_trajectory(self, trajectory_type: str, event_duration: float, source_height: float):
        if trajectory_type == 'rectilinear':
            line, speed = self._generate_rectilinear_trajectory(event_duration, source_height)
        elif trajectory_type == 'bezier':
            line, speed = self._generate_bezier_trajectory(event_duration, source_height)
        else:
            raise ValueError('Trajectory type not supported')
        return line, speed

    def _generate_rectilinear_trajectory(self, sim_time, src_height):
        # Prevent trajectories to get in the area occupied by the car
        forbid_area = shapely.geometry.Polygon([[-1.5, -3], [-1.5, 3], [1.5, 3], [1.5, -3]])

        # Define maximum length
        max_traj_len = self.max_speed * sim_time

        while(True):
            # Define start point
            r = np.random.uniform(3, 100)
            theta = 2 * np.pi * np.random.rand()
            start = [r * np.cos(theta), r * np.sin(theta)]

            # Define end point
            r = np.random.uniform(3, 100)
            theta = 2 * np.pi * np.random.rand()
            end = [r * np.cos(theta), r * np.sin(theta)]

            trajectory = shapely.geometry.LineString([shapely.geometry.Point(start), shapely.geometry.Point(end)])

            if not trajectory.intersects(forbid_area):
                # Check minimum length
                if trajectory.length >= self.min_traj_len and trajectory.length <= max_traj_len:
                    start.append(src_height)
                    end.append(src_height)
                    line = np.stack((np.array(start), np.array(end)), axis=0)
                    break
        speed = trajectory.length / sim_time
        return line, speed

    def _generate_bezier_trajectory(self, sim_time, src_height):
        # Prevent trajectories to get in the area occupied by the car
        forbid_area = shapely.geometry.Polygon([[-1.5, -3], [-1.5, 3], [1.5, 3], [1.5, -3]])

        # Define maximum length
        max_traj_len = self.max_speed * sim_time

        while(True):
            # Define start point
            r = np.random.uniform(3, 100)
            theta = 2 * np.pi * np.random.rand()
            start = [r * np.cos(theta), r * np.sin(theta)]

            # Define end point
            r = np.random.uniform(3, 100)
            theta = 2 * np.pi * np.random.rand()
            end = [r * np.cos(theta), r * np.sin(theta)]
            angle_rad = math.atan2(end[1] - start[1], end[0] - start[0])
            
            # Define start and end angles
            angle1 = np.random.uniform(angle_rad - np.pi/2, angle_rad + np.pi/2, size=(1,1))
            angle2 = np.random.uniform(angle_rad + np.pi / 2, angle_rad - np.pi/2, size=(1,1))

            angle = np.concatenate((angle1, angle2))
            xpoints = [[start[0]], [end[0]]]
            ypoints = [[start[1]], [end[1]]]

            points = np.column_stack((xpoints, ypoints, angle))

            _, c2 = self._get_curve(points, method="prop", r=0.5)

            multiline = []
            for j in range(1, len(c2)):
                multiline.append((c2[j-1], c2[j]))
            trajectory = shapely.geometry.MultiLineString(multiline)

            if not trajectory.intersects(forbid_area):
                # Check minimum length
                if trajectory.length >= self.min_traj_len and trajectory.length <= max_traj_len:
                    line = np.c_[c2, src_height * np.ones(100)]
                    break

        speed = trajectory.length / sim_time
        return line, speed
    
    def _get_curve(self, points, **kw):
        segments = []
        for i in range(len(points)-1):
            seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
            segments.append(seg)
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve

# Bezier curve generation derived from: https://stackoverflow.com/questions/45600246/how-to-connect-points-taking-into-consideration-position-and-orientation-of-each
class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        method = kw.get("method", "const")
        if method=="const":
            self.r = kw.get("r", 1.)
        else:
            r = kw.get("r", 0.3)
            d = np.sqrt(np.sum((self.p2-self.p1)**2))
            self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

        
    def bernstein(self, n,k,t):
        return binom(n,k)* t**k * (1.-t)**(n-k)

    
    def bezier(self, points, num=200):
        N = len(points)
        t = np.linspace(0, 1, num=num)
        curve = np.zeros((num, 2))
        for i in range(N):
            curve += np.outer(self.bernstein(N - 1, i, t), points[i])
        return curve
    
    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = self.bezier(self.p,self.numpoints)
