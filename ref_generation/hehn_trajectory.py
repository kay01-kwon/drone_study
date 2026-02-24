"""
Real-Time Trajectory Generation for Quadrocopters
M. Hehn, R. D'Andrea, IEEE TRO, 31(4), pp. 877-892, 2015.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable


@dataclass
class QuadParams:
    a_max: float = 20.0
    a_min: float = 1.0
    omega_xy_max: float = 10.0
    g: float = 9.81

@dataclass
class DecouplingParams:
    alpha_x: float = 0.5
    alpha_z: float = 0.5
    z_ddot_min: float = -5.0
    a0: float = 9.81

def compute_axis_constraints(qp, dp):
    g = qp.g
    z_lo = max(dp.z_ddot_min, qp.a_min - g)
    z_hi = dp.alpha_z * (qp.a_max - g)
    h_sq = max(qp.a_max**2 - (z_hi + g)**2, 0.0)
    x_max = dp.alpha_x * np.sqrt(h_sq)
    y_max = np.sqrt(max(h_sq - x_max**2, 0.0))
    jmax = (z_lo + g) / np.sqrt(3.0) * qp.omega_xy_max
    return x_max, y_max, z_lo, z_hi, jmax

def _integ(segs, w0, v0, a0):
    w, v, a = w0, v0, a0
    for dt, jk in segs:
        w += v*dt + 0.5*a*dt**2 + (1.0/6.0)*jk*dt**3
        v += a*dt + 0.5*jk*dt**2
        a += jk*dt
    return w, v, a


class AxisTrajectory:
    def __init__(self, w0, v0, a0, a_lo, a_hi, j_max):
        self.w0, self.v0, self.a0 = float(w0), float(v0), float(a0)
        self.a_lo, self.a_hi = float(a_lo), float(a_hi)
        self.j = abs(float(j_max))
        self._segs = []
        self.tf = 0.0
        self._solved = False

    def solve(self, tol=1e-4, tf_max=20.0):
        """Solve for minimum-time trajectory. Relaxed tolerance for real-time."""
        if abs(self.w0)<tol and abs(self.v0)<tol and abs(self.a0)<tol:
            self.tf=0; self._solved=True; return 0.0
        if self.j<1e-12:
            self.tf=tf_max; self._solved=True; return self.tf

        results = []
        for gen, tf_min in self._generators():
            r = self._bisect(gen, tf_min, tol, tf_max)
            if r is not None:
                results.append(r)
                # Early exit if good enough
                if r[0] < tf_min * 1.5:
                    break

        if results:
            self.tf, self._segs = min(results, key=lambda x: x[0])
        else:
            self.tf, self._segs = tf_max, [(tf_max, 0.0)]
        self._solved = True
        return self.tf

    def _bisect(self, gen, tf_min, tol, tf_max):
        lo = max(tf_min, 0.001)
        hi = min(tf_max, lo + 10.0)  # Limit search range
        if lo >= hi: return None
        wlo, slo = gen(lo)
        whi, shi = gen(hi)
        if wlo is None: return None
        if whi is None: return None

        if abs(wlo) < tol:
            return lo, slo
        if abs(whi) < tol:
            return hi, shi

        if wlo * whi > 0:
            # Expand search range
            for m in [2, 4]:
                hi2 = hi * m
                if hi2 > tf_max: hi2 = tf_max
                w2, s2 = gen(hi2)
                if w2 is not None and wlo * w2 <= 0:
                    hi, whi, shi = hi2, w2, s2
                    break
            else:
                return None

        # Reduced iterations for real-time (30 is enough for 1e-4 tolerance)
        for _ in range(30):
            mid = 0.5*(lo+hi)
            wm, sm = gen(mid)
            if wm is None:
                lo = mid; continue
            if abs(wm) < tol or (hi-lo) < tol*0.1:
                return mid, sm
            if wm * wlo > 0:
                lo, wlo = mid, wm
            else:
                hi, whi = mid, wm
        mid = 0.5*(lo+hi)
        _, sm = gen(mid)
        return mid, sm

    def evaluate(self, t):
        if not self._solved:
            return self.w0, self.v0, self.a0, 0.0
        if t <= 0:
            return self.w0, self.v0, self.a0, (self._segs[0][1] if self._segs else 0.0)
        if t >= self.tf:
            return 0.0, 0.0, 0.0, 0.0
        w, v, a = self.w0, self.v0, self.a0
        el = 0.0
        for ds, jk in self._segs:
            if el + ds > t - 1e-14:
                tau = t - el
                return (w+v*tau+0.5*a*tau**2+(1/6)*jk*tau**3,
                        v+a*tau+0.5*jk*tau**2, a+jk*tau, jk)
            w += v*ds+0.5*a*ds**2+(1/6)*jk*ds**3
            v += a*ds+0.5*jk*ds**2; a += jk*ds; el += ds
        return 0.0, 0.0, 0.0, 0.0

    def _generators(self):
        """Return list of (generator_func, tf_minimum) for each structure."""
        j = self.j
        gens = []

        # Type A: 2-phase bang-bang
        for s in [1, -1]:
            sj = s*j
            # t1 = (tf - a0/sj)/2 >= 0 => tf >= a0/sj
            # t2 = tf - t1 >= 0 => always true if t1 <= tf
            # tf_min: both t1>=0 and t2>=0
            # t1>=0: tf >= a0/sj ; t2>=0: tf >= -a0/sj+tf => always
            # Actually t1 = (tf-a0/sj)/2, t2=(tf+a0/sj)/2
            # t1>=0: tf >= a0/sj ; t2>=0: tf >= -a0/sj
            # So tf_min = max(a0/sj, -a0/sj) = |a0|/j
            tf_min = abs(self.a0) / j
            gens.append((self._mk_two(sj), tf_min))

        # Type B: 3-phase
        for asat in [self.a_hi, self.a_lo]:
            da1 = asat - self.a0
            t1 = abs(da1)/j if abs(da1)>1e-10 else 0.0
            da3 = -asat
            t3 = abs(da3)/j if abs(da3)>1e-10 else 0.0
            tf_min = t1 + t3
            gens.append((self._mk_three(asat), tf_min))

        # Type C: 5-phase
        for a1, a2 in [(self.a_hi, self.a_lo), (self.a_lo, self.a_hi)]:
            da1 = abs(a1-self.a0)/j if abs(a1-self.a0)>1e-10 else 0.0
            da3 = abs(a2-a1)/j if abs(a2-a1)>1e-10 else 0.0
            da5 = abs(a2)/j if abs(a2)>1e-10 else 0.0
            tf_min = da1+da3+da5
            gens.append((self._mk_five(a1,a2), tf_min))

        return gens

    def _mk_two(self, sj):
        w0,v0,a0,ahi,alo = self.w0,self.v0,self.a0,self.a_hi,self.a_lo
        j = self.j
        def gen(tf):
            t1 = 0.5*(tf - a0/sj)
            t2 = tf - t1
            if t1<-1e-9 or t2<-1e-9: return None, None
            t1,t2 = max(0,t1), max(0,t2)
            ap = a0 + sj*t1
            if ap > ahi+1e-4 or ap < alo-1e-4: return None, None
            segs = [(t1,sj),(t2,-sj)]
            w,_,_ = _integ(segs,w0,v0,a0)
            return w, segs
        return gen

    def _mk_three(self, asat):
        j = self.j
        w0,v0,a0 = self.w0,self.v0,self.a0
        da1 = asat-a0
        j1 = (j if da1>0 else -j) if abs(da1)>1e-10 else 0.0
        t1 = abs(da1)/j if abs(da1)>1e-10 else 0.0
        da3 = -asat
        j3 = (j if da3>0 else -j) if abs(da3)>1e-10 else 0.0
        t3 = abs(da3)/j if abs(da3)>1e-10 else 0.0
        def gen(tf):
            t2 = tf-t1-t3
            if t2<-1e-9: return None, None
            t2 = max(0,t2)
            segs = []
            if t1>1e-12: segs.append((t1,j1))
            if t2>1e-12: segs.append((t2,0.0))
            if t3>1e-12: segs.append((t3,j3))
            if not segs: return None, None
            w,_,_ = _integ(segs,w0,v0,a0)
            return w, segs
        return gen

    def _mk_five(self, a1, a2):
        j = self.j
        w0,v0,a0_ = self.w0,self.v0,self.a0
        da1 = a1-a0_
        j1 = j if da1>=0 else -j
        t1 = abs(da1)/j if abs(da1)>1e-10 else 0.0
        da3 = a2-a1
        j3 = j if da3>=0 else -j
        t3 = abs(da3)/j if abs(da3)>1e-10 else 0.0
        da5 = -a2
        j5 = j if da5>=0 else -j
        t5 = abs(da5)/j if abs(da5)>1e-10 else 0.0

        def gen(tf):
            coast = tf-t1-t3-t5
            if coast<-1e-9: return None, None
            coast = max(0,coast)

            def vf(f):
                t2,t4 = coast*f, coast*(1-f)
                segs = []
                if t1>1e-12: segs.append((t1,j1))
                if t2>1e-12: segs.append((t2,0.0))
                if t3>1e-12: segs.append((t3,j3))
                if t4>1e-12: segs.append((t4,0.0))
                if t5>1e-12: segs.append((t5,j5))
                w,v,a = _integ(segs,w0,v0,a0_)
                return v, segs, w

            v0f,_,_ = vf(0)
            v1f,_,_ = vf(1)
            if abs(v0f)<1e-6:
                _,s,w = vf(0); return w,s
            if abs(v1f)<1e-6:
                _,s,w = vf(1); return w,s
            if v0f*v1f > 0:
                # Can't satisfy v=0; return the better one but penalized
                if abs(v0f)<abs(v1f):
                    _,s,w = vf(0); return w+1e8*abs(v0f), s
                else:
                    _,s,w = vf(1); return w+1e8*abs(v1f), s

            flo,fhi,vlo = 0.0,1.0,v0f
            for _ in range(15):  # Reduced from 50
                fm = 0.5*(flo+fhi)
                vm,segs,wm = vf(fm)
                if abs(vm)<1e-6 or (fhi-flo)<1e-6:
                    return wm, segs
                if vm*vlo>0: flo,vlo = fm,vm
                else: fhi = fm
            _,segs,wm = vf(0.5*(flo+fhi))
            return wm, segs
        return gen


class HehnTrajectoryGenerator:
    def __init__(self, qp=None):
        self.qp = qp or QuadParams()

    def generate(self, pos0, vel0, acc0, target=None, dp=None, optimize=True):
        pos0 = np.asarray(pos0,float)
        vel0 = np.asarray(vel0,float)
        acc0 = np.asarray(acc0,float)
        target = np.zeros(3) if target is None else np.asarray(target,float)
        p0 = pos0-target

        # Skip optimization for small displacements (near target)
        dist = np.linalg.norm(p0)
        if dist < 0.1:
            optimize = False

        if dp is None: dp = DecouplingParams()
        if optimize: dp = self._opt(p0,vel0,acc0,dp)
        xa,ya,zlo,zhi,jm = compute_axis_constraints(self.qp,dp)
        if jm<1e-6: jm=1.0
        axes = [
            AxisTrajectory(p0[0],vel0[0],acc0[0],-xa,xa,jm),
            AxisTrajectory(p0[1],vel0[1],acc0[1],-ya,ya,jm),
            AxisTrajectory(p0[2],vel0[2],acc0[2],zlo,zhi,jm),
        ]
        for ax in axes: ax.solve()
        return HehnTrajectory(axes,target,max(ax.tf for ax in axes),self.qp,dp)

    def _opt(self, p0, v0, a0, dp0):
        """Optimized decoupling parameter search for real-time performance."""
        best_dp, best_tf = DecouplingParams(**dp0.__dict__), np.inf
        zb = self.qp.a_min - self.qp.g

        # Determine which axes are active
        active = [abs(p0[i])>1e-4 or abs(v0[i])>1e-4 or abs(a0[i])>1e-4 for i in range(3)]

        # Reduced search: only 3 zm values
        for zm in [zb, (zb-0.1)/2, max(zb+0.1,-0.1)]:
            dp = DecouplingParams(0.5,0.5,zm,dp0.a0)
            azl,azh = 0.1,0.9

            # Reduced iterations: 5 instead of 15
            for _ in range(5):
                dp.alpha_z = 0.5*(azl+azh)
                axl,axh = 0.1,0.9

                # Inner: sync x and y (reduced to 5 iterations)
                if active[0] and active[1]:
                    for _ in range(5):
                        dp.alpha_x = 0.5*(axl+axh)
                        tx,ty = self._qt(p0,v0,a0,dp,[0,1])
                        if tx>ty: axh=dp.alpha_x
                        else: axl=dp.alpha_x
                        if (axh-axl)<0.05: break
                elif active[0] and not active[1]:
                    dp.alpha_x = 0.9
                elif active[1] and not active[0]:
                    dp.alpha_x = 0.1
                else:
                    dp.alpha_x = 0.5

                # Outer: sync horizontal vs z
                h_axes = [i for i in [0,1] if active[i]]
                if h_axes and active[2]:
                    txy = max(self._qt(p0,v0,a0,dp,h_axes))
                    tz = self._qt(p0,v0,a0,dp,[2])[0]
                    if tz>txy: azh=dp.alpha_z
                    else: azl=dp.alpha_z
                elif active[2] and not h_axes:
                    dp.alpha_z = 0.9
                    break
                elif h_axes and not active[2]:
                    dp.alpha_z = 0.1
                    break
                if (azh-azl)<0.05: break

            dp.alpha_z = 0.5*(azl+azh)
            tf = max(self._qt(p0,v0,a0,dp,[0,1,2]))
            if tf<best_tf:
                best_tf=tf
                best_dp = DecouplingParams(dp.alpha_x,dp.alpha_z,zm,dp.a0)
        return best_dp

    def _qt(self, p0, v0, a0, dp, axes):
        xa,ya,zlo,zhi,jm = compute_axis_constraints(self.qp,dp)
        if jm<1e-6: return [100.0]*len(axes)
        b = [(-xa,xa),(-ya,ya),(zlo,zhi)]
        return [AxisTrajectory(p0[i],v0[i],a0[i],b[i][0],b[i][1],jm).solve() for i in axes]


class HehnTrajectory:
    def __init__(self, axes, target, tf, qp, dp):
        self.axes,self.target,self.tf,self.qp,self.dp = axes,target,tf,qp,dp
    def get_position(self,t):
        return np.array([ax.evaluate(t)[0] for ax in self.axes])+self.target
    def get_velocity(self,t):
        return np.array([ax.evaluate(t)[1] for ax in self.axes])
    def get_acceleration(self,t):
        return np.array([ax.evaluate(t)[2] for ax in self.axes])
    def get_jerk(self,t):
        return np.array([ax.evaluate(t)[3] for ax in self.axes])
    def get_thrust(self,t):
        f=self.get_acceleration(t)+np.array([0,0,self.qp.g]); return np.linalg.norm(f)
    def get_thrust_dir(self,t):
        f=self.get_acceleration(t)+np.array([0,0,self.qp.g])
        n=np.linalg.norm(f); return f/n if n>1e-8 else np.array([0,0,1.0])
    def get_body_rate_bound(self,t,dt=5e-4):
        fb0,fb1=self.get_thrust_dir(t),self.get_thrust_dir(t+dt)
        return np.linalg.norm((fb1-fb0)/dt)
    @property
    def duration(self): return self.tf
    @property
    def per_axis_durations(self): return np.array([ax.tf for ax in self.axes])
