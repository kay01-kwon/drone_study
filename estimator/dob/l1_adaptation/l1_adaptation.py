import numpy as np
from utils.drone_converter import HexaConverter
from estimator.dob.l1_adaptation.state_predictor import StatePredictor
from estimator.dob.l1_adaptation.adaptation_law import AdaptationLaw
from utils.low_pass_filter import LowPassFilter
from utils.math_tool import quaternion_to_rotm

class L1Adaptation:
    def __init__(self, DynParam, DroneParam, RotorParam ,DobParam):

        # Pass parameter
        dynamic_param = DynParam
        drone_param = DroneParam
        rotor_param = RotorParam
        DobParam = DobParam

        # Get cutoff frequency
        freq_cutoff_trans = DobParam['freq_cutoff_trans']
        freq_cutoff_rot = DobParam['freq_cutoff_rot']

        # Low pass filter setup
        self.trans_lpf_obj = LowPassFilter(cut_off_freq=freq_cutoff_trans,
                                      dim=3)
        self.rot_lpf_obj = LowPassFilter(cut_off_freq=freq_cutoff_rot,
                                    dim=3)

        self.hexa_converter = HexaConverter(DroneParams = drone_param,
                                            RotorParams = rotor_param)

        self.state_predictor = StatePredictor(DynParam = dynamic_param,
                                              DobParam = DobParam)

        self.adaptation_law = AdaptationLaw(DynParam=dynamic_param,
                                            DobParam=DobParam)

        self.sigma_hat = np.zeros(6)
        self.sigma_f_lpf = np.zeros(3)
        self.sigma_tau_lpf = np.zeros(3)

    def dob_estimate(self, t_prev, t_curr, rotor_speed, state):

        u_rot = self.hexa_converter.compute_u(rotor_speed)

        z_hat = self.state_predictor.update(t_prev, t_curr, state, u_rot, self.sigma_hat)
        z = self._pack_state(state)
        z_tilde = z_hat - z

        self.sigma_hat = self.adaptation_law.update(t_prev, t_curr, z_tilde, state)

        sigma_f = np.array([self.sigma_hat[0], self.sigma_hat[4], self.sigma_hat[5]])
        sigma_tau = np.array([self.sigma_hat[1], self.sigma_hat[2], self.sigma_hat[3]])

        dt = t_curr - t_prev
        sigma_f_lpf = self.trans_lpf_obj.do_filter(sigma_f, dt)
        self.sigma_f_lpf = np.array([0.0, 0.0, sigma_f_lpf[2]])
        self.sigma_tau_lpf = self.trans_lpf_obj.do_filter(sigma_tau, dt)

        return np.concatenate([self.sigma_f_lpf, self.sigma_tau_lpf])

    def _pack_state(self, state):
        v_body = state[3:6]
        q = state[6:10]
        R_b_w = quaternion_to_rotm(q)
        v_world = R_b_w @ v_body
        w = state[10:13]
        return np.concatenate([v_world, w])
