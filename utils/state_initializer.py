import numpy as np


def state_initialize(idle_rotor_speed, initial_offset = None, Dim = 6):
    """
    Initialize drone state and rotor state

    Args:
        idle_rotor_speed: Idle rotor speed [RPM]
        initial_offset: Initial position offset (optional)
    """

    if Dim == 6:
        if initial_offset is not None:
            p = np.array(initial_offset)
        else:
            p = np.zeros((3,))

        v = np.zeros((3,))
        q = np.array([1.0, 0.0, 0.0, 0.0])
        w = np.zeros((3,))
        s_drone = np.concatenate([p, v, q, w])

        w_rotor = idle_rotor_speed * np.ones((6,))
        alpha_rotor = np.zeros((6,))
        s_rotor = np.concatenate([w_rotor, alpha_rotor])

    elif Dim == 3:
        if initial_offset is not None:
            p = np.array(initial_offset)
        else:
            p = np.array([0.0, 0.258])

        v = np.zeros((2,))
        theta = 0.0
        q = 0.0
        s_drone = np.concatenate([p, v, theta, q])

        w_rotor = idle_rotor_speed * np.ones((3,))
        alpha_rotor = np.zeros((3,))
        s_rotor = np.concatenate([w_rotor, alpha_rotor])

    return s_drone, s_rotor