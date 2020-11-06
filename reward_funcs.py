import markov_pilot.tasks.rewards as rewards
import markov_pilot.environment.properties as prp

from typing import Tuple

# the next three make_base_reward_components functions re successfully used in the experiments

# the standard 0-output reward function. Pass it in from here, as otherwise, the restore gets nifty due to the indent within a class definition.
def _make_base_reward_components(self):     #may be overwritten by injected custom function
    # pylint: disable=method-hidden
    """
    Just adds an Asymptotic error component as standard reward to the PID_FlightTask.

    May be overwritten by injected custom function.
    """
    base_components = (     
        rewards.ConstantDummyRewardComponent(name = self.name+'_dummyRwd', const_output=0.0),
    )
    return base_components

def make_angular_integral_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ANGLE_DEG_ERROR_SCALING = 0.1
    CMD_TRAVEL_MAX = 2/4  # a qarter of the max. absolute value of the delta-cmd;
    ANGLE_INT_DEG_MAX = self.integral_limit
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_Angle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ANGLE_DEG_ERROR_SCALING,
                                weight=6),
        rewards.LinearErrorComponent(name='rwd_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=CMD_TRAVEL_MAX,
                                weight=4),
        rewards.LinearErrorComponent(name='rwd_Angle_error_Integral',
                                prop=self.prop_error_integral,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ANGLE_INT_DEG_MAX,
                                weight=10),
    )
    return base_components

def make_throttle_integral_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    THROTTLE_DEG_ERROR_SCALING = 0.1
    CMD_TRAVEL_MAX = 2/4  # a qarter of the max. absolute value of the delta-cmd;
    THROTTLE_INT_MAX = self.integral_limit
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_IAS_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=THROTTLE_DEG_ERROR_SCALING,
                                weight=6),
        rewards.LinearErrorComponent(name='rwd_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=CMD_TRAVEL_MAX,
                                weight=4),
        # rewards.LinearErrorComponent(name='rwd_IAS_error_Integral',
        #                         prop=self.prop_error_integral,
        #                         state_variables=self.obs_props,
        #                         target=0.0,
        #                         potential_difference_based=False,
        #                         scaling_factor=THROTTLE_INT_MAX,
        #                         weight=10),
    )
    return base_components

def make_sideslip_angle_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    SIDESLIP_ANGLE_DEG_ERROR_SCALING = 0.5
    RUDDER_CMD_TRAVEL_MAX = 2/4  # a quarter of the max. absolute value of the delta-cmd; leverage the non-linearities
    SIDESLIP_ANGLE_INT_DEG_MAX = self.integral_limit
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_sideslipAngle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=SIDESLIP_ANGLE_DEG_ERROR_SCALING,
                                weight=4),
        rewards.LinearErrorComponent(name='rwd_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=RUDDER_CMD_TRAVEL_MAX,
                                weight=1),      #for the rudder, we increase this priority
        rewards.LinearErrorComponent(name='rwd_sideslipAngle_error_Integral',
                                prop=self.prop_error_integral,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=SIDESLIP_ANGLE_INT_DEG_MAX,
                                weight=1),
    )
    return base_components

def make_glide_path_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ANGLE_DEG_ERROR_SCALING = 0.5 # we need some bigger scaling factor as it's extremely difficult and we need some guiding reward here
    CMD_TRAVEL_MAX = 2/4  # a qarter of the max. absolute value of the delta-cmd;
    ANGLE_INT_DEG_MAX = self.integral_limit
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_Angle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ANGLE_DEG_ERROR_SCALING,
                                weight=6),
        rewards.LinearErrorComponent(name='rwd_Angle_error_Integral',
                                prop=self.prop_error_integral,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ANGLE_INT_DEG_MAX,
                                weight=10),
    )
    return base_components

def make_altitude_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ALTI_ERROR_SCALING = 50 # we need some bigger scaling factor as it's extremely difficult and we need some guiding reward here
    ALTI_INT_DEG_MAX = self.integral_limit
    base_components = (
        rewards.AsymptoticErrorComponent(name='rwd_Altitude_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ALTI_ERROR_SCALING,
                                weight=6),
        rewards.LinearErrorComponent(name='rwd_Altitude_error_Integral',
                                prop=self.prop_error_integral,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ALTI_INT_DEG_MAX,
                                weight=10),
    )
    return base_components

def make_elevator_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ANGLE_DEG_ERROR_SCALING = 0.1
    CMD_TRAVEL_MAX = 2/4  # a qarter of the max. absolute value of the delta-cmd;
    ANGLE_INT_DEG_MAX = self.integral_limit
    base_components = (
        rewards.LinearErrorComponent(name='rwd_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=CMD_TRAVEL_MAX,
                                weight=4),
    )
    return base_components


def make_ias_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    KIAS_ERROR_SCALING = 5  # we need some bigger scaling factor as it's extremely difficult and we need some guiding reward here
    KIAS_INT_MAX = self.integral_limit
    base_components = (
        rewards.AsymptoticErrorComponent(name='rwd_IAS_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=KIAS_ERROR_SCALING,
                                weight=6),
        rewards.LinearErrorComponent(name='rwd_IAS_error_Integral',
                                prop=self.prop_error_integral,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=KIAS_INT_MAX,
                                weight=10),
    )
    return base_components

def make_throttle_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    CMD_TRAVEL_MAX = 1/2  # a qarter of the max. absolute value of the delta-cmd;
    base_components = (
        rewards.LinearErrorComponent(name='rwd_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=CMD_TRAVEL_MAX,
                                weight=4),
    )
    return base_components






###########################################################################
# all make_base_reward_components below were just used during experiments
# some might be useful, but no guarantees

def make_glide_angle_reward_components(self):
    GLIDE_ANGLE_DEG_ERROR_SCALING = 0.1
    GLIDE_ANGLE_INT_DEG_MAX = self.integral_limit
    ELEVATOR_CMD_TRAVEL_MAX = 2/4 # a quarter of the max. absolute value of the delta-cmd; leverage the non-linearities
    GLIDE_ANGULAR_VELOCITY_SCALING = 0.25
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_glideAngle_error',
                    prop=self.prop_error,
                    state_variables=self.obs_props,
                    target=0.0,
                    potential_difference_based=False,
                    scaling_factor=GLIDE_ANGLE_DEG_ERROR_SCALING,
                    weight=4),
        rewards.LinearErrorComponent(name='rwd_glideAngle_error_derivative',
            prop=self.prop_error_derivative,
            state_variables=self.obs_props,
            target=0.0,
            potential_difference_based=False,
            scaling_factor=2,
            weight=1),
        rewards.LinearErrorComponent(name='rwd_glideAngle_error_Integral',
                    prop=self.prop_error_integral,
                    state_variables=self.obs_props,
                    target=0.0,
                    potential_difference_based=False,
                    scaling_factor=GLIDE_ANGLE_INT_DEG_MAX,
                    weight=9),
        rewards.LinearErrorComponent(name='rwd_elevator_cmd_travel_error',
                    prop=self.prop_delta_cmd,
                    state_variables=self.obs_props,
                    target=0.0,
                    potential_difference_based=False,
                    scaling_factor=ELEVATOR_CMD_TRAVEL_MAX,
                    weight=2),
        # rewards.LinearErrorComponent(name='rwd_angular_velocity_q_rad',
        #             prop=prp.q_radps,
        #             state_variables=self.obs_props,
        #             target=0.0,
        #             potential_difference_based=False,
        #             scaling_factor=GLIDE_ANGULAR_VELOCITY_SCALING,
        #             weight=1),
    )
    return base_components

def make_elevator_actuation_reward_components(self):
    ELEVATOR_CMD_TRAVEL_MAX = 2/4 # a quarter of the max. absolute value of the delta-cmd; leverage the non-linearities
    base_components = (
        rewards.LinearErrorComponent(name='rwd_elevator_cmd_travel_error',
                    prop=self.prop_delta_cmd,
                    state_variables=self.obs_props,
                    target=0.0,
                    potential_difference_based=False,
                    scaling_factor=ELEVATOR_CMD_TRAVEL_MAX,
                    weight=2),
    )
    return base_components

def make_glide_path_angle_reward_components(self):
    GLIDE_ANGLE_DEG_ERROR_SCALING = 0.1
    GLIDE_ANGLE_INT_DEG_MAX = self.integral_limit
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_glideAngle_error',
                    prop=self.prop_error,
                    state_variables=self.obs_props,
                    target=0.0,
                    potential_difference_based=False,
                    scaling_factor=GLIDE_ANGLE_DEG_ERROR_SCALING,
                    weight=4),
        rewards.LinearErrorComponent(name='rwd_glideAngle_error_derivative',
            prop=self.prop_error_derivative,
            state_variables=self.obs_props,
            target=0.0,
            potential_difference_based=False,
            scaling_factor=2,
            weight=1),
        rewards.LinearErrorComponent(name='rwd_glideAngle_error_Integral',
                    prop=self.prop_error_integral,
                    state_variables=self.obs_props,
                    target=0.0,
                    potential_difference_based=False,
                    scaling_factor=GLIDE_ANGLE_INT_DEG_MAX,
                    weight=9),
    )
    return base_components

def make_speed_reward_components(self):
    KIAS_ERROR_SCALING = 1
    KIAS_ERR_INT_DEG_MAX = self.integral_limit
    ELEVATOR_CMD_TRAVEL_MAX = 2/4 # a quarter of the max. absolute value of the delta-cmd; leverage the non-linearities
    GLIDE_ANGULAR_VELOCITY_SCALING = 0.25
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_kias_error',
                    prop=self.prop_error,
                    state_variables=self.obs_props,
                    target=0.0,
                    potential_difference_based=False,
                    scaling_factor=KIAS_ERROR_SCALING,
                    weight=4),
        rewards.LinearErrorComponent(name='rwd_kias_error_Integral',
                    prop=self.prop_error_integral,
                    state_variables=self.obs_props,
                    target=0.0,
                    potential_difference_based=False,
                    scaling_factor=KIAS_ERR_INT_DEG_MAX,
                    weight=9),
        rewards.LinearErrorComponent(name='rwd_elevator_cmd_travel_error',
                    prop=self.prop_delta_cmd,
                    state_variables=self.obs_props,
                    target=0.0,
                    potential_difference_based=False,
                    scaling_factor=ELEVATOR_CMD_TRAVEL_MAX,
                    weight=2),
        rewards.LinearErrorComponent(name='rwd_angular_velocity_q_rad',
                    prop=prp.q_radps,
                    state_variables=self.obs_props,
                    target=0.0,
                    potential_difference_based=False,
                    scaling_factor=GLIDE_ANGULAR_VELOCITY_SCALING,
                    weight=1),
    )
    return base_components

def make_roll_angle_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ROLL_ANGLE_DEG_ERROR_SCALING = 0.5
    ROLL_ANGLE_INT_DEG_MAX = self.integral_limit
    AILERON_CMD_TRAVEL_MAX = 2/4  # a quarter of the max. absolute value of the delta-cmd; leverage the non-linearities
    ROLL_ANGULAR_VELOCITY_SCALING = 0.25
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_rollAngle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ROLL_ANGLE_DEG_ERROR_SCALING,
                                weight=4),
        rewards.LinearErrorComponent(name='rwd_rollAngle_error_Integral',
                                prop=self.prop_error_integral,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ROLL_ANGLE_INT_DEG_MAX,
                                weight=9),
        rewards.LinearErrorComponent(name='rwd_aileron_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=AILERON_CMD_TRAVEL_MAX,
                                weight=2),
        #check if the angular-velocity criterion is more helpful to avoid flittering. 
        #from a causality point of view, the command travel should be the criterion, as we want to avoid excessive movement.
        #The control surface travel (derivative) must be presented to the ANN anyways.
        rewards.LinearErrorComponent(name='rwd_roll_angular_velocity',
                                prop=prp.p_radps,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ROLL_ANGULAR_VELOCITY_SCALING,
                                weight=1),
    )
    return base_components

def make_sideslip_angle_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    SIDESLIP_ANGLE_DEG_ERROR_SCALING = 0.5
    SIDESLIP_ANGLE_INT_DEG_MAX = self.integral_limit
    RUDDER_CMD_TRAVEL_MAX = 2/4  # a quarter of the max. absolute value of the delta-cmd; leverage the non-linearities
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_sideslipAngle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=SIDESLIP_ANGLE_DEG_ERROR_SCALING,
                                weight=4),
        rewards.LinearErrorComponent(name='rwd_rudder_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=RUDDER_CMD_TRAVEL_MAX,
                                weight=1),      #for the rudder, we increase this priority
        rewards.LinearErrorComponent(name='rwd_sideslipAngle_error_Integral',
                                prop=self.prop_error_integral,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=SIDESLIP_ANGLE_INT_DEG_MAX,
                                weight=1),
    )
    return base_components

def make_roll_angle_error_only_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ROLL_ANGLE_DEG_ERROR_SCALING = 0.5
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_rollAngle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ROLL_ANGLE_DEG_ERROR_SCALING,
                                weight=1),
    )
    return base_components

def make_roll_angle_error_punish_actuation_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ROLL_ANGLE_DEG_ERROR_SCALING = 0.5
    AILERON_CMD_TRAVEL_MAX = 2  # the max. absolute value of the delta-cmd;

    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_rollAngle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ROLL_ANGLE_DEG_ERROR_SCALING,
                                weight=8),
        rewards.LinearErrorComponent(name='rwd_aileron_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=AILERON_CMD_TRAVEL_MAX,
                                weight=2),
    )
    return base_components

def make_roll_angle_integral_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ROLL_ANGLE_DEG_ERROR_SCALING = 0.5
    AILERON_CMD_TRAVEL_MAX = 2/4  # a qarter of the max. absolute value of the delta-cmd;
    ROLL_ANGLE_INT_DEG_MAX = self.integral_limit
    ROLL_ANGLE_DER_DEG_MAX = 10
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_rollAngle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ROLL_ANGLE_DEG_ERROR_SCALING,
                                weight=4),
        rewards.LinearErrorComponent(name='rwd_aileron_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=AILERON_CMD_TRAVEL_MAX,
                                weight=2),
        rewards.LinearErrorComponent(name='rwd_rollAngle_error_Integral',
                                prop=self.prop_error_integral,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ROLL_ANGLE_INT_DEG_MAX,
                                weight=9),
        rewards.QuadraticErrorComponent(name='rwd_rollAngle_error_Derivative',
                                prop=self.prop_error_derivative,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ROLL_ANGLE_DER_DEG_MAX**2,
                                weight=2),
    )
    return base_components

def make_angular_error_only_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ANGLE_DEG_ERROR_SCALING = 0.1
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_rollAngle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ANGLE_DEG_ERROR_SCALING,
                                weight=1),
    )
    return base_components

def make_angular_error_punish_actuation_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ANGLE_DEG_ERROR_SCALING = 0.5
    CMD_TRAVEL_MAX = 2/4  # a quarter of the max. absolute value of the delta-cmd;

    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_Angle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ANGLE_DEG_ERROR_SCALING,
                                weight=8),
        rewards.LinearErrorComponent(name='rwd_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=CMD_TRAVEL_MAX,
                                weight=2),
    )
    return base_components


def make_angular_derivative_integral_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ANGLE_DEG_ERROR_SCALING = 0.1
    CMD_TRAVEL_MAX = 2/4  # a qarter of the max. absolute value of the delta-cmd;
    ANGLE_INT_DEG_MAX = self.integral_limit
    ANGLE_DER_DEG_MAX = 0.1
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_Angle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ANGLE_DEG_ERROR_SCALING,
                                weight=6),
        rewards.LinearErrorComponent(name='rwd_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=CMD_TRAVEL_MAX,
                                weight=4),
        rewards.LinearErrorComponent(name='rwd_Angle_error_Integral',
                                prop=self.prop_error_integral,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ANGLE_INT_DEG_MAX,
                                weight=10),
        # rewards.AsymptoticErrorComponent(name='rwd_Angle_error_Derivative',
        #                         prop=self.prop_error_derivative,
        #                         state_variables=self.obs_props,
        #                         target=0.0,
        #                         potential_difference_based=False,
        #                         scaling_factor=ANGLE_DER_DEG_MAX,
        #                         weight=3),
    )
    return base_components

def make_rudder_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
    ANGLE_DEG_ERROR_SCALING = 0.1
    CMD_TRAVEL_MAX = 2/4  # a qarter of the max. absolute value of the delta-cmd;
    ANGLE_INT_DEG_MAX = self.integral_limit
    ANGLE_DER_DEG_MAX = 0.1
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_Angle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ANGLE_DEG_ERROR_SCALING,
                                weight=6),
        rewards.LinearErrorComponent(name='rwd_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=CMD_TRAVEL_MAX,
                                weight=6),
        rewards.LinearErrorComponent(name='rwd_Angle_error_Integral',
                                prop=self.prop_error_integral,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ANGLE_INT_DEG_MAX,
                                weight=10),
        rewards.AsymptoticErrorComponent(name='rwd_Angle_error_Derivative',
                                prop=self.prop_error_derivative,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ANGLE_DER_DEG_MAX,
                                weight=3),
    )
    return base_components



