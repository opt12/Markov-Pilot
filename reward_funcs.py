import markov_pilot.tasks.rewards as rewards
import markov_pilot.environment.properties as prp

from typing import Tuple

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
    SIDESLIP_ANGULAR_VELOCITY_SCALING = 0.25
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_sideslipAngle_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=SIDESLIP_ANGLE_DEG_ERROR_SCALING,
                                weight=4),
        rewards.LinearErrorComponent(name='rwd_sideslipAngle_error_derivative',
                                prop=self.prop_error_derivative,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=2,
                                weight=1),
        rewards.LinearErrorComponent(name='rwd_sideslipAngle_error_Integral',
                                prop=self.prop_error_integral,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=SIDESLIP_ANGLE_INT_DEG_MAX,
                                weight=1),
        rewards.LinearErrorComponent(name='rwd_rudder_cmd_travel_error',
                                prop=self.prop_delta_cmd,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=RUDDER_CMD_TRAVEL_MAX,
                                weight=1),      #for the rudder, we increase this priority
        #check if the angular-velocity criterion is more helpful to avoid flittering. 
        #from a causality point of view, the command travel should be the criterion, as we want to avoid excessive movement.
        #The control surface travel (derivative) must be presented to the ANN anyways.
        # rewards.LinearErrorComponent(name='rwd_sideslip_angular_velocity',
        #                         prop=prp.r_radps,
        #                         state_variables=self.obs_props,
        #                         target=0.0,
        #                         potential_difference_based=False,
        #                         scaling_factor=SIDESLIP_ANGULAR_VELOCITY_SCALING,
        #                         weight=5),      #for the rudder, we increase this priority
    )
    return base_components

