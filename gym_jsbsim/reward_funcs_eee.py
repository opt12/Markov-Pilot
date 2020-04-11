from gym_jsbsim import rewards

def make_pitch_reward_components(self):
    PITCH_DEVIATION_SCALING = 0.25 
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_'+self.name+'_asymptotic_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=PITCH_DEVIATION_SCALING,
                                weight=1),
        )
    return base_components

def make_roll_reward_components(self):
    ROLL_DEVIATION_SCALING = 0.1 
    base_components = (
        rewards.AngularAsymptoticErrorComponent(name='rwd_'+self.name+'_asymptotic_error',
                                prop=self.prop_error,
                                state_variables=self.obs_props,
                                target=0.0,
                                potential_difference_based=False,
                                scaling_factor=ROLL_DEVIATION_SCALING,
                                weight=1),
        )
    return base_components

