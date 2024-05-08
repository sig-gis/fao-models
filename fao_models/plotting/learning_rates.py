#%%
import numpy as np
import matplotlib.pyplot as plt
# this does not use the actual built-in keras learning rate schedulers, so not an actual useable module
# mostly a toy plotter to help understand how learning rate schedulers work

class LearningRateScheduler:
    def __init__(self, lr, lr_decay, lr_decay_step, step=0, decay_fn=None):
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_step = lr_decay_step
        self.step = step
        self.decay_fn = decay_fn
        
        # formula from https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/
        def exp_decay(self):
            return self.lr * (self.lr_decay ** (self.step*(1 / self.lr_decay_step)))
        # formula from https://keras.io/api/optimizers/learning_rate_schedules/inverse_time_decay/
        def inverse_time_decay(self):
            return self.lr / (1 + self.lr_decay * self.step / self.lr_decay_step)
        
        self.decay_fns = {
            'exp_decay': exp_decay,
            'inverse_time_decay': inverse_time_decay,
        }
        
        decay_fn_ = self.decay_fns[self.decay_fn]
        self.lr_step = decay_fn_(self)
      

class ExponentialDecayScheduler(LearningRateScheduler):
    def __init__(self, lr, lr_decay, lr_decay_step, step=0, decay_fn='exp_decay'):
        super().__init__(lr, lr_decay, lr_decay_step, step, decay_fn)
        
        decay_fn_ = self.decay_fns[self.decay_fn]
        self.lr_step = decay_fn_(self)

class InverseTimeDecayScheduler(LearningRateScheduler):
    def __init__(self, lr, lr_decay, lr_decay_step, step=0, decay_fn='inverse_time_decay'):
        super().__init__(lr, lr_decay, lr_decay_step, step, decay_fn)
        
        decay_fn_ = self.decay_fns[self.decay_fn]
        self.lr_step = decay_fn_(self)

# would need to extend parent class to accept required Polynomial args or allow parent class
# to accept **kwargs in a super().__init__(**kwargs) call
# https://stackoverflow.com/questions/39887422/more-arguments-in-derived-class-init-than-base-class-init
# class PolynomialDecayScheduler(LearningRateScheduler):
#     def __init__(self, lr, lr_decay, lr_decay_step, step=0, decay_fn='polynomial_decay'):
#         super().__init__(lr, lr_decay, lr_decay_step, step, decay_fn)
        
#         decay_fn_ = self.decay_fns[self.decay_fn]
#         self.lr_step = decay_fn_(self)

#%%
# testing real working experiment hyperparams for learning rate schedulers
# plot learning rate schedules as a function of epochs the way we are doing currently
total_examples = 77000
test_split = 0.2
batch_size = 64
steps_per_epoch = total_examples * test_split // batch_size

#%%
# Inverse Time Decay
# for ITD decay rate needs to be more aggressive the more epochs we have
lr_decay = 1
lr = 0.001
for epochs in [5,50,100]:
    decay_steps = int(steps_per_epoch * epochs)
    # lr_decay = base_decay_rate * (1/5*epochs)
    lr_vals = []
    for i in range(1,decay_steps):
        lr_val = InverseTimeDecayScheduler(lr, lr_decay, decay_steps, step = i).lr_step
        lr_vals.append(lr_val)
    print(lr_vals[:10])
    plt.plot(lr_vals, label=f'{epochs} epochs')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title(f'Inverse Time Decay - lr={lr}, decay_rate={lr_decay}')
# %%
# Exponential Decay
lr_decay = 0.9
lr = 0.001
for epochs in [5,50,100]:
    decay_steps = int(steps_per_epoch * epochs)
    lr_vals = []
    for i in range(1,decay_steps):
        lr_val = ExponentialDecayScheduler(lr, lr_decay, decay_steps, step = i).lr_step
        lr_vals.append(lr_val)
    print(lr_vals[:10])
    plt.plot(lr_vals, label=f'{epochs} epochs')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title(f'Exponential Decay - lr={lr}, decay_rate={lr_decay}')
# %%
