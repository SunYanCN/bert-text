from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

class WarmupCallback(Callback):
	def __init__(self, lr_var, init_lr, total_steps, warmup_steps=0):
		self.step = 0
		self.lr_var = lr_var
		self.init_lr = init_lr
		self.warmup = warmup_steps
		self.total_steps = total_steps
	def on_batch_begin(self, batch, logs):
		self.step += 1
		if self.step <= self.warmup:
			new_lr = self.init_lr * (self.step / self.warmup)
		else:
			new_lr = self.init_lr * max(0, 1 - self.step / self.total_steps)
		K.set_value(self.lr_var, new_lr)

def get_suggested_scheduler(init_lr=5e-5, total_steps=10000, warmup_ratio=0.1):
	opt_lr = K.variable(init_lr)
	warmup_steps = int(warmup_ratio * total_steps)
	warmup = WarmupCallback(opt_lr, init_lr, total_steps, warmup_steps)
	return warmup, opt_lr

