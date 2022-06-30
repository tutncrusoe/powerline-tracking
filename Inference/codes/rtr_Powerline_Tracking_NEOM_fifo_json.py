import pycuda.autoinit  # This is needed for initializing CUDA driver
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import sys
from signal import signal, SIGINT, SIG_DFL
import threading
import os 
import struct
import json
import time

exit_event = threading.Event()

def gstreamer_pipeline(
	sensor_id=0,
	sensor_mode=2,
	capture_width=256,
	capture_height=144,
	display_width=256,
	display_height=144,
	framerate=30,
	flip_method=0,

):
	return (
		"nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
		"video/x-raw(memory:NVMM), "
		"width=(int)%d, height=(int)%d, "
		"format=(string)NV12, framerate=(fraction)%d/1 ! "
		"nvvidconv flip-method=%d ! "
		"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
		"videoconvert ! "
		"video/x-raw, format=(string)BGR ! appsink drop=1"
		% (
			sensor_id,
			sensor_mode,
			capture_width,
			capture_height,
			framerate,
			flip_method,
			display_width,
			display_height,
		)
	)


class HostDeviceMem(object):
	def __init__(self, host_mem, device_mem):
		self.host = host_mem
		self.device = device_mem

	def __str__(self):
		return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

	def __repr__(self):
		return self.__str__()

class TrtModel():
	def __init__(self, engine_path, ipc_fifo_name, max_batch_size=1, dtype=np.float32):
		self.device = cuda.Device(0) # 0 is your GPU number
		self.ctx = self.device.make_context()
		self.engine_path = engine_path
		self.dtype = dtype
		self.logger = trt.Logger(trt.Logger.WARNING)
		self.runtime = trt.Runtime(self.logger)
		self.engine = self.load_engine(self.runtime, self.engine_path)
		self.max_batch_size = max_batch_size

		self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
		self.context = self.engine.create_execution_context()
		
		self.ipc_fifo_name = ipc_fifo_name

	@staticmethod
	def load_engine(trt_runtime, engine_path):
		trt.init_libnvinfer_plugins(None, "")             
		with open(engine_path, 'rb') as f:
			engine_data = f.read()
		engine = trt_runtime.deserialize_cuda_engine(engine_data)
		return engine

	# Allocate host and device buffers, and create a stream.
	def allocate_buffers(self):
		inputs = []
		outputs = []
		bindings = []
		# Create a stream in which to copy inputs/outputs and run inference.
		stream = cuda.Stream()

		for binding in self.engine:
			size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
			# Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
			host_mem = cuda.pagelocked_empty(size, self.dtype)
			# Allocate device memory for inputs and outputs.
			device_mem = cuda.mem_alloc(host_mem.nbytes)
			bindings.append(int(device_mem))
			if self.engine.binding_is_input(binding):
				inputs.append(HostDeviceMem(host_mem, device_mem))
			else:
				outputs.append(HostDeviceMem(host_mem, device_mem))

		return inputs, outputs, bindings, stream

	def __call__(self,x:np.ndarray,batch_size=2):
		x = x.astype(self.dtype)
		np.copyto(self.inputs[0].host,x.ravel())
		for inp in self.inputs:
			cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
		self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
		for out in self.outputs:
			cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
		self.stream.synchronize()
		return [out.host.reshape(batch_size,-1) for out in self.outputs]

	def __del__(self):
		"""Free CUDA memories."""
		del self.outputs
		del self.inputs
		del self.bindings
		del self.stream

	def Powerline_inference(self):
		global exit_event
		threading.Thread.__init__(self)
		batch_size = 1
		t1=0
		t2=0
		number = 0
		dir = './images/' # check one image saved
		self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) # Use Camera
		theta_k_1 = 0
		try:
			while True:
				# self.ctx.push()

				ret_val, image = self.cap.read()
				if number == 0:
					cv2.imwrite(dir + str(number) + '.png', image)
					number = 1
				# img= image.copy()
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				image = cv2.resize(image, (200, 200))
				image = image * (1. / 255)
				image = np.array(image).astype(np.float32)  # scaling
				image = np.expand_dims(image, axis=0)
				## P-dronet
				output = model(image,batch_size)
				tilt, line = output[0].item(), output[1].item()
				theta_k = (1-0.5)*theta_k_1 + 0.5*90*tilt
				theta_k_1 = theta_k
				# if theta_k >= 0:
					# tilt_dir = 0
				# elif theta_k < 0:
					# tilt_dir = 1
				# split_num = str(theta_k).split('.')
				# tilt_int = int(split_num[0])
				# tilt_frac = int(str(split_num[1][0:1]))
				if line >= 0.5:
					powerline = int(1)
				elif line < 0.5:
					powerline = int(0)
				# msg_send = [int(round(theta_k)), int(powerline)]
				WriteJson(self.ipc_fifo_name, [round(theta_k), powerline])
				# cv2.imshow('P-DroNetRT', img)
				# self.send_message(tilt, line)
				keyCode = cv2.waitKey(33) # & 0xFF
				# if keyCode == ord('q'):
				# 	self.ctx.pop()
				# 	cap.release()
				# 	cv2.destroyAllWindows()
				# 	del self.ctx
				# 	del self.context
				# 	del self.engine
				# 	del self.runtime
				print('run')
				# 	break
				if exit_event.is_set():
					self.ctx.pop()
					self.cap.release()
					# cv2.destroyAllWindows()
					del self.ctx
					del self.context
					del self.engine
					del self.runtime
					print('exit-2')
					sys.exit()
					break
		except KeyboardInterrupt:
			exit_event.is_set()
			self.ctx.pop()
			self.cap.release()
			# cv2.destroyAllWindows()
			del self.ctx
			del self.context
			del self.engine
			del self.runtime
			print('exit-3')
			# break

def WriteJson(FIFO, data):
    j = json.dumps(data)
    with open(FIFO, 'w', buffering=1) as f:
        f.write(j)
        f.write('\n')

def handler(signal_received, frame):
	# This function handle ctrl+c command from user to stop the program.
	global exit_event
	exit_event.set()

if __name__ == "__main__":

	FIFO = "Powerline_Tracking.json"

	cuda.init()  # Initialize CUDA
	# signal(SIGINT, handler)
	signal(SIGINT, SIG_DFL)

	trt_engine_path = "../models/epoch-14_0.0008335579186677933.trt"

	model = TrtModel(trt_engine_path, FIFO)
	print('Loaded Model')
	model.Powerline_inference()
