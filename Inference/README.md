# P-DroNet Powerline Tracking

Link repo: https://rtgit.rta.vn/rtr_huylda/powerline-tracking

- Part 1: [Training](/Training/)

- Part 2: [TensorRT](/TensorRT-Inference/)

- Part 3: [Inference](/Inference)

# Part 3: Inference

# **Inference engine TensorRT**

## **Create class TrtModel**

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

---

## **Create Powerline_inference method to inference DroNet**

        def Powerline_inference(self):
            global exit_event
            threading.Thread.__init__(self)
            batch_size = 1
            t1=0
            t2=0
            number = 0
            dir = './images/' // check one image saved
            self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) # Use Camera
            theta_k_1 = 0
            try:
                while True:
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
                    if line >= 0.5:
                        powerline = int(1)
                    elif line < 0.5:
                        powerline = int(0)
                    WriteJson(self.ipc_fifo_name, [round(theta_k), powerline])
                    keyCode = cv2.waitKey(33) # & 0xFF
                    print('run')
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

---

## **Run inference**

#### Run the rtr_get_data_Powerline_Tracking_NEOM_fifo_json.py to create the fifo.json file to transfer tracking angle to rtr_rd22_002.py

    sudo python3 rtr_get_data_Powerline_Tracking_NEOM_fifo_json.py

#### Run the rtr_Powerline_Tracking_NEOM_fifo_json.py to inference the tracking angle by TensorRT.

    sudo python3 rtr_Powerline_Tracking_NEOM_fifo_json.py
