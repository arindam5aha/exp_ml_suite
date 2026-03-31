"""MOT RunBuilder to ML communication interface.

This module implements a Qt-threaded bridge between GUI/experiment-control
components and machine-learning logic. It includes:
- TCP server startup/teardown,
- command-flag dispatching,
- run-parameter compilation and push utilities,
- image/ trace acquisition helpers,
- compressed persistence utilities.

Written by: Arindam Saha, ANU, 2025.
GitHub: arindam5aha, arindam.saha@anu.edu.au
"""

import bz2, pickle
from datetime import datetime, time
import numpy as np
from PySide2.QtCore import Slot, Signal, QThread
import json
import copy
import numpy as np
import os
import time
import subprocess
import signal
from threading import Thread, Event
from queue import Queue
from pueye_cam import ueye_cam_interface
from remote_interface import Receiver


REMOTE_HOST = '127.0.0.1'


class RBMLInterface(QThread):
    """Threaded interface between RunBuilder/LV control and ML components.

    The class receives command flags from an ML-side TCP client, executes
    corresponding control/acquisition actions, and sends responses through
    queue-backed communication channels.

    Signals:
        load_sess(dict): Emits loaded run/session dictionary.
        plot_ch(list, bool): Emits channels for plotting (analog/digital).
        push_run(bool): Triggers run push into control stack.
        exit(): Emitted when the interface is stopping.
        reset(bool, bool): Requests system reset from GUI side.
    """

    load_sess = Signal(dict)            # emit data_dict from run file
    plot_ch = Signal(list, bool)        # emit list of all channels to be plotted in RB, bool=False (True) to plot analog (digital) channels 
    push_run = Signal(bool)             # emit a signal with True to push run
    # get_lv_data = Signal(int)           # emit a signal with number of traces to read from scope
    exit  = Signal()
    reset = Signal(bool, bool)          # emit a signal with True to reset the system

    def __init__(self, runbuilder=None, ml_filename:str='run_rb.py', use_cam=True, check_locks=True):
        """Initialize the interface and default runtime state.

        Args:
            runbuilder: External runbuilder object expected to expose
                `job_done` signal, `work_thread`, and `server`.
            ml_filename: Filename of ML script relative to ./ML.
            use_cam: Whether to initialize camera interface.
            check_locks: Whether to start lock-check listener server.
        """
        super(RBMLInterface, self).__init__()

        self.script_path = os.path.join(os.getcwd(), 'ML', ml_filename)
        self.runbuilder = runbuilder
        self.runbuilder.job_done.connect(self.done)
        self._job_done = False
        self._running = True  # Add a running flag

        self.server = None      # server for mlrb to connect to
        self.ml_proc = None

        self.next_params = []   # stores raw params from ml
        self.next_run = []    # stores the RB executable channel list for next run
        self.data_dict_from_file = {}  # stores the run file data
        self.run_from_file = []   # stores a copy of the run file
        self.sys_info = {
                'params': 3,
                'time_bins': 21, 
                'bounds' : {'Trap freq':(1.6, 8.1),
                            'Repump freq':(2.4, 6.4),
                            'Mag fields':(0.0, 6.0)},
                'min_time': 0.10,
                'max_time' : 0.12,
                'img_start_time': 0.123,
                'img_window': 6e-3,
                'img_frames': 6,
                'img_pulse_duration': 2e-3,
                'img_chan': 'Trig',
                'cam_delay': 0,
                'cam_exposure': 0.3,
                'cam_pixel_clock': 35,
                'cam_frame_rate': 25.0,
                'cam_trigger_mode': 'rising_edge',
                'drop_sleep': 0.5,
                'img_sleep': 0.5
            }
        
        # for threading purposes between RB and ML
        self.comm_cycle = Thread(target=self.comm_thread, args=())
        self.comm_halt = Event()
        self.send_queue = Queue()
        self.read_queue = Queue()
        self.check_locks = check_locks     

        # for system checks and best runs
        self.ref_trace_lvl = 0.0        # stores probe lvl for checking lock break
        self.absorb_amt = 0.0           
        self.best_run_dict = {}
        self.tr_serv = None # locking server
        if use_cam:
            self.cam = ueye_cam_interface()
        else:
            self.cam = None

        self.default_run_path = '.\RunBuilder\Runs\PROBE_RUN'
        self.open_file(self.default_run_path)
        self.default_probe_run_dict = self._open_file('.\RunBuilder\Runs\PROBE_RUN')['channels']
        self.default_imaging_run_dict = self._open_file('.\RunBuilder\Runs\IMAGING_RUN')['channels']

        self.run_data = {'params':[], 'abs_imgs':[], 'ref_imgs':[], 'rewards':[]}
        self.data_file = './data/mlrb_side/run_data_'+get_datetime()+'.bz2'
        self.current_img_ref = None
        self.current_probe_ref = None

    def done(self):
        """Mark current external job as complete."""
        self._job_done = True
    
    def flag_done(self):
        """Queue a done flag response for ML side."""
        self.send_queue.put('<done>')

    def pause(self):
        """Block until external job_done signal is received."""
        print('\n>>> Waiting for LV to finish')
        while not self._job_done:
            time.sleep(0.1)
        print('>>> LV job done')
        self._job_done = False
        
    def start_server(self):
        """Start main ML listener and optional lock-check listener."""
        # create a server for ML purpose
        self.server = Receiver(REMOTE_HOST, 4444)
        self.server.start_server()
        if self.check_locks:
            self.tr_serv = Receiver(REMOTE_HOST, 3333) # all lock
            self.tr_serv.start_server()

        # start the comm cycle
        if not self.comm_cycle.is_alive():
            self.comm_cycle.start()
            #self.comm_cycle.join()

    def stop_server(self):
        """Stop listener threads and halt communication loop."""
        if self.server is not None:
            self.server.listening_thread.halt_set()
        self.comm_halt.set()
        #self.comm_cycle.join() 

    def json_send(self, data):  # send to ML interface
        """JSON-encode and send payload to ML client."""
        try:
            self.server.conn_send(json.dumps(data))
        except:
            print('Caught run time read error')

    def json_read(self):
        """Read from socket and decode JSON when possible.

        Returns:
            Decoded JSON object when valid JSON is received, raw data
            otherwise, or None on empty read.
        """
        # a flexible read solution for both json and non-json data
        data = self.server.conn_read()
        if data is not None and data != '':
            try:
                # if data is json encoded, return decoded data
                data = json.loads(data)
                return data   
            except:
                # else return raw data
                return data
        else:
            print('Could not fetch resonable data')

    def start_ml(self):       # 2
        """Start ML subprocess after ensuring server availability."""
        if self.server is None:
            self.start_server()
        cmd = ['gnome-terminal', '--', 'python', self.script_path, '&']
        self.ml_proc = subprocess.Popen(cmd)

    def stop_ml(self):
        """Stop ML subprocess if currently running."""
        if self.ml_proc is not None:
            os.kill(self.ml_proc.pid, signal.SIGINT)   #  fix this

    def comm_thread(self):
        """Main communication loop handling inbound/outbound messages."""
        self.runbuilder.work_thread.started.connect(self.comm_thread)
        while not self.comm_halt.is_set() and self._running:
            try:
                data = None
                # todo - make not blocking
                data = self.server.conn_read()
                
                # todo - perform some action on the data
                if data is not None:
                    self.read_queue.put(data)
                    self.flag_handler()

                if not self.send_queue.empty():
                    ret_data = self.send_queue.get()
                    self.server.conn_send(ret_data)

                # rate limit
                time.sleep(0.1)

            except Exception as e:
                print(f"\n>>> Error in comm_thread: {e}")
    
    @Slot()
    def quit(self):
        """Stop communication, persist run data, and clean up resources."""
        save_bz2(self.run_data, self.data_file)

        self._running = False
        self.comm_halt.set()  # Signal the communication thread to stop

        # if self.comm_cycle.is_alive():
        #     self.comm_cycle.join(timeout=5)  # Wait for the thread to finish

        if self.server is not None:
            self.stop_server()
            self.stop_ml()  # Stop the ML subprocess
        if self.cam is not None:
            self.cam.exit()
        self.terminate()

    @Slot()
    def flag_handler(self):
        """Dispatch incoming command flags from ML side to local actions.

        Protocol summary:
            The method consumes one command token from ``read_queue`` and may
            read additional payload(s) using ``json_read`` depending on the
            command.

        Supported flags and behavior:
            - ``<sys_info>``:
                Expects a JSON dict payload with system settings. Updates
                ``self.sys_info`` in place.

            - ``<clr>``:
                Clears pending items from LV send/read buffer and queues
                ``<done>``.

            - ``<drop>``:
                Builds and pushes a drop run (blank selected channels), then
                queues completion.

            - ``<acq>``:
                Expects numeric payload ``num_traces``. Acquires traces using
                ``acq_trace(..., flag=True)`` and sends JSON-encoded values.

            - ``<acq_ref>``:
                Expects numeric payload ``num_traces``. Acquires a reference
                trace set, stores it in ``self.ref_trace_lvl``, and sends JSON
                response.

            - ``<exec>``:
                Expects optional parameter payload for run compilation. Stores
                received params in ``run_data['params']`` and executes updated
                run via ``run_params(..., push=True, flag=True)``.

            - ``<reset>``:
                Emits ``reset(True, True)`` to restore user-selected baseline
                run configuration.

            - ``<check>``:
                Triggers lock-state verification via ``check_lock(flag=True)``.

            - ``<probe>``:
                Builds and pushes probe-mode digital segment onto current run.

            - ``<imaging>``:
                Builds and pushes imaging-mode digital segment onto current
                run.

            - ``<img_ref>``:
                Expects boolean payload ``fetch_back``. Captures a reference
                image, stores it in ``run_data['ref_imgs']``, queues done, and
                optionally returns image data as JSON.

            - ``<scan_img>``:
                Captures a sequence of absorption images, stores them in
                ``run_data['abs_imgs']``, computes observation vector using
                ``get_obs``, and returns JSON-encoded observation.

            - ``<jeitgem>``:
                Loads and pushes a predefined EIT/GEM run file.

            - ``<stop>``:
                Persists run data, tears down resources, and emits ``exit``.

        Notes:
            - Most actions that involve external hardware/network I/O are
              blocking in this implementation.
            - Responses to ML side are sent via ``send_queue`` and transmitted
              by ``comm_thread``.
        """
        flag = self.read_queue.get()
        # save system info used by the ml side
        if flag == '<sys_info>':
            new_sys_info = self.json_read()
            for k in new_sys_info:
                self.sys_info[k] = new_sys_info[k]

            print("\n>>> System Info Recieved")
            for k in self.sys_info:
                print(k, ':', self.sys_info[k])

        if flag == '<clr>':
            self.clear_lv_buffer(flag=True)

        # drop the MOT by sending a blank run
        if flag == '<drop>':
            self.drop(push=True, flag=True)
            print("\n>>> Dropped")

        # get trace
        if flag == '<acq>':
            num_traces = self.json_read()
            avg_lvl = self.acq_trace(num_traces, flag=True)
            self.send_queue.put(json.dumps(avg_lvl))
        
        if flag == '<acq_ref>':
            num_traces = self.json_read()
            avg_lvl = self.acq_trace(num_traces, flag=False)
            self.ref_trace_lvl = avg_lvl
            print('\n>>> Reference Trace Acquired')
            self.flag_done()
            self.send_queue.put(json.dumps(avg_lvl))

        # execute the latest run recieved, on LV
        if flag == '<exec>':
            params = self.json_read()
            if params is not None:
                self.run_data['params'].append(params)
                print('\n>>> Next Run Parameters Recieved:')
                print(params)
            else:
                print('\n>>> Running last recieved params')
            self.run_params(params, push=True, flag=True)

        # reset system to user defined config
        if flag == '<reset>':
            self.reset.emit(True, True)
            print('\n>>> System configured to user chosen run file')

        if flag == '<check>':
            self.check_lock(flag=True)

        if flag == '<probe>':
            self.make_run(mode='probe', push=True, plot_digital=True, flag=True)

        if flag == '<imaging>':
            self.make_run(mode='imaging', push=True, plot_digital=True, flag=True)
        
        if flag == '<img_ref>':
            fetch_back = self.json_read()
            self.current_img_ref = self.get_img()
            self.run_data['ref_imgs'].append(self.current_img_ref)
            print('\n>>> Reference Image Captured')
            self.flag_done()
            if fetch_back:
                self.send_queue.put(json.dumps(self.current_img_ref.tolist()))

        if flag == '<scan_img>':
            abs_imgs = self.scan_capture_imaging()
            self.run_data['abs_imgs'].append(abs_imgs)
            obs = self.get_obs(abs_imgs)
            print('\n>>> TOF Images Captured')
            self.flag_done()
            self.send_queue.put(json.dumps(obs.tolist()))

        # stop the tcp and threading server
        if flag == '<stop>':
            self.quit()
            self.exit.emit()

        if flag == '<jeitgem>':
            self.jeitgem(push=True, flag=True)


    def push_lv_run(self, data_dict, flag=False, plot_digital=False, push=True, plot=True):
        """Plot and/or push a run to downstream control workflow."""
        if plot:
            self.plot_ch.emit(data_dict, plot_digital)
        if push:
            self.push_run.emit(flag)
        if not flag:
            self.pause()

    def get_idxs(self, times, limits):
        """Find index bounds in a time vector for a given value interval."""
        L_idx = -1
        U_idx = -1
        for x, y in enumerate(times):
            if y > limits[0] and L_idx == -1:
                L_idx = x
            if y > limits[1]:
                U_idx = x
                break
        return L_idx, U_idx

    def compile_check_params(self, next_params, with_time=False):   # 5
        """Insert new parameter trajectories into active channel run template.

        Args:
            next_params: Parameter arrays grouped per controlled channel.
            with_time: Reserved for compatibility; currently unused.

        Returns:
            Updated channel list with values clipped to configured bounds.
        """
        # Assumed here the given params are of the format [[p1*time_bins], [p2*time_bins], ...]
        assert self.sys_info is not None, "Update required system info"
        demo_times = list(np.linspace(self.sys_info['min_time'], self.sys_info['max_time'], self.sys_info['time_bins']))
        assert self.sys_info['params'] * len(demo_times) == len(next_params[0]) * len(next_params), "Given Parameters are not of required length"
        all_channels = copy.deepcopy(self.next_run)
        for ch_params, (key, bound) in zip(next_params, self.sys_info['bounds'].items()):
            for i, chan in enumerate(all_channels):
                if chan['name'] == key:
                    # Create time-values list with updated values
                    temp_times = chan['points'][0]
                    temp_vals = chan['points'][1]
                    L_idx, U_idx = self.get_idxs(temp_times, (self.sys_info['min_time'], self.sys_info['max_time']))
                    new_times = temp_times[:L_idx] + demo_times + temp_times[U_idx:]
                    new_vals = temp_vals[:L_idx] + ch_params + temp_vals[U_idx:]
                    
                    # Checking bounds for the new values
                    new_vals = list(np.clip(new_vals, bound[0], bound[1]))

                    # Updating new values to channel dictionary
                    chan.update({'points': [new_times, new_vals]})

        self.next_run = all_channels
        self.base_run = all_channels
        return all_channels
    
    def make_run(self, active_time=(0.1201, 0.13), mode='probe', plot_digital=True, push=False, flag=False):
        """Build a probe/imaging run by grafting digital channel segments."""
        nr = copy.deepcopy(self.next_run)
        if mode == 'probe':
            pr = self.default_probe_run_dict
        elif mode == 'imaging':
            pr = self.default_imaging_run_dict
        
        chs = ['Trap Switch', 'Repump Switch', 'Trig', 'Optical Pumping']

        for p_ch, nr_ch in zip(pr, nr):
            if p_ch['name'] in chs:
                pr_times = p_ch['points'][0]
                pr_vals = p_ch['points'][1]
                nr_times = nr_ch['points'][0] 
                nr_vals = nr_ch['points'][1]
                pr_lidx, pr_uidx = self.get_idxs(pr_times, active_time)   
                nr_lidx, nr_uidx = self.get_idxs(nr_times, active_time)
                
                new_nr_vals = nr_vals[:nr_lidx] + pr_vals[pr_lidx:pr_uidx] + nr_vals[nr_uidx:]
                new_nr_times = nr_times[:nr_lidx] + pr_times[pr_lidx:pr_uidx] + nr_times[nr_uidx:]
                nr_ch.update({'points': [new_nr_times, new_nr_vals]})
        
        self.next_run = nr
        self.push_lv_run(nr, flag=flag, plot_digital=plot_digital, push=push, plot=True)

    def run_params(self, params=None, push=False, flag=False):  # 6
        """Apply parameter update and optionally push resulting run."""
        # update channels and push run with new params from ml
        if params is not None:
            self.next_params = params
            all_channels = self.compile_check_params(self.next_params)
        else:
            all_channels = copy.deepcopy(self.next_run)

        self.push_lv_run(all_channels, flag=flag, plot_digital=False, push=push, plot=True)

    def scan_capture_imaging(self):
        """Sweep imaging trigger times and capture corresponding images."""
        diffs = np.linspace(0, self.sys_info['img_window'], self.sys_info['img_frames'])
        imgs = []
        nr = copy.deepcopy(self.next_run)
        for d in diffs:
            # edit the trigger channel
            for ch in nr:
                if ch['name'] == self.sys_info['img_chan']:
                    ch['points'][0] = [0.0, float(self.sys_info['img_start_time'] +d), float(self.sys_info['img_start_time'] +d+ self.sys_info['img_pulse_duration']), 0.5]
                    ch['points'][1] = [0, 1, 0, 0]
                    ch['plot_points'][0] = []
                    ch['plot_points'][1] = []

            # push run and capture image
            self.push_lv_run(data_dict=nr, flag=False, plot_digital=True)    
            time.sleep(self.sys_info['img_sleep'])
            img = self.get_img()
            imgs.append(img)
        print('\n>>> Captured ', len(imgs), 'images')
        return imgs
    
    def get_img(self):
        """Capture one camera frame using configured trigger settings."""
        if self.cam is None:
            print('Camera not connected')
            return None
        
        img = self.cam.capture_on_trig(self.sys_info['cam_delay'], 
                                        exposure=self.sys_info['cam_exposure'], 
                                        pixel_clock=self.sys_info['cam_pixel_clock'], 
                                        frame_rate=self.sys_info['cam_frame_rate'], 
                                        black_level=0, show=False, 
                                        trigger_mode=self.sys_info['cam_trigger_mode'])
        return img
    
    def get_obs(self, abs_imgs):
        """Compute observation vector from absorption images and reference."""
        if self.current_img_ref is None:
            print('No reference image')
            return None
        ref_img = self.current_img_ref
        diff_imgs = []
        for img in abs_imgs:
            img = np.clip(img, 5, 255)/255
            ref_ = np.clip(ref_img, 5, 255)/255
            fixed_img = img/ np.max(img) 
            ref_ = ref_/ np.max(ref_)
            diff_imgs.append(np.log(ref_) - np.log(fixed_img))
            
        avg_imgs = np.mean(np.array(diff_imgs), axis=-1)
        # self.current_img_ref = None     # clearing out the reference image
        return np.concatenate(avg_imgs, axis=0)

    def get_avg_level(self, traces):
        """Compute mean level across a list of traces."""
        vals = []
        for trace in traces:
            vals.append(sum(trace)/len(trace))
        lvl = sum(vals)/len(vals)
        return lvl
    
    def get_cost(self, traces):
        """Compute log-ratio cost against stored probe reference."""
        if self.current_probe_ref is None:
            print('No reference trace')
            return None
        ref_trace = self.current_probe_ref
        avg_run_trace = self.get_avg_level(traces)
        cost = np.log(avg_run_trace/ ref_trace)
        self.current_probe_ref = None     # clearing out the reference trace
        return cost
    
    def lv_read(self, lv_flag=None):      ###############
        """Read a flagged response sequence from RunBuilder/LV server."""
        if lv_flag is not None:
            self.runbuilder.server.conn_send(lv_flag)

        # handling the flag
        while True:
            ret_flag = self.runbuilder.server.conn_read()
            if ret_flag != '' and ret_flag == 'READY':  ###############
                # print('Got Data')
                break
            else:
                time.sleep(0.1)

        # handling the data
        while True:
            ret_data = self.runbuilder.server.conn_read()
            if ret_data != '':
                print('Got data.')
                break
            else:
                time.sleep(0.1)
        return ret_data

    def acq_trace(self, num_traces, flag=False, shot_noise=1e-4, max_iter=15):
        """Acquire trace values until target count or iteration limit."""
        all_data = []
        last_lvl = 0.0
        while num_traces > 0 and max_iter > 0:
            data = self.lv_read('ACQ')
            if data is not None and data != '':
                lvl = json.loads(data)
                if np.abs(lvl - last_lvl) < shot_noise:
                    all_data.append(lvl)   ################ (to be fixed on LV side, input data incomplete)
                    num_traces -= 1
                last_lvl = lvl
            max_iter -= 1

                    
        print('\n>>> Trace Acquisition ')
        print('Collected ',len(all_data), 'traces')
        print(all_data)

        if flag: 
            self.flag_done()
        return all_data
        #self.send_queue.put(json.dumps(all_data))

    def drop(self, push=False, flag=False, blank_params=['Mag Switch']):
        """Generate a drop run by zeroing selected channel values."""
        drop_ch = copy.deepcopy(self.next_run)
        for chan in drop_ch:
            if chan['name'] in blank_params:
                chan['points'][1] = [0.0] * len(chan['points'][1])

        if not push:
            self.plot_ch.emit(drop_ch, True)
        if push:
            self.push_lv_run(drop_ch, flag, True)

    def _open_file(self, file_path):
        """Read and parse a JSON run/session file."""
        # open the file and read the data
        f = open(file_path, 'r')
        lines = ''
        for line in f: lines += line
        f.close()
        return json.loads(lines)
                           
    @Slot()
    def open_file(self, file_path, plot=True):
        """Load run file, update active run copy, and optionally plot."""
        data_dict = self._open_file(file_path)
        self.load_sess.emit(data_dict)
        self.run_from_file = data_dict['channels']
        self.next_run = copy.deepcopy(self.run_from_file)
        if plot:
            self.plot_ch.emit(self.run_from_file, False)

    def clear_lv_buffer(self, flag=False):
        """Drain pending items from LV server read buffer."""
        count = 0
        while True:
            data = self.runbuilder.server.conn_read()
            if data == '':
                break
            count += 1
        if count > 0:
            print('\n>>> Cleared',count,'items from LV send buffer')
        if flag:
            self.send_queue.put('<done>')

    def check_lock(self, flag=False, actonunlck=None):
        """Check laser lock status and handle unlocked condition policy."""
        self.tr_serv.conn_send('CHECK')
        time.sleep(0.3)
        status = self.tr_serv.conn_read()
        if status == 'LOCKED':
            if flag:
                self.send_queue.put('<done>')
            else:
                return
        elif status == 'UNLOCKED':
            print('\n>>> Laser(s) Lock Failed. Fix it and press ENTER to continue')
            
            if actonunlck == 'reset':
                self.reset(push=True)
            elif actonunlck == 'drop':
                self.drop(push=True)
            elif actonunlck == 'resetskip':
                self.reset(push=True)
                return

            key = input('Waiting...')
            self.check_lock(flag)
        else:
            print('Checking lock returned:', status)

    def chk_sys_anamoly(self):
        """Pause until manual resume when probe-lock anomaly is detected."""
        # check ref probe level to see if the cavity is still locked
        # pause the run until the user enters RESUME in terminal
        # chk is current ref trace is much below the ref trace at the run start
        print('\n>>> PROBE LASER LOCK FAILED. ENTER \'RESUME\' TO CONTINUE')
        while input() != 'RESUME':
            time.sleep(0.1)
        return

    def chk_trace_eq(self):
        """Placeholder for trace-equilibrium checks before data collection."""
        # check if the trace has reached an equilibrium
        # before collecting trace for data
        pass

    def jeitgem(self, push=True, flag=False):
        """Load a predefined EIT/GEM run and optionally push it."""
        self.open_file('./Runs/20241216_eit_gem')        
        if push:    
            self.push_lv_run(data_dict=self.run_from_file, plot_digital=False, flag=flag)

def save_bz2(data, filename='./data/run_data.bz2'):
    """Serialize and store object to bz2-compressed pickle file."""
    file = bz2.BZ2File(filename, 'wb')
    file.write(pickle.dumps(data, protocol=4))
    file.close()

def load_bz2(filename='./data/run_data.bz2'):
    """Load object from bz2-compressed pickle file."""
    file = bz2.BZ2File(filename, 'rb')
    dd = pickle.loads(file.read())
    file.close()
    return dd

def get_datetime():
    """Return current datetime string in MonDD_HH-MM format."""
    return str(datetime.now().strftime("%b%d_%H-%M"))

def topk(values, k):
    """Return top-k values and corresponding indices."""
    idxs = np.argpartition(np.array(values), -k)[-k:]
    topk_vals = values[idxs]
    return topk_vals, idxs
