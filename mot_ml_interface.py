"""ML-side interface for controlling MOT/RunBuilder workflows.

This module provides a client-side interface that connects to the
RunBuilder-side server, sends command flags, receives observations/traces,
and computes reward values for optimization loops.

Written by: Arindam Saha, 2025.
"""

import numpy as np
import logging 
import json, pickle
import copy
import time
import logging
import random
import bz2
from datetime import datetime
from tcp_server import Receiver

# FLAGS are sent before sending/ recieveing data for the flagged operation
# Flags Sent        Desc.
# 'exec'            execute on LV with the next run
# 'sys_info'        passing system info
# 'clr'             clear LV send buffer, to be done before running params
# 'drop'            drop the MOT by sending a blank run
# 'acq'             get trace from scope, followed by sending the number of traces
# 'reset'           reset system to user config
# Flags Recieved
# 'done'            operation done
# 'error'           some error occured

class BaseInterface(object):
    """Base class for experiment interfaces.

    Subclasses are expected to implement initialization and run-parameter
    execution methods.
    """

    def __init__(self):
        """Initialize a base interface placeholder."""
        pass

    def run_initialisation(self, params, args=None):
        """Run one-time initialization routine.

        Args:
            params: Initialization parameters.
            args: Optional extra arguments.
        """
        pass

    def run_parameters(self, params, args=None):
        """Execute one run with given parameters.

        Args:
            params: Run parameters.
            args: Optional extra arguments.
        """
        pass

class RunBuilderInterface(BaseInterface):
    """TCP client wrapper for MOT RunBuilder control and data acquisition.

    The class communicates with a RunBuilder-side service using flag-driven
    commands and JSON payloads. It can:
    - update system settings,
    - execute parameterized runs,
    - collect traces/images/observations,
    - compute rewards,
    - persist acquired data.
    """

    def __init__(self, address, port, sys_info=None, data_file=None, transformer=None):
        """Initialize connection settings and runtime state.

        Args:
            address: Server IP/hostname.
            port: Server port.
            sys_info: Optional system-configuration dict override.
            data_file: Optional output path for serialized run data.
            transformer: Optional callable to transform flat parameter vectors
                before reshaping into channel/time-bin format.
        """
        super(RunBuilderInterface, self).__init__()
        self.address = address
        self.port = port
        self.conn = None
        self.log = logging.getLogger('RunbuilderInterface')
        self.all_data = {'params':[], 'obs':[], 'rewards':[], 'ref_lvl':[], 'run_lvl':[]}
        if data_file is not None:
            self.data_file = data_file
        else:
            self.data_file = './data/mlrb_side/run_data_'+get_datetime()+'.bz2'
        self.connected = False
        self.use_transformer = False
        if sys_info is not None:
            self.sys_info = sys_info
        else:
            self.sys_info = {
                        'params': 3,
                        'time_bins': 21, 
                        'bounds' : {'Trap freq':(1.6, 8.1),
                                    'Repump freq':(2.4, 6.4),
                                    'Mag fields':(0.0, 6.0)},
                        'min_time': 0.10,
                        'max_time' : 0.12,
                        'detuning': 50,
                        'img_start_time': 0.1215,
                        'img_window': 6e-3,
                        'img_frames': 6,
                        'img_pulse_duration': 2e-3,
                        'img_chan': 'Trig',
                        'cam_delay': 0,
                        'cam_exposure': 0.3,
                        'cam_pixel_clock': 35,
                        'cam_frame_rate': 25.0,
                        'cam_trigger_mode': 'rising_edge',
                        'probe_sleep': 1,
                        'img_sleep': 5,
                        'noise_lvl': 1e-4
                        }
                    
        self.transformer = transformer
            
        self.connect()
        # self.update_sys_info()
        
    def connect(self):
        """Create listener client object and initiate TCP connection."""
        try:    
            self.conn = Receiver(self.address, self.port)
            self.conn.initiate_connection(self.address, self.port)
            self.connected = True
        except:
            print('Caught run time read error')
    
    def update_sys_info(self, sys_info=None):
        """Push system configuration to server side.

        Args:
            sys_info: Optional dict of keys to update in local ``self.sys_info``
                before transmission.
        """
        time.sleep(1)
        if sys_info is not None:
            for k in sys_info:
                self.sys_info[k] = sys_info[k]

        if self.connected:
            self.flag('sys_info', self.sys_info, wait=False)
        else:
            print('Not connected to the server')

    def close_conn(self):
        """Stop underlying connection listener thread."""
        self.conn.listening_thread.halt_set()

    def json_send(self, data):
        """Serialize payload to JSON and send via active connection."""
        try:
            self.conn.conn_send(json.dumps(data))
        except:
            print('Caught run time read error')

    def json_read(self):
        """Read one message and decode JSON when possible.

        Returns:
            Decoded JSON object when valid JSON is received, raw data
            otherwise, or ``None`` for empty reads.
        """
        # a flexible read solution for both json and non json files
        data = self.conn.conn_read()
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

    def run_parameters(self, params, check=False, img=True, save=True):
        """Execute a full run cycle and return reward/observation.

        Typical sequence:
            1. transform and reshape params,
            2. clear server buffer,
            3. switch imaging mode and execute run,
            4. acquire reference trace/image,
            5. scan TOF images and fetch observation,
            6. switch probe mode and acquire run trace,
            7. compute reward and optionally persist data.

        Args:
            params: Flat parameter list/array expected to map to
                ``sys_info['params'] * sys_info['time_bins']``.
            check: Whether to request lock check before execution.
            img: Reserved compatibility flag (currently unused).
            save: Whether to serialize accumulated ``all_data`` to disk.

        Returns:
            Tuple ``(reward, obs)``.
        """
        # assumed here that the params are just list of values
        self.all_data['params'].append(params)

        if self.transformer is not None:
            params = self.transformer(params)
            assert len(params) == self.sys_info['params']*self.sys_info['time_bins'], print('Transformer output length doesn\'t match sys_info')
        
        # [[param1 * time bins], [param2 * time bins], ...]
        params = np.array(params).reshape(self.sys_info['params'], -1).tolist()

        # clearing junks in LV send buffer
        self.flag('clr')

        if check:
            self.flag('check')

        self.flag('imaging')            # switches to imaging mode, sets proper trigger mode for ref acquisition
        self.flag('exec', params)       # updates next run with the params

        # drop and get the ref trace and img
        self.flag('drop')
        self.flag('acq_ref', 3)
        ref_trace = self.json_read()
        self.flag('img_ref')
        
        # start TOF imaging
        self.flag('imaging')
        time.sleep(self.sys_info['img_sleep'])  # to let the atoms settle
        self.flag('scan_img')
        obs = self.json_read()
        
        # send probe run and get absorbed trace
        self.flag('probe')
        time.sleep(self.sys_info['probe_sleep'])  # to let the atoms settle
        self.flag('acq', 5)
        run_trace = self.json_read()
        
        # calculate the cost
        reward = self.get_OD(run_trace, ref_trace) # to be minimised
        self.all_data['obs'].append(obs)
        self.all_data['ref_lvl'].append(ref_trace)
        self.all_data['rewards'].append(reward)
        self.all_data['run_lvl'].append(run_trace)
        self.all_data['params'].append(params)

        if save:
            for key in self.all_data:
                if isinstance(self.all_data[key], list):
                    self.all_data[key] = np.array(self.all_data[key])
            save_bz2(self.all_data, self.data_file)
    
        return reward, obs
    
    def get_OD(self, run_trace, ref_trace):
        """Compute negative optical-density proxy from traces.

        Args:
            run_trace: Trace(s) measured for current run.
            ref_trace: Reference trace(s).

        Returns:
            Negative OD-style scalar reward (to be minimized).
        """
        if isinstance(run_trace, list):
            run_trace = np.mean(run_trace)
        if isinstance(ref_trace, list):
            ref_trace = np.mean(ref_trace)
        detune = self.sys_info['detuning']
        x = 33.1/4.0
        od = (detune**2 + x)/x
        od *= np.log(run_trace/ref_trace)
        return -od
    
    # def get_absorption(self):
    #     self.flag('clr')
    #     self.flag('drop')
    #     self.flag('acq', 5)
    #     ref_trace = self.json_read()
    #     self.flag('reset')
    #     time.sleep(5)
    #     self.flag('acq', 5)
    #     run_trace = self.json_read()
    #     print(run_trace, ref_trace)
    #     return run_trace, ref_trace

    def reset(self, soft_reset=False, exe=True):
        """Generate random in-bound parameters and optionally execute them.

        Args:
            soft_reset: Reserved compatibility flag.
            exe: If ``True``, sends generated params via ``<exec>``.

        Returns:
            List of parameter trajectories shaped as
            ``[num_params][time_bins]``.
        """
        params = []
        for key in self.sys_info['bounds']:
            samples = [random.uniform(*self.sys_info['bounds'][key]) for _ in range(self.sys_info['time_bins'])]
            params.append(samples)

        if exe:
            self.flag('exec', params)
        return params

    
    def flag(self, args, data=None, wait=True):
        """Send a protocol flag with optional payload.

        Args:
            args: Flag name without angle brackets, e.g. ``'exec'``.
            data: Optional payload sent immediately after the flag.
            wait: Whether to block until ``<done>`` is received.

        Supported flags in this interface:
            - ``'sys_info'``: update server-side system configuration.
            - ``'clr'``: clear pending server buffer.
            - ``'check'``: verify lock status before run operations.
            - ``'imaging'``: switch run template to imaging mode.
            - ``'probe'``: switch run template to probe mode.
            - ``'exec'``: execute run with provided params.
            - ``'drop'``: push drop run.
            - ``'acq_ref'``: acquire reference traces.
            - ``'img_ref'``: capture reference image.
            - ``'scan_img'``: capture imaging scan and send observation.
            - ``'acq'``: acquire run traces.
            - ``'reset'``: restore user configuration on server side.
            - ``'stop'``: request server shutdown.
        """
        self.conn.conn_send('<'+args+'>')
        if data is not None:
            self.json_send(data)
        if wait:
            self.wait()

    def wait(self):
        """Block until server responds with ``<done>``."""
        # wait till the operation on server side sends back '<done>'
        while True:
            flag = self.conn.conn_read()
            if flag == '<done>': break
            else: time.sleep(0.1)
        time.sleep(0.2)
        
        
def save_bz2(data, filename='./data/run_data.bz2'):
    """Serialize and store object to bz2-compressed pickle file.

    Args:
        data: Python object to serialize.
        filename: Output file path.
    """
    file = bz2.BZ2File(filename, 'wb')
    file.write(pickle.dumps(data, protocol=4))
    file.close()

def load_bz2(filename='./data/run_data.bz2'):
    """Load object from bz2-compressed pickle file.

    Args:
        filename: Input file path.

    Returns:
        Deserialized Python object.
    """
    file = bz2.BZ2File(filename, 'rb')
    dd = pickle.loads(file.read())
    file.close()
    return dd

def get_datetime():
    """Return current datetime string in ``MonDD_HH-MM`` format."""
    return str(datetime.now().strftime("%b%d_%H-%M"))
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(module)s:%(message)s")
    rb_interface = RunBuilderInterface(address='127.0.0.1', port=7777)
    input('>')
    