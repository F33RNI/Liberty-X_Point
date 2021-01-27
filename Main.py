import os
import threading
import time

import cv2
import serial
import sys
import glob
from flask import Flask, render_template, Response, redirect, request
import socket
import numpy as np

import Worker

app = Flask(__name__)
worker = Worker.Worker(None, '', '', '')


@app.route('/')
def index():
    global worker
    if worker is None or worker.camera_id is None:
        cnc_ports = []
        rf_ports = []
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                cnc_ports.append(port)
                rf_ports.append(port)
            except (OSError, serial.SerialException):
                pass
        return render_template('index.html', cnc_ports=cnc_ports, rf_ports=rf_ports)
    elif worker.aborting:
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Server stopped.')
        func()
        print('Aborted')
        sys.stdout.flush()
        os._exit(0)
    else:
        time.sleep(0.1)
        pid_roll = {'p': worker.pid_p_gain_roll, 'i': worker.pid_i_gain_roll,
                    'd': worker.pid_d_gain_roll}
        pid_alt = {'p': worker.pid_p_gain_throttle, 'i': worker.pid_i_gain_throttle,
                   'd': worker.pid_d_gain_throttle}
        debug_info = {'d': 'checked' if worker.debug else '', 'v': 'checked' if worker.video else '',
                      'o': 'checked' if worker.frame_enabled else ''}
        holding_drone = 'checked' if worker.enable_stabilization else ''
        landing_drone = 'checked' if worker.start_landing else ''
        return render_template('controller.html', pid_roll=pid_roll, pid_alt=pid_alt, debug_info=debug_info,
                               holding_drone=holding_drone, landing_drone=landing_drone)


@app.route('/start/<camera_id>/<camera_exp>/<string:cnc_port>/<string:rf_port>/<string:udp_ip_port>/<landing_alt>/')
def start_loop(camera_id, camera_exp, cnc_port, rf_port, udp_ip_port, landing_alt):
    global worker
    if cnc_port == 'none':
        cnc_port = ''
    if rf_port == 'none':
        rf_port = ''
    if udp_ip_port == 'none':
        udp_ip_port = ''

    print('----------------------------')
    print('CNC port: ' + cnc_port)
    print('RF port: ' + rf_port)
    print('UDP IP/PORT: ' + udp_ip_port)
    print('Camera ID: ' + camera_id)
    print('----------------------------')

    try:
        worker = Worker.Worker(int(camera_id), cnc_port, rf_port, udp_ip_port)
        worker.camera_exposure = int(camera_exp)
        worker.landing_motor_off_altitude = int(landing_alt)
    except:
        print("Unexpected error: ", sys.exc_info())
        raise
    return redirect('/', code=302)


@app.route('/setup/<pid_roll_p>/<pid_roll_i>/<pid_roll_d>/<pid_alt_p>/<pid_alt_i>/<pid_alt_d>/'
           '<debug_enabled>/<video_enabled>/<frame_enabled>/')
def setup_values(pid_roll_p, pid_roll_i, pid_roll_d, pid_alt_p, pid_alt_i, pid_alt_d,
                 debug_enabled, video_enabled, frame_enabled):
    global worker
    worker.pid_p_gain_roll = float(pid_roll_p)
    worker.pid_i_gain_roll = float(pid_roll_i)
    worker.pid_d_gain_roll = float(pid_roll_d)
    worker.pid_p_gain_throttle = float(pid_alt_p)
    worker.pid_i_gain_throttle = float(pid_alt_i)
    worker.pid_d_gain_throttle = float(pid_alt_d)
    worker.debug = True if debug_enabled == 'true' else False
    worker.video = True if video_enabled == 'true' else False
    worker.frame_enabled = True if frame_enabled == 'true' else False
    settings = [worker.pid_p_gain_roll, worker.pid_i_gain_roll, worker.pid_d_gain_roll,
                worker.pid_p_gain_throttle, worker.pid_i_gain_throttle, worker.pid_d_gain_throttle]
    np.save('pid', settings)
    return redirect('/', code=302)


@app.route('/hold_land/<holding>/<landing>/')
def hold_land(holding, landing):
    global worker
    if holding == 'true':
        worker.enable_stabilization = True
    else:
        worker.enable_stabilization = False
        worker.start_landing = False
        worker.ddc_service_info = 0

    if landing == 'true':
        worker.enable_stabilization = True
        worker.start_landing = True
    else:
        worker.start_landing = False
        worker.ddc_service_info = 0
    return redirect('/', code=302)


@app.route('/abort')
def abort():
    global worker
    print('Aborting...')

    worker.enable_stabilization = False
    worker.start_landing = False
    worker.ddc_service_info = 0
    time.sleep(0.1)
    worker.frame_enabled = False
    worker.video = False
    time.sleep(0.1)

    worker.opencv_running = False
    worker.main_loop_running = False
    worker.rf_loop_running = False
    time.sleep(0.1)

    cv2.destroyAllWindows()

    # noinspection PyBroadException
    try:
        if worker.serial_cnc_port.isOpen():
            worker.serial_cnc_port.flush()
            worker.serial_cnc_port.close()
    except:
        pass

    # noinspection PyBroadException
    try:
        if worker.serial_rf_port.isOpen():
            worker.serial_rf_port.flush()
            worker.serial_rf_port.close()
    except:
        pass
    worker.aborting = True
    return redirect('/', code=302)


def gen():
    global worker
    while True:
        if worker.frame_debug is not None and worker.video:
            (flag, encoded_image) = cv2.imencode(".jpg", worker.frame_debug)
            if not flag:
                continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encoded_image) + b'\r\n')
        else:
            break


@app.route('/video_feed')
def video_feed():
    if worker.frame_debug is not None and worker.video:
        new_response = Response(gen(),
                                mimetype='multipart/x-mixed-replace; boundary=frame')
        new_response.headers.add('Access-Control-Allow-Origin', '*')
        new_response.headers.add('Cache-Control', 'no-cache')
        return new_response
    else:
        return '', 204


if __name__ == '__main__':
    app.run(host=socket.gethostbyname(socket.gethostname()), port='5000', debug=False)
