import math
import threading
import time
import os.path

import cv2
import numpy as np
import serial
import socket
from cv2 import aruco


def valmap(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


def rotation_matrix_to_euler_angles(r):
    def is_rotation_matrix(r):
        Rt = np.transpose(r)
        shouldBeIdentity = np.dot(Rt, r)
        I = np.identity(3, dtype=r.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    assert (is_rotation_matrix(r))

    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0

    return np.array([x, y, z])


class Worker:
    def __init__(self, camera_id, cnc_port, rf_port, udp_ip_port):
        self.camera_id = camera_id
        if self.camera_id is not None:
            self.video = False
            self.debug = True
            self.frame_enabled = True
            self.aborting = False

            # Periphery
            self._R_flip = np.zeros((3, 3), dtype=np.float32)
            self.camera_id = camera_id
            self.cnc_port = cnc_port
            self.rf_port = rf_port
            self.udp_ip_port = udp_ip_port
            self.camera_exposure = -9
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # OpenCV
            self.frame_width = 1280
            self.frame_height = 720
            self.frame = None
            self.frame_debug = None
            self.video_fps = 0
            self.loop_fps = 0
            # Geometry
            # KNOWN_DISTANCE = 100.0
            # KNOWN_PIXELS = 42.0  # 1280*720
            # self.size_koeff = KNOWN_DISTANCE * KNOWN_PIXELS

            self.MARKER_SIZE_CM = 5.00
            self.CAMERA_MATRIX = np.loadtxt('calibration/cameraMatrix.txt', delimiter=',')
            self.CAMERA_DISTORTION = np.loadtxt('calibration/cameraDistortion.txt', delimiter=',')

            # Threads
            self.opencv_running = True
            self.main_loop_running = True
            self.rf_loop_running = True
            # PIDs
            # Roll
            self.pid_p_gain_roll = 0.00
            self.pid_i_gain_roll = 0.00
            self.pid_d_gain_roll = 0.00
            self.pid_max_roll = 300
            # Pitch
            self.pid_p_gain_pitch = self.pid_p_gain_roll
            self.pid_i_gain_pitch = self.pid_i_gain_roll
            self.pid_d_gain_pitch = self.pid_d_gain_roll
            self.pid_max_pitch = self.pid_max_roll
            # Yaw
            self.pid_p_gain_yaw = 14.0
            self.pid_max_yaw = 100
            # Trottle
            self.pid_p_gain_throttle = 0.00
            self.pid_i_gain_throttle = 0.00
            self.pid_d_gain_throttle = 0.00
            self.pid_max_throttle = 300

            # PID From file
            settings = [self.pid_p_gain_roll, self.pid_i_gain_roll, self.pid_d_gain_roll,
                        self.pid_p_gain_throttle, self.pid_i_gain_throttle, self.pid_d_gain_throttle]
            settings = np.array(settings)
            settings_file = 'pid.npy'
            if os.path.isfile(settings_file):
                settings = np.load(settings_file)
                self.pid_p_gain_roll = settings[0]
                self.pid_i_gain_roll = settings[1]
                self.pid_d_gain_roll = settings[2]
                self.pid_p_gain_throttle = settings[3]
                self.pid_i_gain_throttle = settings[4]
                self.pid_d_gain_throttle = settings[5]
            else:
                np.save('pid', settings)

            # Direct Drone Control
            self.ddc_roll_error = 0
            self.ddc_pitch_error = 0
            self.ddc_throttle_error = 0
            self.angle_error = 0
            self.ddc_roll_output = 1500
            self.ddc_pitch_output = 1500
            self.ddc_yaw_output = 1500
            self.ddc_throttle_output = 1500
            self.ddc_z_setpoint = 100.0
            self.landing_motor_off_altitude = 10.0
            self.ddc_service_info = 0  # 0 - Nothing to do, 1 - Stabilization, 2 - Landind, 3 - Disable motors
            # Main form controlled
            self.leds_brightness = 50
            self.enable_stabilization = False
            self.start_landing = False
            self.landing_allowed = False

            self.marker_entered_x = 0
            self.marker_entered_y = 0

            self.marker_x = 0
            self.marker_y = 0
            self.marker_z = 0
            self.marker_angle = 0

            self.tracking_stage = 0  # 0 - No tracking, 1 - ARUco tracking, 2 - Init tracker, 3 - Tracker
            self.drone_observed = False
            print('[MAIN] Opening serial ports...')

            if len(self.cnc_port) > 0:
                self.serial_cnc_port = serial.Serial(cnc_port, 57600)
                self.serial_cnc_port.close()
                self.serial_cnc_port.open()
                print('[MAIN] CNC Serial is open: ' + str(self.serial_cnc_port.isOpen()))

            if len(self.rf_port) > 0:
                self.serial_rf_port = serial.Serial(rf_port, 57600, timeout=None, xonxoff=0, rtscts=0)
                self.serial_rf_port.close()
                self.serial_rf_port.open()
                print('[MAIN] RF Serial is open: ' + str(self.serial_rf_port.isOpen()))

            if len(self.udp_ip_port) > 0:
                sock_udp_ip = self.udp_ip_port.split(':')[0]
                sock_udp_port = int(self.udp_ip_port.split(':')[1])
                self.sock.connect((sock_udp_ip, sock_udp_port))
                print('[MAIN] Connected to: ' + str(sock_udp_ip) + ':' + str(sock_udp_port))

            print('[CNC] Waiting for ready...')
            # while self.serial_cnc_port.read() != b'>':
            #    pass
            # print('[CNC] Centering camera...')
            # self.serial_cnc_port.write(b'G0 X80 Y250\n')
            # while self.serial_cnc_port.read() != b'>':
            #    pass
            if len(self.cnc_port) > 0:
                print('[CNC] Enabling LEDs...')
                self.serial_cnc_port.write(b'M3 P1\n')

            thread = threading.Thread(target=self.opencv_video)
            thread.start()
            thread = threading.Thread(target=self.main_loop)
            thread.start()
            thread = threading.Thread(target=self.drone_direct_controller)
            thread.start()

    def drone_direct_controller(self):

        pid_i_mem_roll = 0
        pid_last_roll_d_error = 0
        pid_i_mem_pitch = 0
        pid_last_pitch_d_error = 0
        pid_i_mem_throttle = 0
        pid_last_throttle_d_error = 0

        while self.rf_loop_running:
            check_byte = 0
            if self.tracking_stage == 0:
                self.ddc_roll_output = 1500
                self.ddc_pitch_output = 1500
                self.ddc_yaw_output = 1500
                self.ddc_throttle_output = 1500
                pid_i_mem_roll = 0
                pid_last_roll_d_error = 0
                pid_i_mem_pitch = 0
                pid_last_pitch_d_error = 0
                pid_i_mem_throttle = 0
                pid_last_throttle_d_error = 0
            else:
                # PID controller
                # Pre-calcutations
                self.pid_p_gain_pitch = self.pid_p_gain_roll
                self.pid_i_gain_pitch = self.pid_i_gain_roll
                self.pid_d_gain_pitch = self.pid_d_gain_roll
                self.pid_max_pitch = self.pid_max_roll

                # Yaw
                self.ddc_yaw_output = (self.angle_error * self.pid_p_gain_yaw) * -1
                if self.ddc_yaw_output > self.pid_max_yaw:
                    self.ddc_yaw_output = self.pid_max_yaw
                elif self.ddc_yaw_output < self.pid_max_yaw * -1:
                    self.ddc_yaw_output = self.pid_max_yaw * -1

                # Altitude (throttle)
                pid_i_mem_throttle += self.pid_i_gain_throttle * self.ddc_throttle_error
                if pid_i_mem_throttle > self.pid_max_throttle:
                    pid_i_mem_throttle = self.pid_max_throttle
                elif pid_i_mem_throttle < self.pid_max_throttle * -1:
                    pid_i_mem_throttle = self.pid_max_throttle * -1

                self.ddc_throttle_output = self.pid_p_gain_throttle * self.ddc_throttle_error + pid_i_mem_throttle
                self.ddc_throttle_output += self.pid_d_gain_throttle * (
                        self.ddc_throttle_error - pid_last_throttle_d_error)
                if self.ddc_throttle_output > self.pid_max_throttle:
                    self.ddc_throttle_output = self.pid_max_throttle
                elif self.ddc_throttle_output < self.pid_max_throttle * -1:
                    self.ddc_throttle_output = self.pid_max_throttle * -1
                pid_last_throttle_d_error = self.ddc_throttle_error

                # Roll
                pid_i_mem_roll += self.pid_i_gain_roll * self.ddc_roll_error
                if pid_i_mem_roll > self.pid_max_roll:
                    pid_i_mem_roll = self.pid_max_roll
                elif pid_i_mem_roll < self.pid_max_roll * -1:
                    pid_i_mem_roll = self.pid_max_roll * -1

                ddc_roll_output_raw = self.pid_p_gain_roll * self.ddc_roll_error + pid_i_mem_roll + \
                                      self.pid_d_gain_roll * (self.ddc_roll_error - pid_last_roll_d_error)
                if ddc_roll_output_raw > self.pid_max_roll:
                    ddc_roll_output_raw = self.pid_max_roll
                elif ddc_roll_output_raw < self.pid_max_roll * -1:
                    ddc_roll_output_raw = self.pid_max_roll * -1
                pid_last_roll_d_error = self.ddc_roll_error

                # Pitch
                pid_i_mem_pitch += self.pid_i_gain_pitch * self.ddc_pitch_error
                if pid_i_mem_pitch > self.pid_max_pitch:
                    pid_i_mem_pitch = self.pid_max_pitch
                elif pid_i_mem_pitch < self.pid_max_pitch * -1:
                    pid_i_mem_pitch = self.pid_max_pitch * -1

                ddc_pitch_output_raw = self.pid_p_gain_pitch * self.ddc_pitch_error + pid_i_mem_pitch + \
                                       self.pid_d_gain_pitch * (self.ddc_pitch_error - pid_last_pitch_d_error)
                if ddc_pitch_output_raw > self.pid_max_pitch:
                    ddc_pitch_output_raw = self.pid_max_pitch
                elif ddc_pitch_output_raw < self.pid_max_pitch * -1:
                    ddc_pitch_output_raw = self.pid_max_pitch * -1

                pid_last_pitch_d_error = self.ddc_pitch_error

                # Pitch and Roll angle correction
                marker_angle_sin = math.sin(math.radians(self.marker_angle))
                marker_angle_cos = math.cos(math.radians(self.marker_angle))

                self.ddc_pitch_output = ddc_pitch_output_raw * marker_angle_cos \
                                        - ddc_roll_output_raw * marker_angle_sin
                self.ddc_roll_output = ddc_pitch_output_raw * marker_angle_sin \
                                       + ddc_roll_output_raw * marker_angle_cos

                self.ddc_roll_output += 1500
                self.ddc_pitch_output += 1500
                self.ddc_yaw_output += 1500
                self.ddc_throttle_output += 1500
                self.ddc_roll_output = int(self.ddc_roll_output)
                self.ddc_pitch_output = int(self.ddc_pitch_output)
                self.ddc_yaw_output = int(self.ddc_yaw_output)
                self.ddc_throttle_output = int(self.ddc_throttle_output)

            transmitt_buffer = [0, ord(b'L'), ord(b'X'), (int(self.ddc_roll_output) >> 8) & 0xFF,
                                int(self.ddc_roll_output) & 0xFF, (int(self.ddc_pitch_output) >> 8) & 0xFF,
                                int(self.ddc_pitch_output) & 0xFF, (int(self.ddc_yaw_output) >> 8) & 0xFF,
                                int(self.ddc_yaw_output) & 0xFF, (int(self.ddc_throttle_output) >> 8) & 0xFF,
                                int(self.ddc_throttle_output) & 0xFF, int(self.ddc_service_info) & 0xFF]

            for i in range(3, 12):
                check_byte ^= transmitt_buffer[i]
            transmitt_buffer.append(check_byte)
            transmitt_buffer.append(0)
            if self.enable_stabilization:
                if len(self.rf_port) > 0:
                    self.serial_rf_port.write(transmitt_buffer)
                if len(self.udp_ip_port) > 0:
                    self.sock.send(bytearray(transmitt_buffer))

            time.sleep(0.030)

    def opencv_video(self):
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, self.camera_exposure)  # - 10 (Microsoft)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        fps_seconds = 1
        fps_counter = 1
        fps_start = 0
        while self.opencv_running:
            ret, self.frame = cap.read()
            if self.frame is None:
                break
            fps_counter += 1
            if fps_counter > 60:
                fps_end = time.time()
                fps_seconds = fps_seconds * 0 + (fps_end - fps_start) * 1
                self.video_fps = int(61 / fps_seconds)
                print('[Camera capture] Video FPS: ' + str(self.video_fps))
                print('[Camera capture] Loop FPS: ' + str(self.loop_fps))
                fps_counter = 0
                fps_start = time.time()
        cap.release()

    def main_loop(self):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        font = cv2.FONT_HERSHEY_PLAIN
        fps_seconds = 1
        fps_counter = 1
        fps_start = 0
        leds_brighness_last = 50
        aruco_abort_counter = 0
        frame_enabled_last = self.frame_enabled
        while self.main_loop_running:
            if self.frame is not None:
                # Change LEDs Brighness
                if leds_brighness_last != self.leds_brightness:
                    leds_brighness_last = self.leds_brightness
                    if len(self.cnc_port) > 0:
                        print('[CNC] Changing brighness...')
                        self.serial_cnc_port.write(b'M3 P' + str(int(self.leds_brightness * 2.55)).encode() + b'\n')

                debug_frame = self.frame.copy()
                # print(debug_frame.shape)
                # debug_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2BGRA)
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                aruco_tracking = False
                # corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict,
                                                             parameters=parameters,
                                                             cameraMatrix=self.CAMERA_MATRIX,
                                                             distCoeff=self.CAMERA_DISTORTION)
                if np.all(ids is not None):
                    # aruco.drawDetectedMarkers(debug_frame, corners)
                    if ids.size == 1 and ids[0] == 9:

                        # self.marker_angle = int(math.degrees(math.acos(vec_2_y / d_2)))
                        # if vec_2_x > 0:
                        #     self.marker_angle = 360 - self.marker_angle
                        # print(1 if vec_2_y > 0 else 0)
                        # if vec_2_y < 0:
                        #     self.marker_angle = 360 - self.marker_angle

                        ret = aruco.estimatePoseSingleMarkers(corners, self.MARKER_SIZE_CM, self.CAMERA_MATRIX,
                                                              self.CAMERA_DISTORTION)

                        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
                        self._R_flip[0, 0] = 1.0
                        self._R_flip[1, 1] = -1.0
                        self._R_flip[2, 2] = -1.0

                        self.marker_x = tvec[0]
                        self.marker_y = tvec[1]
                        self.marker_z = tvec[2]

                        R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
                        R_tc = R_ct.T

                        # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
                        roll_marker, pitch_marker, yaw_marker = rotation_matrix_to_euler_angles(self._R_flip * R_tc)
                        yaw_marker += math.radians(90)

                        self.marker_angle = math.degrees(yaw_marker)
                        if self.marker_angle > 180:
                            self.marker_angle = self.marker_angle - 360

                        if self.marker_z < 200:
                            aruco_tracking = True

                # Define tracking stage
                self.landing_allowed = False
                if not self.enable_stabilization:
                    self.tracking_stage = 0
                elif aruco_tracking:
                    self.tracking_stage = 1  # ARUco tracking
                    aruco_abort_counter = 0
                else:
                    self.angle_error = 0
                    aruco_abort_counter += 1
                    if aruco_abort_counter > 30:
                        self.tracking_stage = 0  # ARUco lost
                    elif self.tracking_stage != 2 and self.tracking_stage != 3:
                        self.tracking_stage = 2  # Re-init tracker

                # Tracker Re-initialization
                if self.tracking_stage == 2:
                    # tracker = TRACKER()
                    # tracker.init(self.frame, tracking_bounding_box)
                    self.tracking_stage = 3

                # If no correct ARUcos found
                # if self.tracking_stage == 2 or self.tracking_stage == 3:
                #    print('[ARUCO] Marker lost')
                # Nothing to do, just wait

                # Calculate DDC values
                if self.tracking_stage == 1 or self.tracking_stage == 3:

                    self.ddc_roll_error = self.ddc_roll_error * 0.7 + self.marker_y * 0.3
                    self.ddc_pitch_error = self.ddc_pitch_error * 0.7 + self.marker_x * 0.3
                    self.ddc_throttle_error = self.ddc_throttle_error * 0.7 \
                                              + (self.ddc_z_setpoint - self.marker_z) * 0.3
                    self.angle_error = self.angle_error * 0.7 + self.marker_angle * 0.3

                    if self.start_landing and abs(self.ddc_roll_error) < 5 and abs(self.ddc_pitch_error) < 5 \
                            and abs(self.angle_error) < 7 and abs(self.ddc_throttle_error) < 5:
                        self.landing_allowed = True
                    if self.ddc_z_setpoint - self.ddc_throttle_error < self.landing_motor_off_altitude:
                        self.landing_allowed = True

                    if not self.drone_observed:
                        self.ddc_z_setpoint = self.marker_z
                    if self.start_landing and self.landing_allowed:
                        if self.ddc_z_setpoint - self.ddc_throttle_error > 100:
                            self.ddc_z_setpoint -= 0.15
                        elif self.ddc_z_setpoint - self.ddc_throttle_error > self.landing_motor_off_altitude:
                            self.ddc_z_setpoint -= 0.07
                        self.ddc_service_info = 2
                        if self.ddc_z_setpoint - self.ddc_throttle_error < self.landing_motor_off_altitude:
                            self.ddc_service_info = 3
                    else:
                        self.ddc_service_info = 1
                    self.drone_observed = True
                else:
                    self.ddc_roll_error = 0
                    self.ddc_pitch_error = 0
                    self.ddc_throttle_error = 0
                    self.angle_error = 0
                    self.drone_observed = False

                # Debug information
                if (self.video or self.frame_enabled) and self.debug:
                    main_color = (0, 0, 0)
                    blk = np.zeros(debug_frame.shape, np.uint8)
                    cv2.rectangle(blk, (0, 0), (250, 480), (255, 255, 255), cv2.FILLED)
                    debug_frame = cv2.addWeighted(debug_frame, 1.0, blk, 0.25, 1)

                    cv2.putText(debug_frame, 'Video FPS: ' + str(self.video_fps), (0, 20),
                                font, 1, main_color, 1, cv2.LINE_AA)
                    cv2.putText(debug_frame, 'Loop FPS: ' + str(self.loop_fps), (0, 40),
                                font, 1, main_color, 1, cv2.LINE_AA)
                    cv2.putText(debug_frame, 'Marker X: ' + str(int(self.marker_x)), (0, 60),
                                font, 1, main_color, 1, cv2.LINE_AA)
                    cv2.putText(debug_frame, 'Marker Y: ' + str(int(self.marker_y)), (0, 80),
                                font, 1, main_color, 1, cv2.LINE_AA)
                    cv2.putText(debug_frame, 'Marker Z: ' + str(int(self.marker_z)), (0, 100),
                                font, 1, main_color, 1, cv2.LINE_AA)
                    cv2.putText(debug_frame, 'Marker Angle (DEG): ' + str(int(self.marker_angle)), (0, 120),
                                font, 1, main_color, 1, cv2.LINE_AA)

                    cv2.putText(debug_frame, 'DDC Roll Output: ' + str(self.ddc_roll_output), (0, 160),
                                font, 1, main_color, 1, cv2.LINE_AA)
                    cv2.putText(debug_frame, 'DDC Pitch Output: ' + str(self.ddc_pitch_output), (0, 180),
                                font, 1, main_color, 1, cv2.LINE_AA)
                    cv2.putText(debug_frame, 'DDC Yaw Output: ' + str(self.ddc_yaw_output), (0, 200),
                                font, 1, main_color, 1, cv2.LINE_AA)
                    cv2.putText(debug_frame, 'DDC Throttle Output: ' + str(self.ddc_throttle_output), (0, 220),
                                font, 1, main_color, 1, cv2.LINE_AA)

                    circle_x = int(valmap(self.ddc_roll_error, 150, -150, 0, 180))
                    circle_y = int(valmap(self.ddc_pitch_error, 200, -200, 0, 230) + 250)
                    altitude_y = int(valmap(self.ddc_z_setpoint - self.ddc_throttle_error, 0, 200, 230, 0) + 250)
                    pitch_prop = valmap(self.ddc_pitch_output, 1100, 1900, -50, 50)
                    roll_prop = valmap(self.ddc_roll_output, 1100, 1900, -50, 50)
                    line_x_rotated_pitch = int(circle_x - (pitch_prop * math.sin(math.radians(self.marker_angle))))
                    line_y_rotated_pitch = int(circle_y + (pitch_prop * math.cos(math.radians(self.marker_angle))))
                    line_x_rotated = int(circle_x - (-40 * math.sin(math.radians(self.marker_angle))))
                    line_y_rotated = int(circle_y + (-40 * math.cos(math.radians(self.marker_angle))))
                    line_x_rotated_roll = int(circle_x + (roll_prop * math.cos(math.radians(self.marker_angle))))
                    line_y_rotated_roll = int(circle_y + (roll_prop * math.sin(math.radians(self.marker_angle))))

                    # Yaw
                    if self.ddc_yaw_output > 1500:
                        cv2.ellipse(debug_frame, (circle_x, circle_y), (15, 15), 0, 110, 250, (255, 128, 0), 2)
                        cv2.circle(debug_frame, (circle_x - 4, circle_y - 15), 3, (255, 128, 0), -1)
                    elif self.ddc_yaw_output < 1500:
                        cv2.ellipse(debug_frame, (circle_x, circle_y), (15, 15), 0, 70, -70, (0, 128, 255), 2)
                        cv2.circle(debug_frame, (circle_x + 4, circle_y - 15), 3, (0, 128, 255), -1)

                    # Vectors
                    cv2.line(debug_frame, (circle_x, circle_y), (line_x_rotated_pitch, line_y_rotated_pitch),
                             (200, 0, 200), 2)
                    cv2.line(debug_frame, (circle_x, circle_y), (line_x_rotated_roll, line_y_rotated_roll),
                             (0, 200, 200), 2)
                    cv2.line(debug_frame, (circle_x, circle_y), (line_x_rotated, line_y_rotated),
                             (0, 0, 0), 1)

                    pitch_y = int(valmap(self.ddc_pitch_output, 1100, 1900, 0, 230) + 250)
                    throttle_y = int(valmap(self.ddc_throttle_output, 1100, 1900, 230, 0) + 250)
                    roll_x = int(valmap(self.ddc_roll_output, 1100, 1900, 0, 180))

                    cv2.circle(debug_frame, (circle_x, circle_y), 4,
                               (120, 255, 0) if abs(self.ddc_roll_error) < 2
                                                and abs(self.ddc_pitch_error) < 2
                                                and abs(self.angle_error) < 5 else (0, 0, 0), -1)
                    cv2.circle(debug_frame, (215, altitude_y), 4,
                               (120, 255, 0) if abs(self.ddc_throttle_error) < 2 else (0, 0, 0), -1)

                    # Roll / Pitch / Throttle controllers
                    cv2.circle(debug_frame, (175, pitch_y), 5, (200, 0, 200), -1)
                    cv2.circle(debug_frame, (175, 365), 5, (200, 0, 200), 1)
                    cv2.line(debug_frame, (175, 250), (175, 480), (200, 0, 200), 1)
                    cv2.circle(debug_frame, (roll_x, 475), 5, (200, 200, 0), -1)
                    cv2.circle(debug_frame, (90, 475), 5, (200, 200, 0), 1)
                    cv2.line(debug_frame, (0, 475), (180, 475), (200, 200, 0), 1)
                    cv2.circle(debug_frame, (245, throttle_y), 5, (0, 200, 200), -1)
                    cv2.circle(debug_frame, (245, 365), 5, (0, 200, 200), 1)
                    cv2.line(debug_frame, (245, 250), (245, 480), (0, 200, 200), 1)

                    # Simulator box
                    cv2.rectangle(debug_frame, (0, 250), (180, 480), (0, 0, 0), 1)
                    cv2.rectangle(debug_frame, (0, 250), (250, 480), (0, 0, 0), 1)
                    if self.tracking_stage == 0:
                        cv2.putText(debug_frame, 'Not tracking', (0, 240),
                                    font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        aruco.drawDetectedMarkers(debug_frame, corners)

                        if self.ddc_service_info == 3:
                            cv2.putText(debug_frame, 'LANDED', (0, 240),
                                        font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        elif self.start_landing and self.landing_allowed and self.ddc_service_info == 2:
                            cv2.putText(debug_frame, 'LANDING...', (0, 240),
                                        font, 1, (0, 255, 255), 1, cv2.LINE_AA)
                        elif self.start_landing and not self.ddc_service_info == 2:
                            cv2.putText(debug_frame, 'LANDING NOT ALLOWED', (0, 240),
                                        font, 1, (0, 128, 255), 1, cv2.LINE_AA)
                        else:
                            cv2.putText(debug_frame, 'Tracking stage: ' + str(self.tracking_stage), (0, 240),
                                        font, 1, main_color, 1, cv2.LINE_AA)

                if self.video:
                    self.frame_debug = debug_frame
                else:
                    self.frame_debug = None

                if self.frame_enabled:
                    if not frame_enabled_last:
                        frame_enabled_last = True
                    cv2.imshow('Liberty-X landing controller', debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                elif frame_enabled_last:
                    cv2.destroyAllWindows()

                fps_counter += 1
                if fps_counter > 30:
                    fps_end = time.time()
                    fps_seconds = fps_seconds * 0 + (fps_end - fps_start) * 1
                    self.loop_fps = int(31 / fps_seconds)
                    fps_counter = 0
                    fps_start = time.time()
        print('[MAIN LOOP] Aborted.')
