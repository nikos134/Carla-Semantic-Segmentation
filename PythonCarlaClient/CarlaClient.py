#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys




# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc

import weakref
import random
import scipy.misc
import re

import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
from unet.model.unet import unet
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=tf_config))
class_colors = [(0,0,0), ( 70, 70, 70), (190, 153, 153), 	(250, 170, 160), (220, 20, 60),(153, 153, 153),(157, 234, 50),(128, 64, 128),(244, 35, 232),(107, 142, 35),( 0, 0, 142),(102, 102, 156),(220, 220, 0)]


try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_p
    from pygame.locals import K_o
    from pygame.locals import K_r
    from pygame.locals import K_c
    from pygame.locals import K_1
    from pygame.locals import K_2
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np

    np.set_printoptions(threshold=sys.maxsize)
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

# ==============================================================================
# -- Weather ---------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientLidar(object):
    """
    This is a module responsible for transform 3D lidar data to 2d and drawing them
    client-side on pygame surface.
    """
    max_x = 16
    max_z = 18
    max_r = 19

    @staticmethod
    def get_lidar_2d(image, camera, lidar, global_points):
        """
        Transform 3D lidar to 2D based on the rbg camera location
        Save the world cooridantes to an array for saving at the end of the program
        """
        if image is not None:

            lidar_matrix = ClientLidar.get_matrix(lidar.get_transform())
            # x=-2, y=-1.4399999976158142, z=0.5
            car_matrix = ClientLidar.get_matrix(carla.Transform(carla.Location(z=-1), carla.Rotation(yaw=90, pitch=0.0)))


            lidar_matrix = np.dot(lidar_matrix, car_matrix)

            #  image size in bytes iterate every 12 bytes (3 elemets each time of 4 bytes)
            pixels = []
            color = []
            for i in range(0, image.__len__(), 12):

                points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'), count=3, offset=i)
                points = np.reshape(points, (int(points.shape[0] / 3), 3))
                points_new = np.ones((np.size(points, 0), 4))  # row x 4 array of zeros
                points_new[:, :-1] = points
                points_new = np.transpose(points_new)

                world_cords = np.dot(lidar_matrix, points_new)

                global_points.append(world_cords)

                camera_matrix = ClientLidar.get_matrix(camera.get_transform())

                world_sensor_matrix = np.linalg.inv(camera_matrix)

                sensor_cords = np.dot(world_sensor_matrix, world_cords)  # x y z

                cords_y_minus_z_x = np.concatenate([sensor_cords[1], sensor_cords[2], sensor_cords[0]])
                lidar_data = (np.dot(camera.calibration, cords_y_minus_z_x))

                r = np.sqrt(np.power(cords_y_minus_z_x[0], 2) + np.power(cords_y_minus_z_x[1], 2) + np.power(cords_y_minus_z_x[2], 2))

                lidar_data = np.append(lidar_data, r, axis=0)

                pos2d = np.array([lidar_data[0] / lidar_data[2], lidar_data[1] / lidar_data[2], lidar_data[2]])
                # print(pos2d)
                if pos2d[2] > 0:

                    x_2d = pos2d[0]
                    y_2d = pos2d[1]
                    # print(x_2d, y_2d)
                    if x_2d >= 0 and x_2d < 1280 and y_2d >= 0 and y_2d < 720:
                        if cords_y_minus_z_x[0] > ClientLidar.max_x:
                            ClientLidar.max_x = cords_y_minus_z_x[0]
                        elif cords_y_minus_z_x[2] > ClientLidar.max_z:
                            ClientLidar.max_z = cords_y_minus_z_x[2]
                        elif lidar_data[3] > ClientLidar.max_r:
                            ClientLidar.max_r = lidar_data[3]

                        # print("max x: %d max z: %d max r: %d" % (ClientLidar.max_x, ClientLidar.max_z, ClientLidar.max_r))
                        #  print("x: %f y: %f z: %f r: %f " % (cords_y_minus_z_x[0], cords_y_minus_z_x[1], cords_y_minus_z_x[2], lidar_data[3]))
                        pixels.append([x_2d, y_2d])
                        r = 255 * (1 - np.min(cords_y_minus_z_x[0]/ClientLidar.max_x, 1))
                        g = 255 * (1 - np.min(cords_y_minus_z_x[2]/ClientLidar.max_z, 1))
                        b = 255 * (1 - np.min(lidar_data[3]/ClientLidar.max_r, 1))
                        # print('r %d g %d b %d' % (r, g, b))
                        # int(np.sqrt(np.power(pos2d[0],2) + np.power(pos2d[1], 2))/120 * 255)
                        color.append([r, g, b])

            pixels = np.array(pixels)
            pixels = pixels.reshape(int(pixels.size/2), 2)

            color = np.array(color)
            color = color.reshape(int(color.size/3), 3)
            if color.size != 0:
                color = np.interp(color, (color.min(), color.max()), (0, +255))

            return pixels, image.frame_number, color, global_points
        return  None

    @staticmethod
    def translate(value, leftMin, leftMax, rightMin, rightMax):
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)

    @staticmethod
    def draw_lidar_pixels(display, pixels, color, flag):
        """
        Draws bounding boxes on pygame display.
        """
        if not flag:
            display.fill((0, 0, 0))

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))

        for i, pixel in enumerate(pixels):

            # print('Color i: %d %d' % ( i, color[i, 0]))
            bb_color = (color[i, 0], color[i, 1], color[i, 2])
            # print('Color I: %d', i)
            # print(bb_color)
            # bb_color = (255, 00, 00)

            pygame.draw.line(bb_surface, bb_color, pixel, pixel)

        display.blit(bb_surface, (0, 0))

    @staticmethod
    def saveLidarImage(pixels, color, frame_number):
        image = np.ones((720, 1280, 3), dtype=np.uint8)

        for i, p in enumerate(pixels):
            pl = int(round(p[1]))
            pr = int(round(p[0]))
            if pl == 720:
                pl = 719
            elif pr == 1280:
                pr = 1279
            image[pl, pr, 0] = color[i, 0]
            image[pl, pr, 1] = color[i, 1]
            image[pl, pr, 2] = color[i, 2]



            # print(image[int(round(p[1])), int(round(p[0]))])
        # image = np.reshape(image, (720, 1280, 3))
        # # png.from_array(image, mode="RGB").save('/media/nikos134/DATADRIVE1/CarlaData/25_07/_out_4/%08d.png' % frame_number)  # works
        # image = pil.fromarray(image)
        # image.save('/media/nikos134/DATADRIVE1/CarlaData/25_07/_out_4/%08d.png' % frame_number, "PNG")
        scipy.misc.toimage(image, cmin=0.0, cmax=...).save('/media/nikos134/DATADRIVE1/CarlaData/19_08/_out_4/%08d.png' % frame_number)


    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.lidar = None
        self.camera_depth = None
        self.camera_segmentation = None
        self.car = None
        self.npc_list = []
        self.autoPilot = True
        self.display_camera = True

        self.display = None
        self.image = None
        self.lidar_image = None
        self.depth_image = None
        self.segmentation_image = None

        self.capture = True
        self.capture_lidar = True
        self.capture_depth = True
        self.capture_segmentation = True
        self.seg = False

        self.record = False

        self.world_point_cloud = []

        self.global_points = []

        self._weather_presets = find_weather_presets()
        self._weather_index = 0

        ROOT_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/CNN/unet")

        self.model_path = os.path.join(ROOT_DIR, 'ver2.h5')

        self.model = load_model(self.model_path, custom_objects={'dice_coef': dice_coef})

    def camera_blueprint(self, filter):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find(filter)
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))

        return camera_bp

    def lidar_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        camera_bp.set_attribute('range', '3000')
        camera_bp.set_attribute('channels', '64')
        camera_bp.set_attribute('points_per_second', '100000')
        camera_bp.set_attribute('rotation_frequency', '10')
        camera_bp.set_attribute('lower_fov', '-100')
        camera_bp.set_attribute('upper_fov', '20')
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.ford.mustang')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)
        self.car.set_autopilot(self.autoPilot)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        as well as depth and segmation camera for data saving
        Sets calibration for lidar rendering.
        """

        """
            Setting Rgb Camera and listen function
        """
        camera_transform = carla.Transform(carla.Location(x=1, z=1), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        """
            Setting Depth Camera and listen function
        """
        camera_transform = carla.Transform(carla.Location(x=1, z=1), carla.Rotation(pitch=0))
        self.camera_depth = self.world.spawn_actor(self.camera_blueprint('sensor.camera.depth'), camera_transform,
                                             attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_depth.listen(lambda image: weak_self().set_depth(weak_self, image))

        """
            Setting Segmentation Camera and listen function
        """
        camera_transform = carla.Transform(carla.Location(x=1, z=1))
        self.camera_segmentation = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'),
                                                         camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation.listen(lambda image: weak_self().set_segmentation(weak_self, image))

        """
           Setting lidar and listen function
        """
        # bound_y = 0.5 + self.car.bounding_box.extent.y
        # print(bound_y)
        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=1.4), carla.Rotation(pitch=0.0))
        self.lidar = self.world.spawn_actor(self.lidar_blueprint(), lidar_transform, attach_to=self.car, attachment_type=carla.AttachmentType.Rigid)
        weak_self = weakref.ref(self)
        self.lidar.listen(lambda image: weak_self().set_lidar(weak_self, image))



        """
            Identity matrix for calibration used in lidar data
        """
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def keyboard_event(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == K_ESCAPE:
                    return True
                elif event.key == K_p :
                    self.destroy_npc()
                elif event.key == K_o :
                    self.autoPilot = not self.autoPilot
                    self.car.set_autopilot(self.autoPilot)
                    print("Autopilot: ", self.autoPilot)
                elif event.key == K_r :
                    self.record = not self.record
                elif event.key == K_1 :
                    self.display_camera = not self.display_camera
                elif event.key == K_c:
                    self.next_weather()
            keys = pygame.key.get_pressed()
            if not self.autoPilot:
                control = car.get_control()
                control.throttle = 0
                if keys[K_w]:
                    control.throttle = 1
                    control.reverse = False
                elif keys[K_s]:
                    control.throttle = 1
                    control.reverse = True
                if keys[K_a]:
                    control.steer = max(-1., min(control.steer - 0.05, 0))
                elif keys[K_d]:
                    control.steer = min(1., max(control.steer + 0.05, 0))
                else:
                    control.steer = 0
                control.hand_brake = keys[K_SPACE]

                car.apply_control(control)

        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False
        if self.record:
            img.save_to_disk('/media/nikos134/DATADRIVE1/CarlaData/19_08/_out_0/%08d' % img.frame_number)

    @staticmethod
    def set_lidar(weak_self, img):
        """
        Sets lidar image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture_lidar:
            self.lidar_image = img
            self.capture_lidar = False
        if self.record:
            img.save_to_disk('/media/nikos134/DATADRIVE1/CarlaData/19_08/_out_3/%08d' % img.frame_number)

    @staticmethod
    def set_segmentation(weak_self, img):
        """
        Sets segmentation image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture_segmentation:
            self.segmentation_image = img
            self.capture_segmentation = False
        if self.record:
            img.convert(cc.CityScapesPalette)
            img.save_to_disk('/media/nikos134/DATADRIVE1/CarlaData/19_08/_out_2/%08d' % img.frame_number)

    @staticmethod
    def set_depth(weak_self, img):
        """
        Sets depth image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture_depth:
            self.depth_image = img
            self.capture_depth = False
        # if self.record:
            # img.save_to_disk('/media/nikos134/DATADRIVE1/CarlaData/25_07/_out_1/%08d' % img.frame_number)

    def set_capture(self, flag):
        self.capture_depth = flag
        self.capture = flag
        self.capture_segmentation = flag
        self.capture_lidar = flag

    def destroy_sensors(self):
        self.camera.destroy()
        self.lidar.destroy()
        self.camera_depth.destroy()
        self.camera_segmentation.destroy()

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None and self.display_camera:
            if self.seg:

                array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))

                array = np.reshape(array, (self.image.height, self.image.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]

                if array.shape[-1] == 4:
                    array = array[..., :3]

                array = cv2.resize(array, (512, 256), interpolation=cv2.INTER_NEAREST)
                image_new = np.zeros((1, 256, 512, 3))
                image_new[0] = array

                features = self.model.predict(image_new)
                image_seg_batch = np.zeros((256, 512, 3))
                image_seg_batch_final = np.zeros((512, 512, 3))

                predicted_image = features.reshape((256, 512, 13))

                for j in range(13):
                    image_seg_batch[:, :, 0] += ((predicted_image[:, :, j] > 0.9) * (class_colors[j][0])).astype(
                        'uint8')
                    image_seg_batch[:, :, 1] += ((predicted_image[:, :, j] > 0.9) * (class_colors[j][1])).astype(
                        'uint8')
                    image_seg_batch[:, :, 2] += ((predicted_image[:, :, j] > 0.9) * (class_colors[j][2])).astype(
                        'uint8')

                image_seg_batch_final = cv2.resize(image_seg_batch, (self.image.width,self.image.height), interpolation=cv2.INTER_NEAREST)


                surface = pygame.surfarray.make_surface(image_seg_batch_final.swapaxes(0, 1))
            else:
                array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (self.image.height, self.image.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def spawn_vehicles(self, number_of_vehicles):
        """

        :param number_of_vehicles:
        :return:
        """


        """
            Get Blueprint library and all available spawn points
        """
        blueprint_library = self.world.get_blueprint_library()

        spawn_points = self.world.get_map().get_spawn_points()
        num_of_spawn_points = len(spawn_points)

        if num_of_spawn_points > number_of_vehicles:
            random.shuffle(spawn_points)
        else:
            print('Not enough spawn points for the requested vehicles')
            return False

        spawn_actor = carla.command.SpawnActor
        set_autopilot = carla.command.SetAutopilot
        future_actor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break

            blueprint = random.choice(blueprint_library.filter('vehicle.*'))
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
                blueprint.set_attribute('role_name', 'autopilot')

            """
                Try to spawn actor, check if point is already occupied
            """
            npc = spawn_actor(blueprint,transform).then(set_autopilot(future_actor, True))
            batch.append(npc)

        """
            Check for responses and if actors were created
        """
        for response in self.client.apply_batch_sync(batch):
            if response.error:
                print(response.error)
            else:
                print("Actor created: ", response.actor_id)
                self.npc_list.append(response.actor_id)
        return True

    def destroy_npc(self):
        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.npc_list])

    def write_world_file(self):
        ply = np.array(self.global_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        ply = PlyElement.describe(ply, 'World')
        PlyData([ply], text=True).write('/media/nikos134/DATADRIVE1/CarlaData/19_08/world.ply')

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.car.get_world().set_weather(preset[0])

    def game_loop(self):
        """
        Main program loop.
        """

        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)

            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            self.spawn_vehicles(5)
            pixel_array = []
            color_array = []
            n = 0

            frame_start = False

            while True:
                if n == 2:
                    pixel_array = []
                    color_array = []
                    n = 0
                self.world.tick()

                self.set_capture(True)
                pygame_clock.tick_busy_loop(20)

                self.render(self.display)
                pixels, frame_number, color, self.global_points = ClientLidar.get_lidar_2d(self.lidar_image, self.camera, self.lidar, self.global_points)
                pixel_array = np.append(pixel_array, pixels)
                pixel_array = pixel_array.reshape(int(pixel_array.size/2), 2)
                color_array = np.append(color_array, color)
                color_array = color_array.reshape(int(color_array.size / 3), 3)

                # ClientLidar.draw_lidar_pixels(self.display, pixel_array, color_array, self.display_camera)
                if frame_start == False:
                    frame_start = frame_number
                print("Frame Number: %d Record: %d Weather: %d" % (frame_number - frame_start, self.record, self._weather_index))

                print("Status Weather: ", ((frame_number - frame_start) % 1000))
                if (frame_number - frame_start) % 1000 == 0:
                    print("Yes")
                    self.next_weather()
                if self.record:
                    ClientLidar.saveLidarImage(pixel_array, color_array, frame_number)


                pygame.display.flip()

                pygame.event.pump()
                n = n +1
                if self.keyboard_event(self.car):
                    return
                if frame_number-frame_start == 100:
                    self.record = True
                elif frame_number-frame_start > 6100:
                    self.record = False

        finally:
            # self.write_world_file()
            self.set_synchronous_mode(False)
            self.destroy_sensors()
            self.car.destroy()
            self.destroy_npc()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
