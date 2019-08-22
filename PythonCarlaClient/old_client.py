#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('/home/nikos134/Unreal_Carla/Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
    from pygame.locals import K_o
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from numpy.linalg import inv
# ==============================================================================
# -- global -------------------------------------------------------------------
# ==============================================================================
world = None
client = None
lidar_matrix = None
camera_matrix = None
actor_list = []
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World Class -------------------------------------------------------------------
# ==============================================================================


class World:
    def __init__(self, world_carla, hud,display):
        self.carla_world = world_carla
        self._autopilot = True
        self._control = carla.VehicleControl()
        self.player = None
        self.map = self.carla_world.get_map()
        self.actor_role_name = "hero"
        self.hud = hud
        self.carla_world.on_tick(hud.on_world_tick)
        self.display = display

        # get blueprint for player
        blueprint = random.choice(self.carla_world.get_blueprint_library().filter("vehicle.ford.mustang"))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')

        # Spawn the player.
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.carla_world.try_spawn_actor(blueprint, spawn_point)

        self.player.set_autopilot(self._autopilot)
        self.camera_manager = CameraManager(self.player, self.hud, self.display)

    @staticmethod
    def destroy_actors():
        global actor_list
        global client
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

    def tick(self, clock):
        self.hud.tick(self, clock)


    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, display):
        self.sensor = []
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.display = display
        self.k = np.identity(3)
        self.k[0, 2] = 1280 / 2.0
        self.k[1, 2] = 720 / 2.0
        self.k[0, 0] = self.k[1, 1] = 1280 / (2.0 * np.tan(90.0 * np.pi / 360.0))

        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            # (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            # (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=0, y=0, z=2.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            # ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            # ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            # ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(1280))
                bp.set_attribute('image_size_y', str(720))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
                bp.set_attribute('channels', '64')
                bp.set_attribute('points_per_second', '100000')
                bp.set_attribute('rotation_frequency', '30')

            item.append(bp)

        for n in range(4):
            self.sensor.append(self._parent.get_world().spawn_actor(
                self.sensors[n][-1],
                self._camera_transforms[n][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[n][1]))
            weak_self = weakref.ref(self)
            print(self.sensor[n])
            print('created %s' % self.sensor[n].type_id)
        self.sensor[0].listen(lambda image: self._parse_image(image, 0))
        self.sensor[1].listen(lambda image: self._parse_image(image, 1))
        self.sensor[2].listen(lambda image: self._parse_image(image, 2))
        self.sensor[3].listen(lambda image: self._parse_image(image, 3))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def _parse_image(self, image, index):

        if not self:
            return
        global lidar_matrix
        global camera_matrix

        if self.sensors[index][0].startswith('sensor.lidar'):
             # points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
             # points = np.reshape(points, (int(points.shape[0] / 3), 3))
             # lidar_data = np.array(points[:, :2])
             # print(lidar_data)
            # lidar_data *= min(self.hud.dim) / 100.0
            # lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            # lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            # lidar_data = lidar_data.astype(np.int32)
            # lidar_data = np.reshape(lidar_data, (-1, 2))
            # lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            # lidar_img = np.zeros((lidar_img_size), dtype = int)
            # lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            # self.surface = pygame.surfarray.make_surface
            lidar_matrix = self.get_matrix(self.sensor[3].get_transform())
            print(self.sensor[3].get_transform())
            #  image size in bytes iterate every 12 bytes (3 elemets each time of 4 bytes)
            pixels = []
            for i in range(0, image.__len__(),  12):
                points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'), count=3, offset=i)
                points = np.reshape(points, (int(points.shape[0] / 3), 3))
                points_new = np.ones((np.size(points, 0), 4))                              #  row x 4 array of zeros
                points_new[:, :-1] = points
                points_new = np.transpose(points_new)
                #
                # point_pos = np.dot(lidar_matrix, points_new)
                world_cords = np.dot(lidar_matrix, points_new)
                camera_matrix = self.get_matrix(self.sensor[0].get_transform())

                world_sensor_matrix = np.linalg.inv(camera_matrix)

                sensor_cords = np.dot(world_sensor_matrix, world_cords) #  x y z

                cords_y_minus_z_x = np.concatenate([sensor_cords[1], -sensor_cords[2], sensor_cords[0]])
                lidar_data = (np.dot(self.k, cords_y_minus_z_x))

                pos2d = np.array([lidar_data[0] / lidar_data[2], lidar_data[1] / lidar_data[2], lidar_data[2]])
                # print(pos2d)
                if pos2d[2] > 0:
                    x_2d = pos2d[0]
                    y_2d = pos2d[1]
                    # print(x_2d, y_2d)
                    if x_2d >= 0 and x_2d < 1280 and y_2d >= 0 and y_2d < 720:
                        pixels.append([x_2d, y_2d])

            self.draw_pixel(pixels)
        else:
            image.convert(self.sensors[index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            if index == 0:
                # self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                camera_matrix = self.get_matrix(image.transform)
            #     print("----------------------Index 0 ----------------------------------")
            #     print("----------------------Frame %d ----------------------------------" % image.frame_number)
            #     print(image.transform)

        # if index == 3:
        #     print("----------------------Index 3 ----------------------------------")
        #     print(image.transform)
        # image.save_to_disk('/media/nikos134/DATADRIVE1/CarlaData/17_06/_out_%d/%08d' % (index, image.frame_number))

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

    def draw_pixel(self, pixels):
        """
        Draws bounding boxes on pygame display.
        """
        BB_COLOR = (248, 64, 24)
        bb_surface = pygame.Surface((1280, 720))
        bb_surface.set_colorkey((0, 0, 0))
        for idx,item in enumerate(pixels):

            bb_surface.set_at([item[0], item[1]], BB_COLOR)

        self.display.blit(bb_surface, (0, 0))

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        # self._notifications = FadingText(font, (width, 40), (0, height - 40))
#        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        # self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        vehicles = world.carla_world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18


# ==============================================================================
# -- loop -------------------------------------------------------------------
# ==============================================================================


def game_loop():
    # In this tutorial script, we are going to add a vehicle to the simulation
    # and let it drive in autopilot. We will also create a camera attached to
    # that vehicle, and save all the images generated by the camera to disk.
    pygame.init()
    pygame.font.init()
    global world
    global client

    num_of_vehicles = 0
    num_of_pedestrians = 20

    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (1280, 720),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(1280, 720)

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = World(client.get_world(), hud, display)

        # controller = KeyboardControl(world, args.autopilot)

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.carla_world.get_blueprint_library()

        # get all available spawn points

        spawn_points = world.carla_world.get_map().get_spawn_points()

        num_of_spawn_points = len(spawn_points)
        print('Number of spawn points: ', num_of_spawn_points)

        if num_of_spawn_points > num_of_vehicles:
            random.shuffle(spawn_points)
        else:
            print('Not enough spawn points terminating script')
            return

        spawn_actor = carla.command.SpawnActor
        set_autopilot = carla.command.SetAutopilot
        future_actor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= num_of_vehicles:
                break
            print('N ', n)
            bp = random.choice(blueprint_library.filter('vehicle.*'))
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            bp.set_attribute('role_name', 'autopilot')

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = spawn_actor(bp, transform).then(set_autopilot(future_actor, True))
            batch.append(npc)

        # Check the responses from the client

        for response in client.apply_batch_sync(batch):
            if response.error:
                print(response.error)
            else:
                actor_list.append(response.actor_id)
                print('created ', response.actor_id)


        # # spwan pedestrians
        #
        # for n, transform in enumerate(spawn_points):
        #     if n >= num_of_pedestrians:
        #         break
        #     print('N ', n)
        #     bp = random.choice(blueprint_library.filter('walker.*'))
        #     if bp.has_attribute('color'):
        #         color = random.choice(bp.get_attribute('color').recommended_values)
        #         bp.set_attribute('color', color)
        #     bp.set_attribute('role_name', 'autopilot')
        #
        #     # This time we are using try_spawn_actor. If the spot is already
        #     # occupied by another object, the function will return None.
        #     npc = spawn_actor(bp, transform).then(set_autopilot(future_actor, True))
        #     batch.append(npc)
        #
        # # Check the responses from the client
        #
        # for response in client.apply_batch_sync(batch):
        #     if response.error:
        #         print(response.error)
        #     else:
        #         actor_list.append(response.actor_id)
        #         print('created ', response.actor_id)
        #

        clock = pygame.time.Clock()

        # loop to run the game
        # parse any key events

        while True:
            clock.tick_busy_loop(60)
            # if controller.parse_events(client, world, clock):
            #    return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()


    finally:
        pass


def main():
    actor_list = []
    try:

        game_loop()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')




if __name__ == '__main__':

    main()
