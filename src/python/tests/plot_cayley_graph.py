from optparse import OptionParser

import cv2
import glfw
import json
import os
import numpy as np

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from diamond import Diamond
from lamplighter import LLElement
from llcayleygraph import LLCayleyGraph

_MAG = 1.5
DISPLAY = (round(1600 * _MAG), round(1024 * _MAG))
# DISPLAY = (800, 600)

VERTEX_COLOR = {
    0: np.array([1, 1, 1]),
    1: np.array([0, 0, 1]),
    2: np.array([1, 0, 0]),
    3: np.array([0, 1, 0]),
    4: np.array([1, 0, 1]),
    5: np.array([0, 1, 1]),
    6: np.array([1, 1, 0]),
    7: np.array([0, 0, 0.75]),
    8: np.array([0.75, 0, 0]),
    9: np.array([0, 0.75, 0]),
    10: np.array([0.5, 0, 0.5]),
    11: np.array([0, 0.5, 0.5]),
    12: np.array([0.5, 0.5, 0]),
    13: np.array([0.1, 0.3, 0.5]),
}

IMG_ROOT = os.path.expanduser("~/llmedia")


def plot(g, frame_i, R0, *, scaling_alpha=0.0003):
    glPushMatrix()
    scale = 1.25 * np.exp(-scaling_alpha * frame_i) * (R0 ** (-0.7))
    glScaled(scale, scale, scale)

    edges = g.get_list_of_edges()
    glColor3d(0.8, 0.8, 0.8)
    glBegin(GL_LINES)
    for edge in edges:
        for v in edge:
            p1 = g.coo[v]
            glVertex3fv(p1)
    glEnd()

    glPointSize(10)
    glBegin(GL_POINTS)
    for v in g.vertexes:
        p1 = g.coo[v]
        glColor3dv(VERTEX_COLOR[g.distance_to_origin[v]])
        glVertex3fv(p1)
    glEnd()

    glPopMatrix()


def get_display_pixels(rendered_image_width, rendered_image_height):
    data = glReadPixels(0, 0, rendered_image_width, rendered_image_height, GL_RGB, GL_UNSIGNED_BYTE)
    return np.frombuffer(data, dtype=np.uint8).reshape(rendered_image_height, rendered_image_width, 3)[::-1]


def init_scene_glfw():
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(DISPLAY[0], DISPLAY[1], "LL", None, None)
    glfw.make_context_current(window)


def init_scene_pygame():
    pygame.init()
    pygame.display.set_mode(DISPLAY, DOUBLEBUF|OPENGL)


def open_scene():
    gluPerspective(45, (DISPLAY[0]/DISPLAY[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)


def open_scene_pygame():
    init_scene_pygame()
    open_scene()


def open_scene_glfw():
    init_scene_glfw()
    open_scene()


def before_plot():
    glRotatef(0.5, 3, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


def start_show_pygame(g, R0, mode=1, scaling_alpha=0.0003):




    frame_i = 0
    while True:
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         quit()

        glRotatef(0.5, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        plot(g, frame_i, R0, scaling_alpha=scaling_alpha)

        # image_buffer = glReadPixels(0, 0, DISPLAY[0], DISPLAY[1], GL_RGB, GL_UNSIGNED_BYTE)
        # image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(DISPLAY[0], DISPLAY[1], 3)[::-1]

        # image = get_display_pixels(DISPLAY[0], DISPLAY[1])
        # cv2.imwrite(os.path.join(img_root, "image.png"), image)
        # exit()

        # glfw.swap_buffers(window)

        pygame.display.flip()
        pygame.time.wait(10)
        frame_i += 1

        if mode == 1:
            if frame_i < 1500:
                g.move(0.025)

            if frame_i % 50 == 0 and frame_i <= 250:
                print("{:4d}".format(frame_i))
            # if frame_i % 250 == 0:
            #     g.expand_via_propagate(0.025)

            # if frame_i == 100:
            #     g.expand_border1()
            # if frame_i % 250 == 0:
            #     if len(g) < 550 and False:
            #         g.binary_expand()
            #         # g.expand_border2()

        elif mode == 2:
            if frame_i < 1500:
                g.move(0.025)

            if frame_i % 50 == 0 and frame_i <= 250:
                print("{:4d}".format(frame_i))

            if frame_i % 250 == 0:
                if len(g) < 50:
                    g.expand_border()
                    # g.expand_border2()


def growing_balls():
    g = LLCayleyGraph()
    g.create_origin()
    g.expand_border()
    print(g)
    R0 = 2
    scaling_alpha = 0.003

    open_scene_pygame()

    frame_i = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        before_plot()
        plot(g, frame_i, R0, scaling_alpha=scaling_alpha)
        pygame.display.flip()
        pygame.time.wait(10)
        frame_i += 1

        if frame_i < 1500:
            g.move(0.025)

        if frame_i % 50 == 0 and frame_i <= 250:
            print("{:4d}".format(frame_i))

        if frame_i % 250 == 0:
            if g.n_expand < 5:
                g.expand_border()



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-a", "--action")

    (options, args) = parser.parse_args()

    action = options.action

    if action == "growing_balls":
        growing_balls()
    elif action == "growing_diamonds":
        growing_diamonds()

    g = LLCayleyGraph()
    # g.create_origin()
    # g.expand_border()
    # print(g)
    # main(g, 1, 2, 0.001)
    # g.build1(3)

    # R0 = 10
    # # g.build_normalized_ball(R0, 0.25 / (R0 ** 0.5))
    # g.build_normalized_ball(R0, 0.01)
    # g.structure_init(z0=-2, z1=1)

    # R0 = 4
    # g.create_rhombus()

    # g.create_simple(3)
    # g.remove_hanging()
    # exit()

    # R0 = 1
    # g.expand_via_propagate_init(0.7)

    _1 = LLElement(0, 0)
    # print("a", _1.a())
    # print("b", _1.a())
    # print("c", _1.a().bi(), _1.c())
    # exit()

    d = Diamond(start=_1)
    print(d)
    d2 = Diamond(start=d)
    print(d2)
    d3 = Diamond(start=d2)
    print(d3)
    # d4 = Diamond(start=d3)
    # print(d4)
    # exit()
    d1 = d2

    R0 = 8
    v_set0 = d1.vertexes()
    v_set = set()
    for k in range(12):
        v_set = v_set.union(v_set0)
        v_set0 = {x.br() for x in v_set0}
    g.build_graph_from_v_set(v_set)

    # for it in range(1):
    #     print("-" * 88)
    #     g.expand_border()
    #     # print(g)
    #     print(g.t_distribution())
    #     print(g.size())
    #     print(len(g.get_list_of_edges()))
    #     if it < 1:
    #         print(g.get_list_of_edges())
    #     # print(json.dumps({str(x): 1 for x, i in g.vertexes.items()}, indent=4))

    # exit()

    start_show_pygame(g, R0)



