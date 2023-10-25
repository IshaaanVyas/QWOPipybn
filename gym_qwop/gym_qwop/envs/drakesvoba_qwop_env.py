import os
import io
import re
import cv2
from mss import mss
import math
import time
import json
import signal
import atexit
import numpy as np

import tesserocr
from PIL import Image

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


GAME_CROP = {
    'top':270,
    'left':215,
    'width':385,
    'height':295
}

GAME_RESIZE_WIDTH = 96
GAME_RESIZE_HEIGHT = 96
GAME_FRAME_STACK_SIZE = 4

SCORE_CROP = {
    'top':225, 
    'left':270, 
    'width':260, 
    'height':50
}
SCORE_RESIZE_WIDTH = 300
SCORE_RESIZE_HEIGHT = 40

# Game over screen
DONE_CROP = {
    'top':450, 
    'left':315, 
    'width':60, 
    'height':25}
DONE_RESIZE_WIDTH = 150
DONE_RESIZE_HEIGHT = 40
GAME_OVER_STRINGS = ["press"]

screen = mss()

ocr_api = tesserocr.PyTessBaseAPI()

class FrameQWOPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        chrome_opts = webdriver.ChromeOptions()
        chrome_opts.add_extension('/home/astriotech/Desktop/CS4246/DQN-QWOP/agent/chrome-driver/flash_player.crx')
        
        self.driver = webdriver.Chrome(options = chrome_opts)
        self.driver.get("http://localhost:3000")

        self.action_space = spaces.MultiBinary(4)

        self.observation_space = spaces.Box(low=0, high=255, shape=(96, 96, 4), dtype=np.uint8)

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)
        signal.signal(signal.SIGTSTP, self.close)
        atexit.register(self.close)

        time.sleep(10)

        self.canvas = self.driver.find_element(By.ID, "game-canvas")

        self.max_fps = 18
        self.last_call = time.time()
        self.last_state = time.time()
        self.old_posx = 0
        self.old_velx = 0

        self.frame_stack = np.zeros((GAME_RESIZE_HEIGHT, GAME_RESIZE_WIDTH, GAME_FRAME_STACK_SIZE), dtype=np.uint8)

        self.get_state()
    
    def obs(self, state=None):
        game_img = screen.grab(GAME_CROP)
        game_img = np.array(game_img, dtype=np.uint8)[:, :, :3]

        # Convert to grayscale and resize
        game_img = cv2.cvtColor(game_img, cv2.COLOR_BGR2GRAY)
        game_img = cv2.resize(game_img, (GAME_RESIZE_WIDTH, GAME_RESIZE_HEIGHT))
        
        # Add channel dimension first (in case you want RGB later)
        game_img = np.reshape(game_img, (GAME_RESIZE_HEIGHT, GAME_RESIZE_WIDTH))
        
        # Roll frame stack and add image to the end
        self.frame_stack = np.roll(self.frame_stack, -1, axis=-1)
        self.frame_stack[:, :, -1] = game_img
        
        
        return self.frame_stack

    def get_state(self):
        state = {}

        timestamp = time.time()
        score_img = screen.grab(SCORE_CROP)
        score_img = np.array(score_img)[:, :, :3]
        section_time = time.time() - timestamp
        print(f"Screen grab T: {section_time:.5f} sec")

        # Smaller, grayscale image with dark text on light background makes for fast OCR
        timestamp = time.time()
        score_img = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
        score_img = cv2.resize(score_img, (SCORE_RESIZE_WIDTH, SCORE_RESIZE_HEIGHT))
        score_img = 255 - score_img
        pil_img = Image.fromarray(score_img)
        section_time = time.time() - timestamp
        print(f"Prep image T: {section_time:.5f} sec")

        # Use tesserocr to get text from image
        ocr_str = ""
        timestamp = time.time()
        try:
            ocr_api.SetImage(pil_img)
            ocr_str = ocr_api.GetUTF8Text()
        except:
            print("ERROR: Could not perform OCR")
            
        # Display timing
        section_time = time.time() - timestamp
        print(f"OCR T: {section_time:.5f} sec")

        # Display OCR results
        score = 0.0
        if ocr_str:
            score_str = ocr_str.split()[0]
            try:
                score = float(float(score_str))
            except ValueError:
                pass
        print(f"OCR string T: {ocr_str}")
        print(f"Score T: {score}")

        timestamp = time.time()
        done_img = screen.grab(DONE_CROP)
        done_img = np.array(done_img)[:, :, :3]
        section_time = time.time() - timestamp
        print(f"Screen grab: {section_time:.5f} sec")

        # Smaller, grayscale image with dark text on light background makes for fast OCR
        timestamp = time.time()
        done_img = cv2.cvtColor(done_img, cv2.COLOR_BGR2GRAY)
        done_img = cv2.resize(done_img, (DONE_RESIZE_WIDTH, DONE_RESIZE_HEIGHT))

        pil_img = Image.fromarray(done_img)
        section_time = time.time() - timestamp
        print(f"Prep image: {section_time:.5f} sec")

        # Use tesserocr to get text from image
        ocr_str = ""
        timestamp = time.time()
        try:
            ocr_api.SetImage(pil_img)
            ocr_str = ocr_api.GetUTF8Text()
        except:
            print("ERROR: Could not perform OCR")
            
        # Display timing
        section_time = time.time() - timestamp
        print(f"OCR: {section_time:.5f} sec")

        # Display OCR results
        done = False
        ocr_str = ocr_str.strip()
        if ocr_str:
            done_str = ocr_str.split()[0].lower()
            if done_str in GAME_OVER_STRINGS:
                done = True
        print(f"OCR string: {ocr_str}")
        print(f"Done: {done}")

            
        state["posx"] = float(score)
        state["terminal"] = done

        current_time = time.time()
        delta = current_time - self.last_state
        self.last_state = current_time
        
        state["diffx"] = state["posx"] - self.old_posx

        state["velx"] = (state["diffx"]) / delta
        state["accelx"] = (state["velx"] - self.old_velx) / delta

        self.old_posx = state["posx"]
        self.old_velx = state["velx"]

        return state
    
    def compute_reward(self, state=None):
        curr_time = time.time()
        state = state if state != None else self.get_state()
        reward = state["diffx"] / (curr_time - self.last_reward)

        reward = -1 if state["terminal"] else reward * 10

        self.last_reward = time.time()

        return reward
    
    def throttle(self):
        call_time = time.time()
        delta = call_time - self.last_call    
        min_delta = 1 / self.max_fps
        if delta < min_delta: time.sleep(min_delta - delta)
        self.last_call = call_time

        return True

    def step(self, action):
        self.throttle()

        state = self.get_state()
        obs = self.obs()
        reward = self.compute_reward(state)
        action_chain = ActionChains(self.driver)

        if action[0] == 1:
            action_chain.key_down('q')
        else:
            action_chain.key_up('q')

        if action[1] == 1:
            action_chain.key_down('w')
        else:
            action_chain.key_up('w')

        if action[2] == 1:
            action_chain.key_down('o')
        else:
            action_chain.key_up('o')

        if action[3] == 1:
            action_chain.key_down('p')
        else:
            action_chain.key_up('p')

        action_chain.perform()

        print(reward)

        return obs, reward, state["terminal"], { "state": state }
    
    def reset(self):
        self.canvas.click()

        ActionChains(self.driver).key_up('p').key_up('o').key_up('w').key_up('q').key_down('r').key_up('r').perform()

        time.sleep(0.03)

        self.last_reward = time.time()

        state = self.get_state()

        self.last_call = time.time()
        self.old_velx = 0

        return self.obs()

    def render(self, mode='human'): pass

    def close(self, *args, **kwargs):
        self.driver.quit()


