### Import libraries

# Screen capture
from mss import mss

# Sending commands (e.g. mouse/keyboard)
import pynput

# OpenCV for image manipulation
import cv2

# Optical character recognition (OCR)
import tesserocr
from PIL import Image

# Farama Foundation Gymnasium (fork of OpenAI gym)
import gymnasium as gym

# Webdriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Other
import time
import datetime
import os
import numpy as np
from matplotlib import pyplot as plt
from typing import Any, Dict, Tuple, Union


GAME_CROP = {
    'top':270,
    'left':215,
    'width':385,
    'height':295
}

# Image resize (minimum of 36x36 for default CnnPolicies)
GAME_RESIZE_WIDTH = 96
GAME_RESIZE_HEIGHT = 96
GAME_FRAME_STACK_SIZE = 4

# How big to make display image (0 for no display)
DISP_SCALE_FACTOR = 3.0

# Where to find the head/hair in the scaled game image
HEAD_ROW_MIN = 1
HEAD_ROW_MAX = 96
HEAD_PIXEL_THRESHOLD = 34
HEAD_HEIGHT_THRESHOLD = 23

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
    'width':100, 
    'height':25}
DONE_RESIZE_WIDTH = 150
DONE_RESIZE_HEIGHT = 40
GAME_OVER_STRINGS = ["press"]

# Action settings
RESTART_MOUSE_POS = (300, 570)
RESTART_KEY = 'r'
ACTIONS_KEY_PRESS_TIME = 0.1
ACTIONS_MAP = {
    0: 'no-op',
    1: 'press q',
    2: 'press w',
    3: 'press qp',
    4: 'press wo'
}

# Reward settings
FALL_REWARD = -20
KNEEL_REWARD = 0
SCORE_MULTIPLIER = 5
GOAL_MARKER = 20
GOAL_REWARD = 100
LONG_PRESS_THRESHOLD = 0.5  # Seconds
LONG_PRESS_REWARD = 0

# CnnPolicy requires 8-bit unsigned integers for images
DTYPE = np.uint8

# Size of sliding window average to calculate FPS
FPS_AVG_LEN = 30

# Checkpoint config
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FREQ = 25_000 # 25k steps with 15 fps is about 30 min

# Log config
LOG_DIR = "logs"
LOG_FREQ = 5_000 # 5k steps with 30 fps is about 5 min


class FrameQWOPEnv(gym.Env):

    # Set up the environment, action, and observation shapes. Optional timeout in seconds.
    def __init__(self, 
                 timeout=0.0, 
                 disp_scale=0.0, 
                 fps_limit=0.0, 
                 show_fps=False, 
                 debug_time=False):
        
        # Call superclass's constructor
        super().__init__()

        chrome_opts = webdriver.ChromeOptions()
        chrome_opts.add_extension('/home/astriotech/Desktop/CS4246/DQN-QWOP/agent/chrome-driver/flash_player.crx')
        
        self.driver = webdriver.Chrome(options = chrome_opts)
        self.driver.get("http://localhost:3000")

        time.sleep(10)

        self.canvas = self.driver.find_element(By.ID, "game-canvas")

        # Env requires us to define the action space
        self.action_space = gym.spaces.Discrete(len(ACTIONS_MAP))

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(GAME_RESIZE_HEIGHT, GAME_RESIZE_WIDTH, GAME_FRAME_STACK_SIZE),
            dtype=DTYPE
        )

        # Screen capture object
        self.screen = mss()
        
        # OCR context
        self.ocr_api = tesserocr.PyTessBaseAPI()
        
        # Interaction objects
        self.keyboard = pynput.keyboard.Controller()
        self.mouse = pynput.mouse.Controller()
        
        # Record total score between rounds (to calculate reward each step)
        self.score = 0.0

        # States for rewarding actions
        self.prev_action = 0
        self.action_timestamp = time.time()
        self.action_reward_received = False

        # Used to record the time
        self.timeout = timeout
        self.start_time = 0.0
        if self.timeout > 0.0:
            self.start_time = time.time()

        # How much to scale the render window
        self.disp_scale = disp_scale
        
        # Limit the FPS for consistency
        self.fps_limit = fps_limit
        
        # Record time for debugging and showing FPS in render window
        self.fps = 0.0
        self.avg_fps_array = [0.0] * FPS_AVG_LEN
        self.show_fps = show_fps
        self.timestamp = time.time()
        self.debug_time = debug_time
        self.debug_start_time = time.time()
        self.debug_timestamp = time.time()

        # Initialize game frame stack
        self.frame_stack = np.zeros((GAME_RESIZE_HEIGHT, GAME_RESIZE_WIDTH, GAME_FRAME_STACK_SIZE), dtype=DTYPE)
        
        # Show rendering in new window if requested
        if self.disp_scale > 0.0:
            cv2.namedWindow('Game Image')
        

    # What happens when you take a step in the game (e.g. each frame)
    def step(self, action):
        
        # Debug timing
        self._show_debug_time("Step start")
        
        # Set initial reward
        reward = 0
        
        # Release all keys
        self.keyboard.release('q')
        self.keyboard.release('w')
        self.keyboard.release('o')
        self.keyboard.release('p')

        # Perform action (don't do anything for no-op)
        if ACTIONS_MAP[action] == 'press q':
            self.keyboard.press('q')
        elif ACTIONS_MAP[action] == 'press w':
            self.keyboard.press('w')
        elif ACTIONS_MAP[action] == 'press qp':
            self.keyboard.press('q')
            self.keyboard.press('p')
        elif ACTIONS_MAP[action] == 'press wo':
            self.keyboard.press('w')
            self.keyboard.press('o')
        self._show_debug_time("Perform action")

        # Reward agent for holding the same key(s) for a while to encourage leg switching
        now = time.time()
        if action == self.prev_action:
            if (now - self.action_timestamp >= LONG_PRESS_THRESHOLD and 
                    not self.action_reward_received):
                
                # Enforce switching between 'qp' and 'wo'
                if ACTIONS_MAP[action] == 'press qp' and self.prev_key_combo == "wo":
                    reward += LONG_PRESS_REWARD
                    self.prev_key_combo = "qp"
                    print(f"Keys held: {ACTIONS_MAP[action]} for reward: {reward}")
                if ACTIONS_MAP[action] == 'press wo' and self.prev_key_combo == "qp":
                    reward += LONG_PRESS_REWARD
                    self.prev_key_combo = "wo"
                    print(f"Keys held: {ACTIONS_MAP[action]} for reward: {reward}")
                    
                # Prevent recurring rewards for same key press
                self.action_reward_received = True
        
        if action == self.prev_action:
            if (now - self.action_timestamp >= LONG_PRESS_THRESHOLD and 
                    not self.action_reward_received):
                reward += LONG_PRESS_REWARD
                self.action_reward_received = True
                print(f"Keys held: {reward}")
        else:
            self.action_timestamp = time.time()
            self.action_reward_received = False
        self.prev_action = action

        # Get next observation, render, and add to frame stack
        frame_stack = self.get_observation()
        self._show_debug_time("Get obs")
        
        # Use distance as reward score. Calculate score difference between this step and previous.
        prev_score = self.score
        self.score = self.get_score()
        reward += SCORE_MULTIPLIER * (self.score - prev_score)
        self._show_debug_time("Get score")

        # See if we ran the distance set by the goal
        if self.score >= GOAL_MARKER:
            reward += GOAL_REWARD
            terminated = True
            
        # Check if done, penalize agent for falling
        else:
            terminated = self.get_done()
            self._show_debug_time("Get done")
            if terminated:
                reward += FALL_REWARD
        
        # Penalize agent for letting head drop below a given row to discourage "scooting"
        head_row = self.get_head_row(frame_stack[-1])
        if head_row > HEAD_HEIGHT_THRESHOLD:
            reward += KNEEL_REWARD
        self._show_debug_time("Get head row")

        # Check if we've exceeded the time limit
        elapsed_time = 0.0
        truncated = False
        if not terminated and self.timeout > 0.0:
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.timeout:
                truncated = True

        # Release all control keys if ending
        if terminated or truncated:
            self.keyboard.release('q')
            self.keyboard.release('w')
            self.keyboard.release('o')
            self.keyboard.release('p')
            
        # Wait if needed to meet FPS limit
        now = time.time()
        if self.fps_limit > 0.0:
            to_wait = (1 / self.fps_limit) - (now - self.timestamp)
            if to_wait > 0:
                time.sleep(to_wait)

        # Calculate FPS and slide average FPS window
        now = time.time()
        self.fps = 1 / (now - self.timestamp)
        self.avg_fps_array = self.avg_fps_array[1:]
        self.avg_fps_array.append(self.fps)
        self.timestamp = now
        
        # Return auxiliary information for debugging
        info = {'score': self.score, 'time': elapsed_time, 'fps': self.fps}
        
        # Done debugging time
        self._show_debug_time("Final checks")
        if self.debug_time:
            print("---")

        return frame_stack, reward, terminated, truncated, info
    

    # Visualize the game using OpenCV
    def render(self, img, track_head=False):
        if self.disp_scale > 0:
            
            # Draw tracking marker for head and threshold row
            if track_head:
                head_row = self.get_head_row(img)
                head_col = int(img.shape[1] / 2)
                img[head_row, head_col] = 255
                img[HEAD_HEIGHT_THRESHOLD, ::3] = 255
            
            # Resize our game image to something that can be easily seen
            disp_width = int(GAME_RESIZE_WIDTH * DISP_SCALE_FACTOR)
            disp_height = int(GAME_RESIZE_HEIGHT * DISP_SCALE_FACTOR)
            disp_img = cv2.resize(img, (disp_width, disp_height), interpolation=cv2.INTER_AREA)
            
            # Add FPS counter to image
            if self.show_fps:
                fps = self.get_avg_fps()
                disp_img = cv2.putText(disp_img, 
                                       f"fps: {fps:.1f}", 
                                       (10, 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                       1, 
                                       (255), 
                                       2, 
                                       cv2.LINE_AA)
                

                
            # Draw and wait 1 ms
            cv2.imshow('Game Image', disp_img)
            cv2.waitKey(1)

    # Restart the game
    def reset(self):
        
        # Wait, move mouse to game window, click for focus
        time.sleep(0.5)
        self.mouse.position = RESTART_MOUSE_POS
        self.mouse.press(pynput.mouse.Button.left)
        self.mouse.release(pynput.mouse.Button.left)
        
        # Press 'r' to restart game
        self.keyboard.press(RESTART_KEY)
        time.sleep(ACTIONS_KEY_PRESS_TIME)
        self.keyboard.release(RESTART_KEY)
        
        # Reset score and time
        self.score = 0.0
        if self.timeout > 0.0:
            self.start_time = time.time()
            
        # States for rewarding actions
        self.prev_action = 0
        self.action_timestamp = time.time()
        self.action_reward_received = False
        self.prev_key_combo = "wo"
        
        # Let the game restart before getting the first observation
        time.sleep(0.3)
        
        # Reinitialize frame stack and get first observation of new game
        self.frame_stack[:, :, :] = 0
        frame_stack = self.get_observation()
        
        # Return auxiliary information for debugging
        info = {'score': self.score, 'time': 0.0, 'fps': 0.0}
        
        return frame_stack, info
    
    # Close down the game: release keys, close OpenCV windows, end OCR context
    def close(self):
        self.keyboard.release('q')
        self.keyboard.release('w')
        self.keyboard.release('o')
        self.keyboard.release('p')
        cv2.destroyAllWindows()
        self.ocr_api.End()
        self.driver.quit()

    # Get the part of the observation of the game that we want (e.g. crop, resize)
    def get_observation(self):
        
        # Get screen grab and drop alpha channel
        game_img = self.screen.grab(GAME_CROP)
        game_img = np.array(game_img, dtype=DTYPE)[:, :, :3]

        # Convert to grayscale and resize
        game_img = cv2.cvtColor(game_img, cv2.COLOR_BGR2GRAY)
        game_img = cv2.resize(game_img, (GAME_RESIZE_WIDTH, GAME_RESIZE_HEIGHT))
        
        # Add channel dimension first (in case you want RGB later)
        game_img = np.reshape(game_img, (GAME_RESIZE_HEIGHT, GAME_RESIZE_WIDTH))
        
        # Roll frame stack and add image to the end
        self.frame_stack = np.roll(self.frame_stack, -1, axis=-1)
        self.frame_stack[:, :, -1] = game_img
        
        # Render
        self.render(game_img, track_head=True)
        
        return self.frame_stack
    
    # Get the distance ran to use as a total score and to calculate rewards
    def get_score(self):
        
        # Get screen grab and drop alpha channel
        score_img = self.screen.grab(SCORE_CROP)
        score_img = np.array(score_img)[:, :, :3]

        # Resize, convert to grayscale, invert for fast OCR
        score_img = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
        score_img = cv2.resize(score_img, (SCORE_RESIZE_WIDTH, SCORE_RESIZE_HEIGHT))
        score_img = 255 - score_img
        pil_img = Image.fromarray(score_img)

        # Do OCR
        ocr_str = ""
        try:
            self.ocr_api.SetImage(pil_img)
            ocr_str = self.ocr_api.GetUTF8Text()
        except:
            print("ERROR: Could not perform OCR")
        
        # Extract score as a number
        score = 0.0
        ocr_str = ocr_str.strip()
        if ocr_str:
            score_str = ocr_str.split()[0]
            try:
                score = float(float(score_str))
            except ValueError:
                pass
        
        return score
    
    # Get the row of the head
    def get_head_row(self, img):
        
        # Give row of first dark pixel in likely head range
        locs = np.where(img[HEAD_ROW_MIN:HEAD_ROW_MAX, :] < HEAD_PIXEL_THRESHOLD)
        rows = np.sort(locs[0]) + HEAD_ROW_MIN
        if rows.size > 0:
            return rows[0]
        else:
            return -1
        

    # Get the done text using OCR
    def get_done(self):
        
        # Get screen grab and drop alpha channel
        done_img = self.screen.grab(DONE_CROP)
        done_img = np.array(done_img)[:, :, :3]
        
        # Resize, convert to grayscale, invert for fast OCR
        done_img = cv2.cvtColor(done_img, cv2.COLOR_BGR2GRAY)
        done_img = cv2.resize(done_img, (SCORE_RESIZE_WIDTH, SCORE_RESIZE_HEIGHT))
        pil_img = Image.fromarray(done_img)

        #cv2.namedWindow('Done Image')
        #cv2.imshow('Done Image', done_img)
        #cv2.waitKey(5000)
        #cv2.destroyAllWindows()
        
        # Do OCR
        ocr_str = ""
        try:
            self.ocr_api.SetImage(pil_img)
            ocr_str = self.ocr_api.GetUTF8Text()
        except:
            print("ERROR: Could not perform OCR")

        # Extract done state as a boolean
        done = False
        ocr_str = ocr_str.strip()
        if ocr_str:
            done_str = ocr_str.split()[0].lower()
            if done_str in GAME_OVER_STRINGS:
                done = True
                
        return done
    
    # Get average FPS
    def get_avg_fps(self):
        return sum(self.avg_fps_array) / len(self.avg_fps_array)
    
    # Report time elapsed from environment start and time elapsed from last call
    def _show_debug_time(self, msg=""):
        if self.debug_time:
            debug_now = time.time()
            print(f"Timestamp: {(debug_now - self.debug_start_time):.2f} | "
                  f"Since last: {(debug_now - self.debug_timestamp):.2f} | "
                  f"{msg}")
            self.debug_timestamp = debug_now
        
    


