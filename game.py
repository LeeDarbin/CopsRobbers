import pygame
import random
from enum import Enum # 열거형
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

Point = namedtuple('Point', 'x, y') # Dictionary와 비슷, 메모리를 효율적으로 자룰 수 있음

# rgb 색
WHITE = (255, 255, 255)
RED   = (200, 0, 0)
BLUE  = (0, 0, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 15 # 격자 크기
SPEED = 40      # 경찰과 도둑 속도
# Cops and Robbers
class CopsRobbersAI:
    def __init__(self, w = 330, h = 330):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Cops & Robber')
        self.clock = pygame.time.Clock()

        
        self.dn = 2
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.directiont = Direction.RIGHT

        self.thief  = Point(random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
         random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE)
        if(random.randint(0,1)): # x 좌표가 0 or 끝
            x = random.choice([0, self.w - BLOCK_SIZE])
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.door   = Point(BLOCK_SIZE * 5, self.w - BLOCK_SIZE)
            self.police = Point(x, y)
        else:                    # y 좌표가 0 or 끝
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.choice([0, self.h - BLOCK_SIZE])
            self.door   = Point(BLOCK_SIZE * 5, self.w - BLOCK_SIZE)
            self.police = Point(x, y)
        self.door = []
        for i in range(self.dn):
            self.door    += [Point(BLOCK_SIZE * random.randint(0, self.w // BLOCK_SIZE), self.w - BLOCK_SIZE)]

        self.Score = 0 # 경찰 점수
        self.Scoret = 0 # 도둑 점수
        self.pNum = 1 # 경찰 수
        self.tNum = 1 # 도둑 수
        
        self.frame_iteration = 0
    
    def _place_door(self):
        # if(random.randint(0,1)): # x 좌표가 0 or 끝
        #     x = random.choice([0, self.w - BLOCK_SIZE])
        #     y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        # else:                    # y 좌표가 0 or 끝
        #     x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        #     y = random.choice([0, self.h - BLOCK_SIZE])
        
        # self.door = Point(x, y)
        for i in range(self.dn):
            self.door    += [Point(BLOCK_SIZE * random.randrange(0, self.w // BLOCK_SIZE), self.w - BLOCK_SIZE)]

    def play_step(self, action, actiont):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # 경찰 이동
        self._movet(actiont) # 도둑 놈 이동
        # pygame.snake.insert(o, self. head) # 뭔지 모름

        # 3. check if game over
        reward = 0
        rewardt = 0
        game_over = False
        ### game over 상황
        if self.is_collision() or self.frame_iteration > 200: # 경
            game_over = True
            reward = -10 # 보상 값 변화
            rewardt = -10
            return reward, game_over, self.Score, self.Scoret, rewardt
        
        # 4. 
        if self.thief == self.police: # 도둑 잡힘
            print("도둑 경찰 잡다")
            self.Score += 1
            reward = 20
            rewardt = -30
            game_over = True
            return reward, game_over, self.Score, self.Scoret, rewardt
        for i in range(self.dn):
            if self.thief == self.door[i]:
                print("도둑 탈출")
                reward = -20
                rewardt = 50
                game_over = True
                return reward, game_over, self.Score, self.Scoret, rewardt

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over ans score
        return reward, game_over, self.Score, self.Scoret, rewardt


    ### def is_collision()
    def is_collision(self, pt = None):
        if pt is None:
            pt = self.police
         # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        return False
    def _update_ui(self):
        self.display.fill(WHITE)

        if self.pNum:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(self.police.x, self.police.y, BLOCK_SIZE, BLOCK_SIZE))
        if self.tNum:
            pygame.draw.rect(self.display, RED, pygame.Rect(self.thief.x, self.thief.y, BLOCK_SIZE, BLOCK_SIZE))
       
        for i in range(self.dn):
            pygame.draw.rect(self.display, BLACK, pygame.Rect(self.door[i].x, self.door[i].y, BLOCK_SIZE, BLOCK_SIZE))
        
        ### 점수 text 화면 보이기
        pygame.display.flip()
    
    def _move(self, action): # 경찰 move
        ###########
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        for i in range(len(action)):
            if action[i]:
                self.direction = clock_wise[i]
        ############
        # [right, left, up, down]
        x = self.police.x
        y = self.police.y
        if self.direction == Direction.RIGHT:
            if x < self.w - BLOCK_SIZE:
                x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            if x > 0:
                x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            if y < self.h - BLOCK_SIZE:
                y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            if y > 0:
                y -= BLOCK_SIZE
        self.police = Point(x, y)
    
    def _movet(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        for i in range(len(action)):
            if action[i]:
                self.directiont = clock_wise[i]
        x = self.thief.x
        y = self.thief.y
        if self.directiont == Direction.RIGHT:
            if x < self.w - BLOCK_SIZE:
                x += BLOCK_SIZE
        elif self.directiont == Direction.LEFT:
            if x > 0:
                x -= BLOCK_SIZE
        elif self.directiont == Direction.DOWN:
            if y < self.h - BLOCK_SIZE:
                y += BLOCK_SIZE
        elif self.directiont == Direction.UP:
            if y > 0:
                y -= BLOCK_SIZE
        self.thief = Point(x, y)

    