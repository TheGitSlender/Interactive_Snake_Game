import pygame
import cv2
import numpy as np
import mediapipe as mp
import random
from threading import Thread
import queue
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()
width, height = 600, 400  # Reduced map size
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Hand-Controlled Snake")

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BORDER_COLOR = (64, 64, 64)

# Snake and game variables
cell_size = 20
border_thickness = 20  # Border thickness
play_area_width = width - (2 * border_thickness)
play_area_height = height - (2 * border_thickness)
high_score = 0
snake_speed = 8

# Direction thresholds for hand tracking
THRESHOLD_CENTER = 0.1
UP_THRESHOLD = 0.4
DOWN_THRESHOLD = 0.6
LEFT_THRESHOLD = 0.38
RIGHT_THRESHOLD = 0.65

def init_game():
    return {
        'snake_pos': [(width//2, height//2)],
        'direction': 'RIGHT',
        'food_pos': (
            random.randrange(border_thickness//cell_size, (width-border_thickness)//cell_size) * cell_size,
            random.randrange(border_thickness//cell_size, (height-border_thickness)//cell_size) * cell_size
        ),
        'score': 0,
        'game_over': False,
        'current_direction': None,
        'start_time': time.time(),  # Add start time for countdown
        'time_left': 60  # 60 seconds countdown
    }

game_state = init_game()

# Frame processing queue
frame_queue = queue.Queue(maxsize=2)
camera_active = True

def get_direction_from_finger(finger_x, finger_y):
    # Center zone - no direction change
    if (0.5 - THRESHOLD_CENTER < finger_x < 0.5 + THRESHOLD_CENTER and 
        0.5 - THRESHOLD_CENTER < finger_y < 0.5 + THRESHOLD_CENTER):
        return None
    
    # Determine direction based on finger position
    if finger_y < UP_THRESHOLD:
        return 'UP'
    elif finger_y > DOWN_THRESHOLD:
        return 'DOWN'
    elif finger_x < LEFT_THRESHOLD:
        return 'LEFT'
    elif finger_x > RIGHT_THRESHOLD:
        return 'RIGHT'
    
    return None

def process_hand_frame():
    cap = cv2.VideoCapture(0)
    
    while camera_active:
        success, frame = cap.read()
        if not success:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Create a black frame
        black_frame = np.zeros_like(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index finger tip position (landmark 8)
                finger_y = hand_landmarks.landmark[8].y
                finger_x = hand_landmarks.landmark[8].x
                
                # Draw only the index finger tip point
                h, w, c = frame.shape
                cx, cy = int(finger_x * w), int(finger_y * h)
                cv2.circle(black_frame, (cx, cy), 5, (0, 255, 0), -1)
                
                new_direction = get_direction_from_finger(finger_x, finger_y)
                
                if new_direction:
                    if not frame_queue.full():
                        frame_queue.put(new_direction)
        
        # Display direction zones
        h, w, c = black_frame.shape
        cv2.line(black_frame, (int(LEFT_THRESHOLD * w), 0), (int(LEFT_THRESHOLD * w), h), (64, 64, 64), 1)
        cv2.line(black_frame, (int(RIGHT_THRESHOLD * w), 0), (int(RIGHT_THRESHOLD * w), h), (64, 64, 64), 1)
        cv2.line(black_frame, (0, int(UP_THRESHOLD * h)), (w, int(UP_THRESHOLD * h)), (64, 64, 64), 1)
        cv2.line(black_frame, (0, int(DOWN_THRESHOLD * h)), (w, int(DOWN_THRESHOLD * h)), (64, 64, 64), 1)
        
        if game_state['current_direction']:
            cv2.putText(black_frame, f"Direction: {game_state['current_direction']}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Hand Tracking", black_frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Start hand tracking in a separate thread
hand_thread = Thread(target=process_hand_frame, daemon=True)
hand_thread.start()

clock = pygame.time.Clock()

def update_direction():
    try:
        new_direction = frame_queue.get_nowait()
        current_direction = game_state['direction']
        
        # Define valid direction changes
        valid_changes = {
            'UP': ['LEFT', 'RIGHT'],
            'DOWN': ['LEFT', 'RIGHT'],
            'LEFT': ['UP', 'DOWN'],
            'RIGHT': ['UP', 'DOWN']
        }
        
        # Only update direction if it's a valid change
        if new_direction in valid_changes[current_direction]:
            game_state['direction'] = new_direction
            game_state['current_direction'] = new_direction
    except queue.Empty:
        pass

def move_snake():
    global high_score
    
    x, y = game_state['snake_pos'][0]
    if game_state['direction'] == 'UP':
        y -= cell_size
    elif game_state['direction'] == 'DOWN':
        y += cell_size
    elif game_state['direction'] == 'LEFT':
        x -= cell_size
    elif game_state['direction'] == 'RIGHT':
        x += cell_size
    
    # Implement wrap-around
    x,y = x % width, y % height
    
    new_head = (x, y)
    
    # Check collision with self
    if new_head in game_state['snake_pos'][1:]:
        game_state['game_over'] = True
        return
    
    game_state['snake_pos'].insert(0, new_head)
    
    # Check if snake ate food
    if game_state['snake_pos'][0] == game_state['food_pos']:
        game_state['score'] += 1
        if game_state['score'] > high_score:
            high_score = game_state['score']
        game_state['food_pos'] = (
            random.randrange(border_thickness//cell_size, (width-border_thickness)//cell_size) * cell_size,
            random.randrange(border_thickness//cell_size, (height-border_thickness)//cell_size) * cell_size
        )
    else:
        game_state['snake_pos'].pop()

def update_timer():
    current_time = time.time()
    elapsed_time = current_time - game_state['start_time']
    game_state['time_left'] = max(0, 60 - int(elapsed_time))
    
    if game_state['time_left'] == 0:
        game_state['game_over'] = True

def draw_game():
    window.fill(BLACK)
    
    # Draw borders
    pygame.draw.rect(window, BORDER_COLOR, pygame.Rect(0, 0, width, border_thickness))  # Top
    pygame.draw.rect(window, BORDER_COLOR, pygame.Rect(0, height-border_thickness, width, border_thickness))  # Bottom
    pygame.draw.rect(window, BORDER_COLOR, pygame.Rect(0, 0, border_thickness, height))  # Left
    pygame.draw.rect(window, BORDER_COLOR, pygame.Rect(width-border_thickness, 0, border_thickness, height))  # Right
    
    # Draw grid lines for better visibility
    for x in range(border_thickness, width-border_thickness, cell_size):
        pygame.draw.line(window, (32, 32, 32), (x, border_thickness), (x, height-border_thickness))
    for y in range(border_thickness, height-border_thickness, cell_size):
        pygame.draw.line(window, (32, 32, 32), (border_thickness, y), (width-border_thickness, y))
    
    for pos in game_state['snake_pos']:
        pygame.draw.rect(window, GREEN, pygame.Rect(pos[0], pos[1], cell_size-2, cell_size-2))
    
    pygame.draw.rect(window, RED, pygame.Rect(game_state['food_pos'][0], game_state['food_pos'][1], cell_size-2, cell_size-2))
    
    font = pygame.font.Font(None, 36)
    
    # Draw current score in top left
    score_text = font.render(f'Score: {game_state["score"]}', True, WHITE)
    window.blit(score_text, (border_thickness + 10, border_thickness + 10))
    
    # Draw high score in top right
    high_score_text = font.render(f'High Score: {high_score}', True, WHITE)
    high_score_rect = high_score_text.get_rect()
    high_score_rect.topright = (width - border_thickness - 10, border_thickness + 10)
    window.blit(high_score_text, high_score_rect)
    
    # Draw timer
    timer_text = font.render(f'Time: {game_state["time_left"]}s', True, WHITE)
    timer_rect = timer_text.get_rect()
    timer_rect.midbottom = (width // 2, height - border_thickness - 10)
    window.blit(timer_text, timer_rect)
    
    if game_state['game_over']:
        game_over_text = font.render('Game Over! Press C to Play Again or Q to Quit', True, WHITE)
        text_rect = game_over_text.get_rect(center=(width/2, height/2))
        window.blit(game_over_text, text_rect)
    
    pygame.display.update()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_c and game_state['game_over']:
                game_state = init_game()
    
    if not game_state['game_over']:
        update_direction()
        move_snake()
        update_timer()
    
    draw_game()
    clock.tick(snake_speed)

# Cleanup
camera_active = False
pygame.quit()