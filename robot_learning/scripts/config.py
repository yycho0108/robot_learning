#network configuration
BATCH_SIZE=16
VAL_BATCH_SIZE=8
IMG_WIDTH=320
IMG_HEIGHT=240
IMG_DEPTH=3
TIME_STEPS=6
LSTM_SIZE=512
NUM_LSTM=2
W_COST=[1.,1.,100.]
SAVE_STEPS=100
VAL_STEPS=10
TRAIN_STEPS=2000

# learning rate

# initial ramp
LR_RAMP_0 = 1e-6
LR_RAMP_STEPS=100

LEARNING_RATE = 1e-3 # true initial learning rate
FINAL_DECAY = 0.1
NUM_DECAY   = 5
STEPS_PER_DECAY = float(TRAIN_STEPS) / (2 + NUM_DECAY)
#EPOCHS_PER_DECAY = 0.2
#STEPS_PER_EPOCH  = (2332 / BATCH_SIZE)
#STEPS_PER_DECAY  = (STEPS_PER_EPOCH * EPOCHS_PER_DECAY)
DECAY_FACTOR = FINAL_DECAY ** (1.0 / NUM_DECAY)

# Data Queue
Q_THREADS = 4
