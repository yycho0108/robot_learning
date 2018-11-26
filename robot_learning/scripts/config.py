# == Common ==
IMG_WIDTH=256
IMG_HEIGHT=192
IMG_DEPTH=3
Q_THREADS=8 # Data Queue
# ===========

# == VONet Configuration == 

# -- Technical --
LSTM_SIZE=512
NUM_LSTM=2
W_COST=[0.1,0.1,100.0]

# -- Training --
BATCH_SIZE=8
VAL_BATCH_SIZE=8
TIME_STEPS=8
SAVE_STEPS=100
VAL_STEPS=10
TRAIN_STEPS=2000
LOG_STEPS=100

# -- Learning Rate --
# initial ramp
LR_RAMP_0 = 1e-6
LR_RAMP_STEPS = 100

# initial / constant learning rate
LEARNING_RATE = 1e-3

# learning rate decay configuration
# (output = STEPS_PER_DECAY, DECAY_FACTOR)
FINAL_DECAY = 0.1
NUM_DECAY   = 5
STEPS_PER_DECAY = float(TRAIN_STEPS) / (1 + NUM_DECAY)
DECAY_FACTOR = FINAL_DECAY ** (1.0 / NUM_DECAY)
# (Deprecated)
# EPOCHS_PER_DECAY = 0.2
# STEPS_PER_EPOCH  = (2332 / BATCH_SIZE)
# STEPS_PER_DECAY  = (STEPS_PER_EPOCH * EPOCHS_PER_DECAY)
# ========================= 

# == FlowNet Params ==

# -- Technical -- 
FN_USE_XCOR=True
FN_ERR_DECAY=4.0

# -- Training --
FN_BATCH_SIZE=16
FN_TRAIN_STEPS=80000
FN_STOP=80000

# -- Learning Rate --
FN_RAMP_0 = 1e-6
FN_RAMP_STEPS = int(1e3)
FN_LEARNING_RATE=1e-4
FN_STEPS_PER_DECAY = float(FN_TRAIN_STEPS) / (1 + NUM_DECAY)
FN_DECAY_FACTOR = FINAL_DECAY ** (1.0 / NUM_DECAY)
# ====================
