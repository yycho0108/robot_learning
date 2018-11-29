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
W_COST=[1.0, 5.0, 4.0]

# -- Training --
BATCH_SIZE=16
VAL_BATCH_SIZE=8
TIME_STEPS=16
SAVE_STEPS=100
VAL_STEPS=10
TRAIN_STEPS=10000
LOG_STEPS=100
FREEZE_CNN=True

# -- Learning Rate --
# initial ramp
LR_RAMP_0 = 1e-6
LR_RAMP_STEPS = 1000

# initial / constant learning rate
LEARNING_RATE = 1e-3

# learning rate decay configuration
# (output = STEPS_PER_DECAY, DECAY_FACTOR)
FINAL_DECAY = 0.1
NUM_DECAY   = 5
LR_DECAY_STEPS = 10000
STEPS_PER_DECAY = float(LR_DECAY_STEPS) / (1 + NUM_DECAY)
DECAY_FACTOR = FINAL_DECAY ** (1.0 / NUM_DECAY)
# (Deprecated)
# EPOCHS_PER_DECAY = 0.2
# STEPS_PER_EPOCH  = (2332 / BATCH_SIZE)
# STEPS_PER_DECAY  = (STEPS_PER_EPOCH * EPOCHS_PER_DECAY)
# ========================= 

# == FlowNet Params ==

# -- Technical -- 
FN_USE_XCOR=True
FN_ERR_SCALE=4.0
FN_ERR_SCALE_DECAY_FACTOR=0.5
FN_ERR_SCALE_DECAY_STEPS=3000

# -- Training --
FN_BATCH_SIZE=16
FN_TRAIN_STEPS=1000000

# -- Learning Rate --
FN_LR_RAMP_0 = 1e-6
FN_LR_RAMP_STEPS = int(1e3)
FN_LR0=2e-4 # initial learning rate
FN_LR1=1e-5 # final learning rate
FN_LR_DECAY_STEPS=200000
FN_LR_STEPS_PER_DECAY = float(FN_LR_DECAY_STEPS) / (NUM_DECAY)
FN_LR_DECAY_FACTOR = (FN_LR1 / FN_LR0) ** (1.0 / NUM_DECAY)
# ====================
