# Model parameters
gen_X_scale = 32
gen_Y_scale = 32

in_channels = 3

model_name = "ALLI_AI.pth"


# Training parameters
train_timesteps = 1000
current_generation = 0
genIncrease = 0

def IncreaseGen(r):
    global genIncrease
    genIncrease = r

# Generation paraments
gen_timesteps = 1000
image_X_scale = 32
image_Y_scale = 32
image_shape = (3, image_X_scale, image_Y_scale)


image_gen_count = 0
training_gen_count = 0

def image_Gen_Count():
    global image_gen_count
    image_gen_count += 1
    return image_gen_count

def training_Gen_Count():
    global training_gen_count
    training_gen_count += 1
    return training_gen_count

def current_Generation():
    global current_generation
    current_generation += genIncrease
    return current_generation