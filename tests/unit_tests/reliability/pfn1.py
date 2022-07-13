def model_i(samples):
    resistance = samples[0, 0]
    stress = samples[0, 1]
    return resistance - stress