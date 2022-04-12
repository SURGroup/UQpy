import logging

import numpy as np


class SerialExecution:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run(self, model, n_existing_simulations, n_new_simulations, samples):
        results = []
        for i in range(n_existing_simulations, n_existing_simulations + n_new_simulations):
            sample = model.preprocess_single_sample(i, samples)

            execution_output = model.execute_single_sample(i, sample)

            results.append(model.postprocess_single_file(i, execution_output))

        self.logger.info("\nUQpy: Serial execution of the python model complete.\n")
        return results
