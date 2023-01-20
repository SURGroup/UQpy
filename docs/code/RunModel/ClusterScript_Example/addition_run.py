import os
import shutil
import fire

def runAddition(index):
    index = int(index)

    inputRealizationPath = os.path.join(os.getcwd(), 'run_' + str(index), 'InputFiles', 'inputRealization_' \
                                        + str(index) + ".json")
    outputPath = os.path.join(os.getcwd(), 'OutputFiles')
    
    # This is where pre-processing commands would be executed prior to running the cluster script. 
    command1 = ("echo \"This is where pre-processing would be happening\"")
    
    os.system(command1)    

if __name__ == '__main__':
    fire.Fire(runAddition)
