from pathlib import Path

splits = ['traindev', 'test']
SNR_list = [0, -5, -10, -15, -20, -25, -30]

# Input files folder
input_files_classes = ['ambience', 'horn', 'siren', 'vehicle']
for split in splits:
    for name in input_files_classes:
        Path('Data/input_files').joinpath(split, name).mkdir(parents=True, exist_ok=True)

# Dataset folder
for split in splits:
    # noise path
    Path('Data/dataset').joinpath(split, 'noise/noise').mkdir(parents=True, exist_ok=True)

    # horn path
    for SNR in SNR_list:
        Path('Data/dataset').joinpath(split, 'horn/horn', 'SNR'+str(int(abs(SNR)))).mkdir(parents=True, exist_ok=True)
    
    # siren path
    subclasses = ['wail', 'yelp', 'hilo']
    for subclass in subclasses:
        for SNR in SNR_list:
            Path('Data/dataset').joinpath(split, 'siren', subclass, 'SNR'+str(int(abs(SNR)))).mkdir(parents=True, exist_ok=True)
