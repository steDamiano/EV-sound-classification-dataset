import numpy as np
import random
from pathlib import Path
import librosa
import pyroadacoustics as pyroad
from trajectory_generator import TrajectoryGenerator
from scipy.io import wavfile

class AcousticScene:
    def __init__(
            self,
            num_static_noise_sources: int = 3,
            num_moving_noise_sources: int = 2,
            height_sources: float or None= 1,
            temperature: float = 20,
            pressure: float = 1,
            relative_humidity: int = 50
        ) -> None:
        self.num_static_noise_sources = num_static_noise_sources
        self.num_moving_noise_sources = num_moving_noise_sources
        self.height_sources = height_sources
        self.temperature = temperature
        self.pressure = pressure
        self.relative_humidity = relative_humidity

    def set_num_static_noise_sources(self, n: int):
        self.num_static_noise_sources = n
    
    def set_num_moving_noise_sources(self, n: int):
        self.num_moving_noise_sources = n
    
    def set_temperature(self, T: float):
        self.temperature = T
    
    def set_pressure(self, p: float):
        self.pressure = p
    
    def set_rel_humidity(self, rel_hum: int):
        self.relative_humidity = rel_hum
    
class Simulator:
    def __init__(
            self,
            microphone_array: np.ndarray,
            sample_duration: float = 1.0,
            sample_rate: int = 16000,
            silence_margin: float = 0.5,
            dataset_split: str = 'traindev',
            SNR_range: list = [0, -5, -10, -15, -20, -25, -30],
        ) -> None:
        self.microphone_array = microphone_array
        self.sample_duration = sample_duration
        self.sample_rate = sample_rate
        self.silence_margin = silence_margin
        self.dataset_split = dataset_split
        self.trajectory_generator = TrajectoryGenerator(min_traj_len=3)
        self.SNR_range = SNR_range

    def _generate_noise(self, acoustic_scene: AcousticScene):
        position = np.zeros((acoustic_scene.num_static_noise_sources, 3))

        noise_signal = np.zeros((len(self.microphone_array), int(self.sample_duration * self.sample_rate)))
        noise_signal_ext = np.zeros((len(self.microphone_array), int((self.sample_duration + self.silence_margin) * self.sample_rate)))

        for i in range(acoustic_scene.num_static_noise_sources):
            # Define audio file to be used as noise
            data, _, _ = self._load_audio('ambience')

            # Select T seconds from a random position in the file
            init_idx = np.random.randint(0, len(data) - (self.sample_duration + self.silence_margin) * self.sample_rate)
            noise_sig1 = data[init_idx : init_idx + int((self.sample_duration + self.silence_margin) * self.sample_rate)]

            noise_sig = self._normalize_signal(noise_sig1, 'peak')

            # Create noise source
            radius = np.random.uniform(20,80)
            height = acoustic_scene.height_sources
            theta = 2 * np.pi * random.random()
            position[i] = np.array([radius * np.cos(theta),radius * np.sin(theta),height])

            # Add source and array to simulation
            env = pyroad.Environment(fs = self.sample_rate, temperature = acoustic_scene.temperature, pressure = acoustic_scene.pressure, rel_humidity = acoustic_scene.relative_humidity)
            env.add_source(position = position[i], signal = noise_sig)
            env.add_microphone_array(self.microphone_array)

            # Define simulation parameters
            interp_method = "Allpass"
            include_reflection = False
            include_air_absorption = False

            env.set_simulation_params(interp_method, include_reflection, include_air_absorption)

            # Run simulation
            noise_signal_ext += env.simulate()

        noise_signal = noise_signal_ext[:,int(self.silence_margin*self.sample_rate):int((self.sample_duration+self.silence_margin)*self.sample_rate)]

        # Moving sources
        for i in range(acoustic_scene.num_moving_noise_sources):
            # Define audio file to be used as noise
            data, _, _ = self._load_audio('vehicle')
            noise_sig = self._preprocess_audio(data, 'vehicle') 

            # Define trajectory and speed
            trajectory_type = np.random.choice(['rectilinear', 'bezier'],1,replace=False)
            line, speed = self.trajectory_generator.generate_trajectory(trajectory_type, self.sample_duration, acoustic_scene.height_sources)
            
            # Add source and array to simulation
            env = pyroad.Environment(fs = self.sample_rate, temperature = acoustic_scene.temperature, pressure = acoustic_scene.pressure, rel_humidity = acoustic_scene.relative_humidity)
            env.add_source(position = line[0,:], trajectory_points = line, source_velocity = np.array([speed]), signal = noise_sig)
            env.add_microphone_array(self.microphone_array)

            # Define simulation parameters
            interp_method = "Allpass"
            include_reflection = False
            include_air_absorption = False

            env.set_simulation_params(interp_method, include_reflection, include_air_absorption)

            # Run simulation 
            signal = env.simulate()
            noise_signal += signal[:,:int(self.sample_duration*self.sample_rate)]
        return noise_signal

    def _generate_signal(self, acoustic_scene: AcousticScene, event_type: str = 'siren'):
        # Choose random sample from path
        data, event_label, input_file = self._load_audio(event_type)
        
        # Create audio sample with given simulation length
        source_signal = self._preprocess_audio(data, event_type)

        # Generate Trajectory
        trajectory_type = np.random.choice(['rectilinear', 'bezier'],1,replace=False)
        line, speed = self.trajectory_generator.generate_trajectory(trajectory_type, self.sample_duration, acoustic_scene.height_sources)    

        # Create simulation environment
        env = pyroad.Environment(fs = self.sample_rate, temperature = acoustic_scene.temperature, pressure = acoustic_scene.pressure, rel_humidity = acoustic_scene.relative_humidity)
        env.add_source(position = line[0,:], trajectory_points = line, source_velocity = np.array([speed]), signal = source_signal)
        env.add_microphone_array(self.microphone_array)

        # Define simulation parameters
        interp_method = "Allpass"
        include_reflection = True
        include_air_absorption = True

        env.set_simulation_params(interp_method, include_reflection, include_air_absorption)

        mic_signal = env.simulate()
        mic_signal = mic_signal[:,:int(self.sample_duration*self.sample_rate)]

        return mic_signal, event_label, input_file
    

    def _combine_signal_noise(self, source_signal: np.ndarray, noise_signal: np.ndarray, SNR):
        signal = np.zeros_like(source_signal)

        # Attenuation based on SNR computed on first channel (arbitrary reference)
        noise_attenuation = np.sqrt(np.sum(source_signal[0]** 2) / (10 ** (SNR/10) 
                * np.sum(noise_signal[0] ** 2)))

        for m in range(len(self.microphone_array)):  
            signal[m] = source_signal[m] + noise_signal[m] * noise_attenuation
    
        return signal

    def _load_audio(self, event_class: str):
        file_path = random.choice(list(Path('Data/input_files').joinpath(self.dataset_split, event_class).rglob('*.wav')))
        input_file = file_path.stem
        data, _ = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
        class_label = file_path.stem
        class_label = ''.join([i for i in class_label if not i.isdigit()])
        
        return data, class_label, input_file

    def _preprocess_audio(self, audio_signal, event_type):      
        # If file is longer than T, cut random part of file to fit desired duration
        if len(audio_signal)/self.sample_rate > self.sample_duration:
            onset = int(np.random.uniform(0,len(audio_signal) - self.sample_duration * self.sample_rate))
            signal = audio_signal[onset:onset + int(self.sample_duration*self.sample_rate)]
        # Else use all file with random onset
        else:
            k = np.random.randint(0,self.sample_duration*self.sample_rate - len(audio_signal))
            # Write audio into silent segment
            signal = np.zeros(int(self.sample_duration*self.sample_rate))
            signal[k : k + len(audio_signal)] = audio_signal
        signal = self._normalize_signal(signal, 'peak')

        return signal

    def _normalize_signal(self, signal: np.ndarray, mode: str = 'peak'):
        if mode == 'peak':
                signal =  signal / np.max(np.abs(signal))
                return signal
        elif mode == 'meanvar':
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-07)
            return signal
        else:
            raise NotImplementedError('Normalization method not supported')
    
    def simulate_datapoint(self, event_type):
        # Define acoustic scene
        acoustic_scene = AcousticScene(
            num_static_noise_sources=3,
            num_moving_noise_sources=2,
            height_sources=1,
            temperature=random.uniform(0,30),
            pressure=1,
            relative_humidity=int(random.uniform(0,100))
        )
        signal = self._generate_noise(acoustic_scene)
        label = 'noise'
        input_file = 'noise'
        SNR = None
        if event_type != 'noise':
            # Simulate data point
            source_signal, label, input_file = self._generate_signal(acoustic_scene, event_type)
            SNR = random.choice(self.SNR_range)
            signal = self._combine_signal_noise(source_signal, signal, SNR)
        return signal, label, input_file, SNR

def create_folder_structure(SNR_list):
    splits = ['traindev', 'test']

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

def generate_data(batch_num,num_samples,dataset_split,event_class,sample_duration,sample_rate):
    create_folder_structure(SNR_range)
    microphone_array = np.array([[0., 0., 1.]])
    SNR_range = [0, -5, -10, -15, -20, -25, -30]

    simulator = Simulator(microphone_array=microphone_array,
                          sample_duration=sample_duration,
                          sample_rate=sample_rate,
                          dataset_split=dataset_split,
                          SNR_range=SNR_range)
    
    # Simulate data
    for i in range(num_samples):
        audio, label, input_file, SNR = simulator.simulate_datapoint(event_class)

        file_path = Path('Data/dataset').joinpath(dataset_split,event_class,label)
        if SNR is not None:
            file_path = file_path.joinpath('SNR'+str(int(np.abs(SNR))))
        
        file_path = file_path.joinpath(str(batch_num)+'_'+str(i)+ '_'+ str(input_file) +'.wav')

        wavfile.write(file_path, sample_rate, audio.T.astype(np.float32))
        print(f'Processed {i+1}/{num_samples}')

if __name__=='__main__':
    import argparse
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-j', '--job_number', type=int, default=103)
    argParser.add_argument('-n', '--num_samples', type=int, default=1)
    argParser.add_argument('-s', '--dataset_split', type=str, default='traindev')
    argParser.add_argument('-c', '--event_class', type=str, default='noise')
    argParser.add_argument('-t', '--sample_duration', type=float, default=1.0)
    argParser.add_argument('-f', '--sample_rate', type=int, default=16000)
    args = argParser.parse_args()

    generate_data(batch_num=args.job_number,
                  num_samples=args.num_samples,
                  dataset_split=args.dataset_split,
                  event_class=args.event_class,
                  sample_duration=args.sample_duration,
                  sample_rate=args.sample_rate,
    )
