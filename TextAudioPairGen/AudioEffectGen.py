import numpy as np
from scipy.signal import convolve
import librosa

class AudioEffect_Generater:
    def __init__(self, config):
        self.config = config

        self.rir = np.load('reverb_rir.npy')

    def simulate_spatial_movement(self, audio, relation, volume_type="exponential"):
        """
        Simulate an audio event moving closer or farther away for a monochannel waveform,
        with volume change based on the specified type (linear or exponential).
        
        :param audio: The input audio segment (pydub AudioSegment).
        :param move_towards: If True, the audio moves closer; otherwise, it moves farther away.
        :param volume_type: Type of volume change ("linear" or "exponential").
        :return: Processed AudioSegment with the spatial effect applied.
        """
        assert relation in ['Approaching', 'Departuring'], "Invalid relation. Choose 'Approaching' or 'Departuring'."
        num_samples = len(audio)
        
        # Create volume curve based on movement and volume type
        if volume_type == "linear":
            # Linear change in volume (closer = increase, farther = decrease)
            k = 10
            if relation in ['Approaching']:
                volume_curve = k*np.linspace(0.5, 1.5, num_samples)  # Moving closer, increasing volume
            else:
                volume_curve = k*np.linspace(1.5, 0.5, num_samples)  # Moving farther
        elif volume_type == "exponential":
            # Exponential change in volume
            k = 4  # Adjust rate constant 'k' for exponential change
            if relation in ['Approaching']:
                time_steps = np.linspace(0., 1., num_samples)  # Time steps in normalized range
                # Moving closer (exponential growth)
                volume_curve = np.exp(time_steps*k) - 1  # Exponential increase
            else:
                time_steps = np.linspace(1., 0., num_samples)  # Time steps in normalized range
                # Moving farther (exponential decay)
                volume_curve = np.exp(time_steps * k) - 1  # Exponential decay
        else:
            raise ValueError("Invalid volume_type. Choose 'linear' or 'exponential'.")

        # Apply the volume modulation to the audio samples
        samples = audio * volume_curve
        
        # Ensure samples are within valid range for audio data (e.g., 16-bit integer range)
        audio_with_effect = np.clip(samples, -1., 1.)

        return audio_with_effect
    
    def simulate_balancing_effect(self, audio_list):
        """
        given two audios, increase the volume of the first audio and decrease the volume of the second audio
        """
        audio1, audio2 = audio_list[0], audio_list[1]
        audio1 = audio1 * self.config['DATA_CREATION_CONFIG']['Perceptuality_Config']['balancing_increase_ratio']
        audio2 = audio2 * self.config['DATA_CREATION_CONFIG']['Perceptuality_Config']['balancing_decrease_ratio']

        audio1 = np.clip(audio1, -1., 1.)
        audio2 = np.clip(audio2, -1., 1.)

        return audio1, audio2
    
    def normalize_audio(self, audio):
        """
        Normalize an audio segment to have its peak amplitude at 0 dBFS.
        :param audio: The input audio segment (pydub AudioSegment).
        :return: Normalized AudioSegment.
        """
        assert np.max(np.abs(audio)) > 0, "Audio segment must have non-zero amplitude."
        normalized_audio = audio/np.max(np.abs(audio))

        return normalized_audio
    
    def simulate_blending_effect(self, audio_list):
        """
        given two audios, blend them together so that they acoustically indistinguishable
        """
        audio1_normalized = self.normalize_audio(audio_list[0])
        audio2_normalized = self.normalize_audio(audio_list[1])

        return audio1_normalized, audio2_normalized
        
        # # Ensure both audios are of the same length
        # min_len = min(len(audio1_normalized), len(audio2_normalized))
        # audio1_normalized = audio1_normalized[:min_len]
        # audio2_normalized = audio2_normalized[:min_len]

        # blended_audio = np.stack([audio1_normalized, audio2_normalized], axis=0)

        # return blended_audio
    
    def simulate_timestretching_effect(self, audio):
        """
        Apply time stretching effect to an audio segment by changing its playback speed.
        :param audio: The input audio segment (pydub AudioSegment).
        :param factor: Factor by which to stretch or compress the audio (e.g., 0.5 for half speed).
        :return: Processed AudioSegment with the time stretching effect applied.
        """
        # the stretched audio is temporally longer than the input audio
        slowdown_factor = 4.
        assert slowdown_factor > 1, "Slowdown factor must be greater than 1."
        stretched_audio = librosa.effects.time_stretch(audio, rate = 1./slowdown_factor)

        stretch_audio_sec = int(float(16000 + 0.5))
        stretch_audio_sec = 5 if stretch_audio_sec > 5 else stretch_audio_sec
        if stretched_audio.shape[0] < stretch_audio_sec*16000:
            stretched_audio = np.pad(stretched_audio, (0, stretch_audio_sec*16000 - stretched_audio.shape[0]), 'constant')
        else:
            stretched_audio = stretched_audio[:stretch_audio_sec*16000]

        return stretched_audio
    
    def simulate_amplification_effect(self, audio):
        """
        Apply amplification effect to an audio segment by increasing its volume.
        """
        amplify_factor = self.config['DATA_CREATION_CONFIG']['Perceptuality_Config']['amplification_ratio']
        assert amplify_factor > 1, "Amplification factor must be greater than 1."
        amplified_audio = audio * amplify_factor
        amplified_audio = np.clip(amplified_audio, -1., 1.)

        return amplified_audio
    
    def simulate_attenuation_effect(self, audio):
        """
        Apply attenuation effect to an audio segment by decreasing its volume.
        
        :param audio: The input audio segment (pydub AudioSegment).
        :param factor: Factor by which to attenuate the audio (e.g., 0.5 for halving the volume).
        :return: Processed AudioSegment with the attenuation effect applied.
        """
        # Apply attenuation effect using pydub's apply_gain method
        atten_factor = self.config['DATA_CREATION_CONFIG']['Perceptuality_Config']['attenuation_ratio']
        assert atten_factor < 1, "Attenuation factor must be less than 1."
        audio /= np.max(np.abs(audio))
        attenuated_audio = audio * atten_factor

        return attenuated_audio
    
    def simulate_reverberation_effect(self, audio):
        #load RIR data
        rir = np.load('reverb_rir.npy')

        reverb_audio = convolve(audio, rir, mode='full')
        #round to the nearest integer
        reverb_audio_sec = int(float(reverb_audio.shape[0])/16000. + 0.5)
        reverb_audio_sec = 5 if reverb_audio_sec > 5 else reverb_audio_sec
        if reverb_audio.shape[0] < reverb_audio_sec*16000:
            reverb_audio = np.pad(reverb_audio, (0, reverb_audio_sec*16000 - reverb_audio.shape[0]), 'constant')
        else:
            reverb_audio = reverb_audio[:reverb_audio_sec*16000]

        return reverb_audio