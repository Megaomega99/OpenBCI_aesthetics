"""
OpenBCI Cyton + Mountain Car BCI Controller
Uses motor imagery detection to control the Mountain Car environment
"""

import numpy as np
import gymnasium as gym
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import time
from collections import deque
from scipy.signal import butter, lfilter
import threading

class BCIMountainCarController:
    def __init__(self, serial_port='COM6', render_mode='human'):
        """
        Initialize BCI controller for Mountain Car
        
        Args:
            serial_port: COM port for OpenBCI Cyton
            render_mode: Visualization mode for gymnasium
        """
        # Initialize BrainFlow for OpenBCI Cyton
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.board_id = BoardIds.CYTON_BOARD.value
        self.board = BoardShim(self.board_id, self.params)
        
        # Get board info
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        
        # Motor cortex channels (C3, C4) - channels 3 and 4 on Cyton
        self.motor_channels = [2, 3]  # 0-indexed
        
        # Signal processing parameters
        self.buffer_size = int(self.sampling_rate * 2)  # 2 second buffer
        self.data_buffer = deque(maxlen=self.buffer_size)
        
        # Bandpass filter parameters for mu/beta rhythms (8-30 Hz)
        self.lowcut = 8.0
        self.highcut = 30.0
        
        # Initialize environment
        self.env = gym.make('MountainCar-v0', render_mode=render_mode)
        self.action_space = self.env.action_space.n
        
        # Control parameters
        self.threshold_left = 0.7  # Threshold for left motor imagery
        self.threshold_right = 0.7  # Threshold for right motor imagery
        self.baseline_window = 5  # seconds for baseline calculation
        
        # Thread control
        self.running = False
        self.data_thread = None
        
    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        """Design butterworth bandpass filter"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """Apply bandpass filter to data"""
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def calculate_band_power(self, data):
        """
        Calculate band power for motor imagery detection
        Uses event-related desynchronization (ERD) in mu/beta bands
        """
        # Apply bandpass filter
        filtered = self.bandpass_filter(data, self.lowcut, self.highcut, self.sampling_rate)
        
        # Calculate power using Welch's method approximation
        power = np.mean(filtered ** 2)
        return power
    
    def detect_motor_imagery(self, baseline_power):
        """
        Detect left/right motor imagery based on ERD
        Returns: 0 (left), 1 (neutral), 2 (right)
        """
        if len(self.data_buffer) < self.buffer_size:
            return 1, None, None  # Return neutral if buffer not full
        
        # Get recent data
        recent_data = np.array(list(self.data_buffer))
        
        # Calculate power for C3 (left motor cortex - controls right movement)
        c3_data = recent_data[:, self.motor_channels[0]]
        c3_power = self.calculate_band_power(c3_data)
        c3_erd = (baseline_power[0] - c3_power) / baseline_power[0]
        
        # Calculate power for C4 (right motor cortex - controls left movement)
        c4_data = recent_data[:, self.motor_channels[1]]
        c4_power = self.calculate_band_power(c4_data)
        c4_erd = (baseline_power[1] - c4_power) / baseline_power[1]
        
        # Store feedback data
        feedback = {
            'c3_power': c3_power,
            'c4_power': c4_power,
            'c3_erd': c3_erd,
            'c4_erd': c4_erd,
            'baseline_c3': baseline_power[0],
            'baseline_c4': baseline_power[1]
        }
        
        # Decision logic based on ERD
        if c4_erd > self.threshold_left:  # Right cortex suppression = left movement
            return 0, feedback, 'LEFT'  # Push left
        elif c3_erd > self.threshold_right:  # Left cortex suppression = right movement
            return 2, feedback, 'RIGHT'  # Push right
        else:
            return 1, feedback, 'NEUTRAL'  # No push (neutral)
    
    def collect_baseline(self):
        """Collect baseline EEG data for calibration"""
        print("Collecting baseline data. Please relax for {} seconds...".format(self.baseline_window))
        
        baseline_data = []
        start_time = time.time()
        
        while time.time() - start_time < self.baseline_window:
            data = self.board.get_board_data()
            if data.shape[1] > 0:
                eeg_data = data[self.eeg_channels, :].T
                baseline_data.extend(eeg_data)
            time.sleep(0.1)
        
        baseline_array = np.array(baseline_data)
        
        # Calculate baseline power for each motor channel
        baseline_power = []
        for ch in self.motor_channels:
            power = self.calculate_band_power(baseline_array[:, ch])
            baseline_power.append(power)
        
        print("Baseline collection complete!")
        return baseline_power
    
    def data_acquisition_thread(self):
        """Thread for continuous data acquisition"""
        while self.running:
            data = self.board.get_board_data()
            if data.shape[1] > 0:
                # Extract EEG data and add to buffer
                eeg_data = data[self.eeg_channels, :].T
                for sample in eeg_data:
                    self.data_buffer.append(sample)
            time.sleep(0.01)  # Small delay to prevent CPU overload
    
    def start(self):
        """Start the BCI-controlled Mountain Car game"""
        try:
            # Prepare session
            self.board.prepare_session()
            self.board.start_stream()
            print("OpenBCI stream started on {}".format(self.params.serial_port))
            
            # Start data acquisition thread
            self.running = True
            self.data_thread = threading.Thread(target=self.data_acquisition_thread)
            self.data_thread.start()
            
            # Collect baseline
            baseline_power = self.collect_baseline()
            
            # Instructions
            print("\nStarting Mountain Car BCI Control!")
            print("Instructions:")
            print("- Imagine moving your RIGHT hand to push the car LEFT")
            print("- Imagine moving your LEFT hand to push the car RIGHT")
            print("- Relax to let the car coast")
            print("\nPress Ctrl+C to stop\n")
            
            # Game loop - continuous play
            episode = 1
            total_steps = 0
            
            while self.running:
                observation, info = self.env.reset()
                done = False
                step_count = 0
                
                print(f"\n--- Episode {episode} ---")
                
                while not done and self.running:
                    # Detect motor imagery and get action
                    action, feedback, action_name = self.detect_motor_imagery(baseline_power)
                    
                    # Take action in environment
                    observation, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    step_count += 1
                    total_steps += 1
                    
                    # Display real-time feedback every 10 steps
                    if step_count % 10 == 0 and feedback:
                        print(f"\n[Step {step_count}] Position: {observation[0]:.3f}, Velocity: {observation[1]:.4f}")
                        print(f"├─ C3: Power={feedback['c3_power']:.4f} (Base={feedback['baseline_c3']:.4f}) ERD={feedback['c3_erd']:.2%}")
                        print(f"├─ C4: Power={feedback['c4_power']:.4f} (Base={feedback['baseline_c4']:.4f}) ERD={feedback['c4_erd']:.2%}")
                        print(f"├─ Thresholds: Left={self.threshold_left:.2f}, Right={self.threshold_right:.2f}")
                        print(f"└─ Action: {action_name} ({'▼' if action == 0 else '▲' if action == 2 else '═'})")
                    
                    # Small delay for visualization
                    time.sleep(0.05)
                
                if done:
                    print(f"Episode {episode} completed in {step_count} steps! (Total: {total_steps})")
                    episode += 1
            
        except KeyboardInterrupt:
            print("\nStopping BCI controller...")
        finally:
            self.stop()
    
    def stop(self):
        """Clean up resources"""
        self.running = False
        if self.data_thread:
            self.data_thread.join()
        
        self.board.stop_stream()
        self.board.release_session()
        self.env.close()
        print("BCI controller stopped.")

def main():
    """Main function to run BCI Mountain Car controller"""
    # Enable BrainFlow logging for debugging
    BoardShim.enable_dev_board_logger()
    
    # Create and start controller
    controller = BCIMountainCarController(serial_port='COM6', render_mode='human')
    controller.start()

if __name__ == "__main__":
    main()