"""
OpenBCI Hardware Interface for Real Device Connection
=====================================================

Production-ready interface for OpenBCI Cyton/Cyton+Daisy boards.
Implements robust error handling, automatic device detection, and
real-time data acquisition with professional logging.

Author: OpenBCI Hardware Integration Module
Date: June 2025
Version: 1.0.0

Requirements:
    - brainflow >= 5.10.0
    - pyserial >= 3.5
    - numpy >= 1.21.0
    - matplotlib >= 3.4.0
"""

import logging
import sys
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, List, Optional, 
    Protocol, Tuple, Type, Union, cast
)

import numpy as np
import serial.tools.list_ports
from brainflow import (
    BoardIds, BoardShim, BrainFlowError, 
    BrainFlowInputParams, DataFilter, FilterTypes, 
    LogLevels, WindowOperations
)
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'openbci_session_{datetime.now():%Y%m%d_%H%M%S}.log')
    ]
)

logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class OpenBCIError(Exception):
    """Base exception for OpenBCI operations"""
    pass

class DeviceNotFoundError(OpenBCIError):
    """Raised when OpenBCI device cannot be found"""
    pass

class ConfigurationError(OpenBCIError):
    """Raised when device configuration fails"""
    pass

class DataAcquisitionError(OpenBCIError):
    """Raised when data acquisition fails"""
    pass


class BoardType(Enum):
    """Supported OpenBCI board types"""
    CYTON = "cyton"
    CYTON_DAISY = "cyton_daisy"
    GANGLION = "ganglion"
    SYNTHETIC = "synthetic"  # For testing without hardware
    
    @property
    def board_id(self) -> int:
        """Get BrainFlow board ID"""
        mapping = {
            BoardType.CYTON: BoardIds.CYTON_BOARD.value,
            BoardType.CYTON_DAISY: BoardIds.CYTON_DAISY_BOARD.value,
            BoardType.GANGLION: BoardIds.GANGLION_BOARD.value,
            BoardType.SYNTHETIC: BoardIds.SYNTHETIC_BOARD.value
        }
        return mapping[self]
    
    @property
    def n_channels(self) -> int:
        """Get number of EEG channels"""
        mapping = {
            BoardType.CYTON: 8,
            BoardType.CYTON_DAISY: 16,
            BoardType.GANGLION: 4,
            BoardType.SYNTHETIC: 8
        }
        return mapping[self]


@dataclass
class OpenBCIConfig:
    """Configuration for OpenBCI device"""
    board_type: BoardType = BoardType.CYTON
    serial_port: Optional[str] = None  # Auto-detect if None
    serial_number: Optional[str] = None  # For WiFi/Bluetooth
    ip_address: Optional[str] = None  # For WiFi shield
    ip_port: Optional[int] = None  # For WiFi shield
    sampling_rate: Optional[int] = None  # Use board default if None
    gain: int = 24  # Amplifier gain (1, 2, 4, 6, 8, 12, 24)
    
    # Advanced settings
    timeout: float = 15.0  # Connection timeout in seconds
    buffer_size: int = 450000  # BrainFlow ring buffer size
    log_level: int = LogLevels.LEVEL_INFO.value
    
    # Channel settings
    channels_off: List[int] = field(default_factory=list)  # Channels to disable
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        valid_gains = [1, 2, 4, 6, 8, 12, 24]
        if self.gain not in valid_gains:
            raise ConfigurationError(f"Invalid gain {self.gain}. Must be one of {valid_gains}")
        
        if self.board_type == BoardType.SYNTHETIC:
            logger.warning("Using synthetic board for testing - no real hardware required")


class DeviceDetector:
    """Automatic detection of OpenBCI devices"""
    
    VENDOR_IDS = {
        0x0403: "FTDI",  # Common for OpenBCI
        0x10C4: "Silicon Labs",  # Alternative
    }
    
    PRODUCT_IDS = {
        0x6001: "FT232R USB UART",  # OpenBCI V3
        0x6015: "FT230X Basic UART",  # Some variants
    }
    
    @classmethod
    def find_openbci_ports(cls) -> List[Tuple[str, str]]:
        """
        Find potential OpenBCI serial ports
        
        Returns:
            List of (port, description) tuples
        """
        potential_ports = []
        
        for port in serial.tools.list_ports.comports():
            # Check by VID/PID
            if (port.vid in cls.VENDOR_IDS and 
                port.pid in cls.PRODUCT_IDS):
                potential_ports.append((port.device, port.description))
                logger.info(f"Found potential OpenBCI device: {port.device} - {port.description}")
            
            # Check by description keywords
            elif any(keyword in (port.description or "").lower() 
                    for keyword in ["openbci", "ftdi", "uart"]):
                potential_ports.append((port.device, port.description))
                logger.info(f"Found potential device by description: {port.device} - {port.description}")
        
        return potential_ports
    
    @classmethod
    def auto_detect(cls, board_type: BoardType) -> Optional[str]:
        """
        Auto-detect OpenBCI serial port
        
        Args:
            board_type: Expected board type
            
        Returns:
            Serial port path or None if not found
        """
        ports = cls.find_openbci_ports()
        
        if not ports:
            logger.warning("No potential OpenBCI devices found")
            return None
        
        if len(ports) == 1:
            port, desc = ports[0]
            logger.info(f"Auto-detected single device: {port}")
            return port
        
        # Multiple devices found - try to connect to each
        logger.info(f"Found {len(ports)} potential devices, testing connections...")
        
        for port, desc in ports:
            if cls._test_connection(port, board_type):
                logger.info(f"Successfully connected to {port}")
                return port
        
        logger.warning("Could not establish connection to any detected device")
        return None
    
    @staticmethod
    def _test_connection(port: str, board_type: BoardType) -> bool:
        """Test if port has valid OpenBCI device"""
        try:
            params = BrainFlowInputParams()
            params.serial_port = port
            
            board = BoardShim(board_type.board_id, params)
            board.prepare_session()
            board.release_session()
            return True
            
        except Exception as e:
            logger.debug(f"Failed to connect to {port}: {e}")
            return False


class DataProcessor:
    """Real-time data processing pipeline"""
    
    def __init__(self, sampling_rate: int, n_channels: int):
        """
        Initialize data processor
        
        Args:
            sampling_rate: Sampling frequency in Hz
            n_channels: Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        
        # Filter parameters
        self.notch_freq = 60.0  # or 50.0 for Europe
        self.bandpass_low = 1.0
        self.bandpass_high = 50.0
        
        # Initialize filters
        self._init_filters()
        
    def _init_filters(self) -> None:
        """Initialize signal filters"""
        # Create filter coefficients
        nyquist = self.sampling_rate / 2
        
        # Bandpass filter
        self.bp_sos = signal.butter(
            4, [self.bandpass_low/nyquist, self.bandpass_high/nyquist], 
            btype='band', output='sos'
        )
        
        # Notch filter
        self.notch_b, self.notch_a = signal.iirnotch(
            self.notch_freq, 30, self.sampling_rate
        )
    
    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Process single EEG sample
        
        Args:
            sample: Raw EEG data (n_channels,)
            
        Returns:
            Processed EEG data
        """
        # Apply notch filter
        filtered = signal.filtfilt(self.notch_b, self.notch_a, sample)
        
        # Apply bandpass filter
        filtered = signal.sosfiltfilt(self.bp_sos, filtered)
        
        return filtered
    
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process a chunk of EEG data.

        Args:
            chunk: Raw EEG data (n_channels, n_samples).

        Returns:
            Processed EEG data.
        """
        if chunk.ndim == 1 or chunk.shape[1] == 0:
            return chunk
        
        # Apply notch and bandpass filters along the time axis (axis=1)
        try:
            # filtfilt requires data to be > 3 * filter_order
            if chunk.shape[1] > 15:
                filtered = signal.filtfilt(self.notch_b, self.notch_a, chunk, axis=1)
                processed_chunk = signal.sosfiltfilt(self.bp_sos, filtered, axis=1)
                return processed_chunk
            return chunk
        except ValueError:
            # filtfilt can fail if the data chunk is too short
            logger.debug("Skipping filtering for short data chunk.")
            return chunk

    def compute_band_powers(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute power in standard frequency bands from a data chunk.
        """
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        band_powers = {}
        nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)

        for ch in range(self.n_channels):
            # Compute PSD for the entire signal of the channel once
            psd = DataFilter.get_psd_welch(
                data[ch],
                nfft=nfft,
                overlap=nfft // 2,
                sampling_rate=self.sampling_rate,
                window=WindowOperations.BLACKMAN_HARRIS
            )
            psd_values = psd[0]
            freq_range = psd[1]
            
            for band_name, (low, high) in bands.items():
                if band_name not in band_powers:
                    band_powers[band_name] = np.zeros(self.n_channels)
                
                try:
                    mask = (freq_range >= low) & (freq_range <= high)
                    power = np.trapz(psd_values[mask], freq_range[mask])
                    band_powers[band_name][ch] = power
                except Exception as e:
                    logger.warning(f"Could not compute {band_name} power for ch {ch}: {e}")
                    band_powers[band_name][ch] = np.nan

        return band_powers


class OpenBCIDevice:
    """Main interface for OpenBCI hardware"""
    
    def __init__(self, config: OpenBCIConfig):
        """
        Initialize OpenBCI device interface
        
        Args:
            config: Device configuration
        """
        self.config = config
        self.config.validate()
        
        self.board: Optional[BoardShim] = None
        self.is_streaming = False
        self._marker_channel: Optional[int] = None
        self._callbacks: List[Callable[[np.ndarray], None]] = []
        
        # Data processor
        self.processor: Optional[DataProcessor] = None
        
        # Initialize BrainFlow logging
        BoardShim.set_log_level(config.log_level)
        BoardShim.set_log_file(f'brainflow_{datetime.now():%Y%m%d_%H%M%S}.log')
        
        logger.info(f"OpenBCI device initialized with {config.board_type.name}")
    
    @contextmanager
    def session(self) -> Generator['OpenBCIDevice', None, None]:
        """Context manager for device session"""
        try:
            self.connect()
            yield self
        finally:
            self.disconnect()
    
    def connect(self) -> None:
        """Connect to OpenBCI device"""
        if self.board is not None:
            logger.warning("Device already connected")
            return
        
        # Auto-detect serial port if needed
        if (self.config.serial_port is None and 
            self.config.board_type != BoardType.SYNTHETIC):
            detected_port = DeviceDetector.auto_detect(self.config.board_type)
            if detected_port:
                self.config.serial_port = detected_port
            else:
                raise DeviceNotFoundError(
                    "No OpenBCI device found. Please check connections."
                )
        
        # Prepare connection parameters
        params = BrainFlowInputParams()
        if self.config.serial_port:
            params.serial_port = self.config.serial_port
        if self.config.serial_number:
            params.serial_number = self.config.serial_number
        if self.config.ip_address:
            params.ip_address = self.config.ip_address
        if self.config.ip_port:
            params.ip_port = self.config.ip_port
        
        # Create board instance
        try:
            self.board = BoardShim(self.config.board_type.board_id, params)
            
            # Prepare session with timeout
            start_time = time.time()
            while time.time() - start_time < self.config.timeout:
                try:
                    self.board.prepare_session()
                    break
                except BrainFlowError as e:
                    if "ALREADY_PREPARED" in str(e):
                        break
                    time.sleep(0.5)
            else:
                raise ConfigurationError(
                    f"Connection timeout after {self.config.timeout} seconds"
                )
            
            # Get board info
            self._setup_board_info()
            
            # Configure board
            self._configure_board()
            
            # Initialize data processor
            sampling_rate = self.board.get_sampling_rate(self.config.board_type.board_id)
            self.processor = DataProcessor(sampling_rate, self.config.board_type.n_channels)
            
            logger.info(f"Connected to {self.config.board_type.name} on {self.config.serial_port}")
            
        except BrainFlowError as e:
            self.board = None
            raise ConfigurationError(f"Failed to connect: {e}")
    
    def _setup_board_info(self) -> None:
        """Setup board-specific information"""
        if not self.board:
            return
        
        board_id = self.config.board_type.board_id
        
        # Get channel information
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.aux_channels = BoardShim.get_accel_channels(board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(board_id)
        
        # Marker channel (if supported)
        try:
            self._marker_channel = BoardShim.get_marker_channel(board_id)
        except BrainFlowError:
            logger.warning("Board does not support marker channel")
            self._marker_channel = None
        
        logger.info(f"Board info - EEG channels: {len(self.eeg_channels)}, "
                   f"Aux channels: {len(self.aux_channels)}")
    
    def _configure_board(self) -> None:
        """Configure board settings"""
        if not self.board:
            return
        
        # Board-specific configuration commands
        if self.config.board_type in [BoardType.CYTON, BoardType.CYTON_DAISY]:
            # Set channel gains
            gain_command = {
                1: '0', 2: '1', 4: '2', 6: '3',
                8: '4', 12: '5', 24: '6'
            }
            
            if self.config.gain in gain_command:
                # Set gain for all channels
                for ch in range(1, self.config.board_type.n_channels + 1):
                    command = f'x{ch:X}{gain_command[self.config.gain]}110X'
                    self.board.config_board(command)
                    time.sleep(0.1)
                
                logger.info(f"Set gain to {self.config.gain}x for all channels")
            
            # Turn off channels if specified
            for ch in self.config.channels_off:
                if 1 <= ch <= self.config.board_type.n_channels:
                    command = f'{ch}'  # Number key turns off channel
                    self.board.config_board(command)
                    logger.info(f"Disabled channel {ch}")
    
    def start_streaming(self) -> None:
        """Start data streaming"""
        if not self.board:
            raise RuntimeError("Device not connected")
        
        if self.is_streaming:
            logger.warning("Already streaming")
            return
        
        try:
            self.board.start_stream(self.config.buffer_size)
            self.is_streaming = True
            logger.info("Started data streaming")
            
        except BrainFlowError as e:
            raise DataAcquisitionError(f"Failed to start streaming: {e}")
    
    def stop_streaming(self) -> None:
        """Stop data streaming"""
        if not self.board or not self.is_streaming:
            return
        
        try:
            self.board.stop_stream()
            self.is_streaming = False
            logger.info("Stopped data streaming")
            
        except BrainFlowError as e:
            logger.error(f"Error stopping stream: {e}")
    
    def get_data(self, n_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get data from device buffer
        
        Args:
            n_samples: Number of samples to retrieve (None for all available)
            
        Returns:
            Data array or None if no data available
        """
        if not self.board or not self.is_streaming:
            return None
        
        try:
            if n_samples:
                data = self.board.get_board_data(n_samples)
            else:
                data = self.board.get_board_data()
            
            if data.shape[1] == 0:
                return None
            
            return data
            
        except BrainFlowError as e:
            logger.error(f"Error getting data: {e}")
            return None
    
    def insert_marker(self, marker: float) -> None:
        """
        Insert marker into data stream
        
        Args:
            marker: Marker value
        """
        if not self.board or not self._marker_channel:
            logger.warning("Marker insertion not supported")
            return
        
        try:
            self.board.insert_marker(marker)
            logger.debug(f"Inserted marker: {marker}")
            
        except BrainFlowError as e:
            logger.error(f"Error inserting marker: {e}")
    
    def add_data_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Add callback for real-time data processing"""
        self._callbacks.append(callback)
    
    def remove_data_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Remove data callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def process_data_stream(self, duration: Optional[float] = None) -> None:
        """
        Process data stream with callbacks
        
        Args:
            duration: Processing duration in seconds (None for infinite)
        """
        if not self.is_streaming:
            raise RuntimeError("Not streaming")
        
        start_time = time.time()
        
        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Get new data
                data = self.get_data()
                if data is not None and data.shape[1] > 0:
                    # Process with callbacks
                    for callback in self._callbacks:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("Data processing interrupted by user")
    
    def save_data(self, filename: str, data: Optional[np.ndarray] = None) -> None:
        """
        Save data to file
        
        Args:
            filename: Output filename
            data: Data to save (if None, saves all buffered data)
        """
        if data is None:
            data = self.get_data()
        
        if data is None or data.shape[1] == 0:
            logger.warning("No data to save")
            return
        
        # Save as CSV with BrainFlow
        try:
            DataFilter.write_file(
                data, filename, 'w'
            )
            logger.info(f"Saved {data.shape[1]} samples to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from device"""
        if self.is_streaming:
            self.stop_streaming()
        
        if self.board:
            try:
                self.board.release_session()
                logger.info("Disconnected from device")
            except BrainFlowError as e:
                logger.error(f"Error during disconnect: {e}")
            finally:
                self.board = None


class RealTimeVisualizer:
    """Advanced real-time EEG visualization with per-channel subplots."""
    
    def __init__(self, device: OpenBCIDevice, window_duration: float = 5.0):
        """
        Initialize visualizer
        
        Args:
            device: OpenBCI device instance
            window_duration: Display window in seconds
        """
        self.device = device
        self.window_duration = window_duration
        
        # Get device info
        self.n_channels = device.config.board_type.n_channels
        self.sampling_rate = BoardShim.get_sampling_rate(
            device.config.board_type.board_id
        )
        
        # Data buffer
        self.buffer_size = int(self.sampling_rate * window_duration)
        self.data_buffer = np.zeros((self.n_channels, self.buffer_size))
        self.time_buffer = np.linspace(-window_duration, 0, self.buffer_size)
        
        # Setup figure
        self.fig, self.axes = None, None
        self.lines = []
        self._setup_figure()
        
        # Animation
        self.animation: Optional[FuncAnimation] = None
        
        logger.info(f"Visualizer initialized for {self.n_channels} channels, each in its own subplot.")
    
    def _setup_figure(self) -> None:
        """Setup matplotlib figure with a subplot for each channel."""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create a figure and a set of subplots
        self.fig, self.axes = plt.subplots(
            self.n_channels, 1, 
            figsize=(12, 2 * self.n_channels), 
            sharex=True,
            gridspec_kw={'hspace': 0.5}
        )
        
        # If there's only one channel, axes is not a list, so make it one
        if self.n_channels == 1:
            self.axes = [self.axes]
            
        self.fig.suptitle('Real-Time EEG Data (Per-Channel)', fontsize=16)
        
        for i, ax in enumerate(self.axes):
            line, = ax.plot(self.time_buffer, self.data_buffer[i], color='c', linewidth=1.0)
            self.lines.append(line)
            
            ax.set_ylabel(f'Ch {i+1}\n(μV)', rotation=0, labelpad=30, va='center')
            ax.set_xlim(-self.window_duration, 0)
        
        self.axes[-1].set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update_visualization(self, data: np.ndarray) -> None:
        """Update visualization with new data."""
        if data.shape[1] == 0:
            return
        
        # Extract and process EEG data
        eeg_data = data[self.device.eeg_channels, :]
        processed_eeg_data = eeg_data
        if self.device.processor:
            processed_eeg_data = self.device.processor.process_chunk(eeg_data)

        # Update data buffer
        n_new_samples = min(processed_eeg_data.shape[1], self.buffer_size)
        self.data_buffer = np.roll(self.data_buffer, -n_new_samples, axis=1)
        self.data_buffer[:, -n_new_samples:] = processed_eeg_data[:self.n_channels, -n_new_samples:]
        
        # Update EEG lines and autoscale each subplot
        for i, line in enumerate(self.lines):
            line.set_ydata(self.data_buffer[i])
            ax = self.axes[i]
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

    def start(self) -> None:
        """Start real-time visualization."""
        def animation_update(frame):
            data = self.device.get_data()
            if data is not None and data.shape[1] > 0:
                self.update_visualization(data)
            
            # Return the updated line artists
            return self.lines
        
        self.animation = FuncAnimation(
            self.fig, 
            animation_update,
            interval=100,  # Refresh rate in ms
            blit=True,     # Use blitting for performance
            cache_frame_data=False
        )
        
        plt.show()
    
    def stop(self) -> None:
        """Stop visualization."""
        if self.animation:
            self.animation.event_source.stop()


def test_hardware_connection():
    """Test basic hardware connection and data acquisition"""
    
    print("\n" + "="*80)
    print("OpenBCI Hardware Connection Test")
    print("="*80 + "\n")
    
    # Detect available devices
    print("Scanning for OpenBCI devices...")
    ports = DeviceDetector.find_openbci_ports()
    
    if not ports:
        print("No devices found. Please check:")
        print("- Device is powered on")
        print("- USB/Serial cable is connected")
        print("- Drivers are installed")
        return
    
    print(f"\nFound {len(ports)} potential device(s):")
    for i, (port, desc) in enumerate(ports):
        print(f"{i+1}. {port} - {desc}")
    
    # Select board type
    print("\nSelect board type:")
    print("1. Cyton (8 channels)")
    print("2. Cyton + Daisy (16 channels)")
    print("3. Synthetic (test without hardware)")
    
    choice = input("\nEnter choice (1-3): ")
    
    board_type_map = {
        '1': BoardType.CYTON,
        '2': BoardType.CYTON_DAISY,
        '3': BoardType.SYNTHETIC
    }
    
    board_type = board_type_map.get(choice, BoardType.SYNTHETIC)
    
    # Create configuration
    config = OpenBCIConfig(
        board_type=board_type,
        gain=24  # Standard gain
    )
    
    # Create device
    device = OpenBCIDevice(config)
    
    try:
        # Connect
        print(f"\nConnecting to {board_type.name}...")
        device.connect()
        print("✓ Connected successfully!")
        
        # Start streaming
        print("\nStarting data stream...")
        device.start_streaming()
        print("✓ Streaming started!")
        
        # Collect some data
        print("\nCollecting data for 5 seconds...")
        time.sleep(5)
        
        # Get data
        data = device.get_data()
        if data is not None:
            print(f"✓ Received {data.shape[1]} samples")
            print(f"  Channels: {len(device.eeg_channels)}")
            print(f"  Sampling rate: {device.board.get_sampling_rate(board_type.board_id)} Hz")
            
            # Save sample data
            filename = f"test_data_{datetime.now():%Y%m%d_%H%M%S}.csv"
            device.save_data(filename, data)
            print(f"✓ Saved test data to {filename}")
        else:
            print("✗ No data received")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        logger.exception("Connection test failed")
        
    finally:
        # Cleanup
        print("\nCleaning up...")
        device.disconnect()
        print("✓ Disconnected")


def run_realtime_visualization():
    """Run real-time visualization with hardware"""
    
    print("\n" + "="*80)
    print("Real-time EEG Visualization")
    print("="*80 + "\n")
    
    # Let user select gain
    print("Select amplifier gain. If you see square waves, the gain is too high.")
    print("Valid gains: 1, 2, 4, 6, 8, 12, 24")
    gain_choice = input("Enter gain (default is 8): ")
    try:
        gain = int(gain_choice)
        if gain not in [1, 2, 4, 6, 8, 12, 24]:
            print("Invalid gain, defaulting to 8x.")
            gain = 8
    except ValueError:
        print("Invalid input, defaulting to 8x.")
        gain = 8

    # Configuration
    config = OpenBCIConfig(
        board_type=BoardType.CYTON,  # Change to CYTON_DAISY for 16 channels
        gain=gain
    )
    
    # Create device
    device = OpenBCIDevice(config)
    
    try:
        # Connect and start streaming
        device.connect()
        device.start_streaming()
        
        # Create visualizer
        visualizer = RealTimeVisualizer(device, window_duration=5.0)
        
        print("\nStarting visualization...")
        print("If signals still look square, try restarting with a lower gain.")
        print("Also, ensure electrodes have good contact.")
        print("Press Ctrl+C in the terminal to stop.")
        
        # Start visualization (blocking)
        visualizer.start()
        
    except KeyboardInterrupt:
        print("\nStopping...")
        
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Visualization error")
        
    finally:
        device.disconnect()


def main():
    """Main entry point for hardware testing"""
    
    print("OpenBCI Hardware Interface")
    print("1. Test hardware connection")
    print("2. Real-time visualization")
    print("3. Run with your protocol")
    
    choice = input("\nSelect option (1-3): ")
    
    if choice == '1':
        test_hardware_connection()
    elif choice == '2':
        run_realtime_visualization()
    elif choice == '3':
        print("\nTo integrate with your protocol:")
        print("1. Import OpenBCIDevice from this module")
        print("2. Create configuration with your parameters")
        print("3. Use device.get_data() in your analysis loop")
        print("\nExample:")
        print(">>> from openbci_hardware_interface import OpenBCIDevice, OpenBCIConfig, BoardType")
        print(">>> config = OpenBCIConfig(board_type=BoardType.CYTON)")
        print(">>> with OpenBCIDevice(config).session() as device:")
        print(">>>     device.start_streaming()")
        print(">>>     # Your analysis code here")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
