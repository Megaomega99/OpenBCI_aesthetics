"""
Integrated OpenBCI Protocol for Belcorp
=======================================

This script integrates real OpenBCI hardware with the neurophysiological
analysis protocol for olfactory stimulation experiments.

Author: Integration module for Belcorp training
Date: June 2025
Version: 1.0.0
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import our modules
from openbci_hardware_interface import (
    OpenBCIDevice, OpenBCIConfig, BoardType, 
    DeviceDetector, RealTimeVisualizer
)
from protocolo import (
    EEGConfiguration, ChannelConfiguration, EEGAnalyzer,
    VisualizationManager, ReportGenerator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedExperimentSystem:
    """
    Integrated system combining real OpenBCI hardware with 
    advanced neurophysiological analysis for Belcorp
    """
    
    def __init__(self, board_type: BoardType = BoardType.CYTON):
        """
        Initialize integrated system
        
        Args:
            board_type: Type of OpenBCI board to use
        """
        self.board_type = board_type
        
        # Hardware configuration
        self.hardware_config = OpenBCIConfig(
            board_type=board_type,
            gain=24,  # Standard gain for EEG
            timeout=30.0  # Longer timeout for initial connection
        )
        
        # Analysis configuration
        n_channels = ChannelConfiguration.EIGHT_CHANNELS if board_type == BoardType.CYTON else ChannelConfiguration.SIXTEEN_CHANNELS
        self.analysis_config = EEGConfiguration(
            sampling_rate=250,  # OpenBCI standard
            n_channels=n_channels
        )
        
        # Initialize components
        self.device: Optional[OpenBCIDevice] = None
        self.analyzer = EEGAnalyzer(self.analysis_config)
        self.viz_manager = VisualizationManager(self.analysis_config)
        self.report_gen = ReportGenerator(self.analysis_config)
        
        # Data storage
        self.session_data: Dict[str, List[np.ndarray]] = {
            'pleasant': [],
            'neutral': [],
            'unpleasant': []
        }
        
        # Trigger codes for synchronization
        self.trigger_codes = {
            'baseline_start': 1.0,
            'baseline_end': 2.0,
            'pleasant_start': 10.0,
            'pleasant_end': 11.0,
            'neutral_start': 20.0,
            'neutral_end': 21.0,
            'unpleasant_start': 30.0,
            'unpleasant_end': 31.0,
            'trial_start': 100.0,
            'trial_end': 101.0
        }
        
        logger.info(f"Integrated system initialized for {board_type.name}")
    
    def setup_hardware(self) -> bool:
        """
        Setup and test hardware connection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Detect devices
            logger.info("Detecting OpenBCI devices...")
            ports = DeviceDetector.find_openbci_ports()
            
            if not ports and self.board_type != BoardType.SYNTHETIC:
                logger.error("No OpenBCI devices found")
                return False
            
            # Create device
            self.device = OpenBCIDevice(self.hardware_config)
            
            # Connect
            logger.info("Connecting to device...")
            self.device.connect()
            
            # Test data acquisition
            logger.info("Testing data acquisition...")
            self.device.start_streaming()
            time.sleep(2)  # Collect test data
            
            test_data = self.device.get_data()
            if test_data is None or test_data.shape[1] == 0:
                logger.error("No data received from device")
                return False
            
            self.device.stop_streaming()
            logger.info(f"Hardware test successful - received {test_data.shape[1]} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Hardware setup failed: {e}")
            return False
    
    def calibrate_system(self, duration: float = 60.0) -> Optional[Dict[str, float]]:
        """
        Perform system calibration with baseline recording
        
        Args:
            duration: Calibration duration in seconds
            
        Returns:
            Calibration metrics or None if failed
        """
        if not self.device:
            logger.error("Device not initialized")
            return None
        
        try:
            logger.info(f"Starting {duration}s calibration...")
            
            # Start streaming
            self.device.start_streaming()
            
            # Insert calibration start marker
            self.device.insert_marker(self.trigger_codes['baseline_start'])
            
            # Collect baseline data
            start_time = time.time()
            baseline_data = []
            
            while (time.time() - start_time) < duration:
                data = self.device.get_data()
                if data is not None and data.shape[1] > 0:
                    eeg_data = data[self.device.eeg_channels, :]
                    baseline_data.append(eeg_data)
                
                time.sleep(0.1)  # 10 Hz polling
            
            # Insert calibration end marker
            self.device.insert_marker(self.trigger_codes['baseline_end'])
            
            # Stop streaming
            self.device.stop_streaming()
            
            # Concatenate baseline data
            if baseline_data:
                baseline_array = np.hstack(baseline_data)
                
                # Compute baseline metrics
                metrics = {
                    'mean_amplitude': np.mean(np.abs(baseline_array)),
                    'std_amplitude': np.std(baseline_array),
                    'max_amplitude': np.max(np.abs(baseline_array)),
                    'n_samples': baseline_array.shape[1]
                }
                
                # Check signal quality
                if metrics['mean_amplitude'] < 1.0:
                    logger.warning("Very low signal amplitude - check electrode connections")
                elif metrics['mean_amplitude'] > 100.0:
                    logger.warning("High signal amplitude - possible noise or poor contact")
                else:
                    logger.info("Signal quality appears good")
                
                logger.info(f"Calibration complete - {metrics['n_samples']} samples collected")
                return metrics
                
            else:
                logger.error("No baseline data collected")
                return None
                
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return None
    
    def run_single_trial(self, condition: str, trial_number: int,
                        stimulus_duration: float = 10.0,
                        response_duration: float = 30.0) -> Optional[np.ndarray]:
        """
        Run a single experimental trial
        
        Args:
            condition: 'pleasant', 'neutral', or 'unpleasant'
            trial_number: Trial index
            stimulus_duration: Stimulus presentation time
            response_duration: Post-stimulus recording time
            
        Returns:
            Collected EEG data or None if failed
        """
        if not self.device:
            return None
        
        logger.info(f"Starting trial {trial_number}: {condition}")
        
        try:
            # Start streaming
            self.device.start_streaming()
            
            # Pre-stimulus baseline (5 seconds)
            logger.info("Recording pre-stimulus baseline...")
            self.device.insert_marker(self.trigger_codes['trial_start'])
            time.sleep(5.0)
            
            # Present stimulus
            logger.info(f"Presenting {condition} stimulus...")
            self.device.insert_marker(self.trigger_codes[f'{condition}_start'])
            
            # TODO: Here you would trigger actual olfactory device
            # For now, just wait
            time.sleep(stimulus_duration)
            
            self.device.insert_marker(self.trigger_codes[f'{condition}_end'])
            
            # Record response
            logger.info("Recording post-stimulus response...")
            time.sleep(response_duration)
            
            # End trial
            self.device.insert_marker(self.trigger_codes['trial_end'])
            
            # Stop streaming and get all data
            self.device.stop_streaming()
            trial_data = self.device.get_data()
            
            if trial_data is not None and trial_data.shape[1] > 0:
                # Extract EEG channels
                eeg_data = trial_data[self.device.eeg_channels, :]
                
                # Store trial data
                self.session_data[condition].append(eeg_data)
                
                logger.info(f"Trial complete - {eeg_data.shape[1]} samples collected")
                return eeg_data
            else:
                logger.error("No data collected for trial")
                return None
                
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            self.device.stop_streaming()
            return None
    
    def run_experiment_protocol(self, n_trials_per_condition: int = 5) -> bool:
        """
        Run complete experiment protocol
        
        Args:
            n_trials_per_condition: Number of trials for each condition
            
        Returns:
            True if successful, False otherwise
        """
        # Check hardware
        if not self.device:
            logger.error("Device not initialized")
            return False
        
        # Create trial order
        conditions = ['pleasant', 'neutral', 'unpleasant'] * n_trials_per_condition
        np.random.shuffle(conditions)  # Randomize order
        
        logger.info(f"Starting experiment with {len(conditions)} total trials")
        logger.info(f"Trial order: {conditions}")
        
        # Run all trials
        for i, condition in enumerate(conditions):
            trial_success = self.run_single_trial(
                condition=condition,
                trial_number=i + 1,
                stimulus_duration=10.0,
                response_duration=30.0
            )
            
            if not trial_success:
                logger.error(f"Trial {i + 1} failed")
                # Continue with next trial
            
            # Inter-trial interval
            if i < len(conditions) - 1:
                logger.info("Inter-trial interval (30s)...")
                time.sleep(30.0)
        
        logger.info("Experiment protocol complete")
        return True
    
    def analyze_session_data(self) -> Dict:
        """
        Analyze all collected session data
        
        Returns:
            Analysis results dictionary
        """
        # Convert session data to format expected by analyzer
        formatted_data = {}
        
        for condition, trials in self.session_data.items():
            if trials:
                # Stack trials
                trial_array = np.array(trials)
                
                # Create condition object
                condition_obj = self.analyzer.get_experimental_conditions()[condition]
                
                formatted_data[condition] = {
                    'data': trial_array,
                    'condition': condition_obj
                }
        
        if not formatted_data:
            logger.error("No data to analyze")
            return {}
        
        logger.info("Running neurophysiological analysis...")
        
        # Compute spectral features
        spectral_results = self.analyzer.compute_spectral_features(formatted_data)
        
        # Statistical analysis
        statistical_results = self.analyzer.perform_statistical_analysis(spectral_results)
        
        # Machine learning classification
        ml_results = self.analyzer.train_classifier(spectral_results)
        
        # Frontal asymmetry
        asymmetries = self.analyzer.calculate_frontal_asymmetry(spectral_results)
        
        # Generate visualizations
        figure = self.viz_manager.create_comprehensive_figure(
            formatted_data, spectral_results, statistical_results, 
            ml_results, asymmetries
        )
        
        # Generate report
        report = self.report_gen.generate_report(
            formatted_data, spectral_results, statistical_results,
            ml_results, asymmetries
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save figure
        figure.savefig(f'results_{timestamp}.png', dpi=300, bbox_inches='tight')
        
        # Save report
        with open(f'report_{timestamp}.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"Analysis complete - results saved with timestamp {timestamp}")
        
        return {
            'spectral': spectral_results,
            'statistical': statistical_results,
            'ml': ml_results,
            'asymmetries': asymmetries,
            'figure': figure,
            'report': report
        }
    
    def run_realtime_monitoring(self) -> None:
        """Run real-time monitoring mode"""
        if not self.device:
            logger.error("Device not initialized")
            return
        
        try:
            # Start streaming
            self.device.start_streaming()
            
            # Create real-time visualizer
            visualizer = RealTimeVisualizer(self.device, window_duration=10.0)
            
            logger.info("Starting real-time monitoring...")
            logger.info("Press Ctrl+C to stop")
            
            # Run visualization (blocking)
            visualizer.start()
            
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.device.stop_streaming()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.device:
            self.device.disconnect()
            logger.info("System cleanup complete")


def quick_test_mode():
    """Quick test mode for hardware verification"""
    print("\n" + "="*80)
    print("QUICK HARDWARE TEST MODE")
    print("="*80 + "\n")
    
    system = IntegratedExperimentSystem(BoardType.CYTON)
    
    try:
        # Setup hardware
        if not system.setup_hardware():
            print("✗ Hardware setup failed")
            return
        
        print("✓ Hardware connected successfully")
        
        # Quick calibration
        metrics = system.calibrate_system(duration=10.0)
        if metrics:
            print(f"✓ Calibration complete")
            print(f"  Mean amplitude: {metrics['mean_amplitude']:.2f} μV")
            print(f"  Signal quality: {'Good' if 1 < metrics['mean_amplitude'] < 100 else 'Check connections'}")
        
        # Optional: Run single test trial
        response = input("\nRun a test trial? (y/n): ")
        if response.lower() == 'y':
            system.run_single_trial('neutral', 1, 
                                  stimulus_duration=5.0, 
                                  response_duration=10.0)
            print("✓ Test trial complete")
        
    finally:
        system.cleanup()


def full_experiment_mode():
    """Full experiment mode with complete protocol"""
    print("\n" + "="*80)
    print("FULL EXPERIMENT MODE")
    print("="*80 + "\n")
    
    # Get board type
    print("Select board configuration:")
    print("1. Cyton (8 channels)")
    print("2. Cyton + Daisy (16 channels)")
    
    choice = input("\nEnter choice (1-2): ")
    board_type = BoardType.CYTON if choice == '1' else BoardType.CYTON_DAISY
    
    system = IntegratedExperimentSystem(board_type)
    
    try:
        # Setup
        if not system.setup_hardware():
            print("✗ Hardware setup failed")
            return
        
        # Calibration
        print("\nPerforming system calibration...")
        metrics = system.calibrate_system(duration=30.0)
        if not metrics:
            print("✗ Calibration failed")
            return
        
        # Get number of trials
        n_trials = int(input("\nNumber of trials per condition (default 3): ") or "3")
        
        # Confirm start
        print(f"\nReady to start experiment with {n_trials * 3} total trials")
        input("Press Enter to begin...")
        
        # Run experiment
        success = system.run_experiment_protocol(n_trials_per_condition=n_trials)
        
        if success:
            print("\n✓ Experiment complete!")
            
            # Analyze data
            print("\nAnalyzing collected data...")
            results = system.analyze_session_data()
            
            if results:
                print("\n✓ Analysis complete!")
                print(f"  Classification accuracy: {results['ml']['cv_accuracy_mean']:.1%}")
                print(f"  Results saved to current directory")
        
    finally:
        system.cleanup()


def realtime_monitoring_mode():
    """Real-time monitoring mode"""
    print("\n" + "="*80)
    print("REAL-TIME MONITORING MODE")
    print("="*80 + "\n")
    
    system = IntegratedExperimentSystem(BoardType.CYTON)
    
    try:
        # Setup
        if not system.setup_hardware():
            print("✗ Hardware setup failed")
            return
        
        print("✓ Starting real-time visualization...")
        print("  - EEG signals")
        print("  - Band powers")
        print("  - Spectrogram")
        print("\nPress Ctrl+C to stop")
        
        # Run monitoring
        system.run_realtime_monitoring()
        
    finally:
        system.cleanup()


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("INTEGRATED OPENBCI SYSTEM FOR BELCORP")
    print("Neurophysiological Analysis of Olfactory Responses")
    print("="*80 + "\n")
    
    print("Select operation mode:")
    print("1. Quick hardware test")
    print("2. Full experiment protocol")
    print("3. Real-time monitoring")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        quick_test_mode()
    elif choice == '2':
        full_experiment_mode()
    elif choice == '3':
        realtime_monitoring_mode()
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
