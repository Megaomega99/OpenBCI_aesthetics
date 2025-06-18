"""
Sistema de Análisis Neurofisiológico para Respuestas Olfativas
Desarrollado para Belcorp - Capacitación OpenBCI Cyton con Python

Este script demuestra el potencial completo de análisis que se puede realizar
con datos EEG adquiridos durante protocolos de estimulación olfativa.
Simula el análisis de datos reales que se obtendrían del OpenBCI Cyton.

Versión mejorada con soporte para 8 y 16 canales, mejor manejo de errores
y arquitectura modular siguiendo principios SOLID.

Autores: Sistema de análisis basado en principios de neurociencia computacional
Referencias: Makeig et al. (1997), Delorme & Makeig (2004), Gramfort et al. (2013)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from contextlib import contextmanager
import warnings
from brainflow import BoardShim, BrainFlowInputParams, BoardIds

def setup_openbci_board(serial_port: str, board_type: str = "cyton"):
    """Configure real OpenBCI board connection"""
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    
    if board_type == "cyton":
        board_id = BoardIds.CYTON_BOARD
    elif board_type == "cyton_daisy":
        board_id = BoardIds.CYTON_DAISY_BOARD
    else:
        raise ValueError(f"Unknown board type: {board_type}")
    
    board = BoardShim(board_id, params)
    return board


# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output during demos
warnings.filterwarnings('ignore')

# Configuration for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ChannelConfiguration(Enum):
    """Enum for supported channel configurations"""
    EIGHT_CHANNELS = 8
    SIXTEEN_CHANNELS = 16


@dataclass
class EEGConfiguration:
    """Configuration dataclass for EEG analysis parameters"""
    sampling_rate: int = 250
    n_channels: ChannelConfiguration = ChannelConfiguration.SIXTEEN_CHANNELS
    
    # Frequency bands for neurophysiological analysis
    frequency_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'delta': (1, 4),    # Associated with deep states
        'theta': (4, 8),    # Related to emotions and memory
        'alpha': (8, 13),   # Relaxation, positive valence
        'beta': (13, 30),   # Alertness, cognitive processing
        'gamma': (30, 45)   # High-level conscious processing
    })
    
    def get_channel_count(self) -> int:
        """Get the actual number of channels"""
        return self.n_channels.value
    
    def get_electrode_positions(self) -> Dict[str, Tuple[float, float]]:
        """Get electrode positions based on channel configuration"""
        if self.n_channels == ChannelConfiguration.EIGHT_CHANNELS:
            # 8-channel configuration (basic OpenBCI Cyton)
            return {
                'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
                'C3': (-0.3, 0), 'C4': (0.3, 0),
                'P3': (-0.3, -0.5), 'P4': (0.3, -0.5),
                'O1': (-0.3, -0.9), 'O2': (0.3, -0.9)
            }
        else:
            # 16-channel configuration (Cyton + Daisy)
            return {
                'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
                'F7': (-0.7, 0.5), 'F3': (-0.3, 0.5), 'Fz': (0, 0.5), 
                'F4': (0.3, 0.5), 'F8': (0.7, 0.5),
                'T7': (-0.9, 0), 'C3': (-0.3, 0), 'Cz': (0, 0), 
                'C4': (0.3, 0), 'T8': (0.9, 0),
                'P7': (-0.7, -0.5), 'P3': (-0.3, -0.5), 'Pz': (0, -0.5), 
                'P4': (0.3, -0.5)
            }


@dataclass
class ExperimentalCondition:
    """Dataclass representing an experimental condition"""
    name: str
    display_name: str
    alpha_power: float
    beta_power: float
    frontal_asymmetry: float


class SignalGenerator(ABC):
    """Abstract base class for EEG signal generation"""
    
    @abstractmethod
    def generate_signal(self, n_samples: int, condition: ExperimentalCondition, 
                       config: EEGConfiguration) -> np.ndarray:
        """Generate EEG signal for a given condition"""
        pass


class RealisticEEGGenerator(SignalGenerator):
    """Generates realistic EEG signals with neurophysiological characteristics"""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed for reproducibility"""
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Random seed set to {seed} for reproducibility")
    
    def generate_signal(self, n_samples: int, condition: ExperimentalCondition,
                       config: EEGConfiguration) -> np.ndarray:
        """
        Generate realistic EEG signal with specified characteristics
        
        Args:
            n_samples: Number of samples to generate
            condition: Experimental condition parameters
            config: EEG configuration
            
        Returns:
            Generated EEG signal array (channels x samples)
        """
        n_channels = config.get_channel_count()
        trial_data = np.zeros((n_channels, n_samples))
        
        # Time vector for signal generation
        t = np.arange(n_samples) / config.sampling_rate
        
        for ch in range(n_channels):
            # Background neurophysiological noise (spontaneous cortical activity)
            background_noise = np.random.randn(n_samples) * 0.1
            
            # Alpha component modulated by experimental condition
            alpha_freq = np.random.uniform(8, 13)  # Individual variability in alpha frequency
            alpha_component = (condition.alpha_power * 
                             np.sin(2 * np.pi * alpha_freq * t) * 
                             np.random.randn() * 0.5)
            
            # Beta component modulated by experimental condition
            beta_freq = np.random.uniform(13, 30)
            beta_component = (condition.beta_power * 
                            np.sin(2 * np.pi * beta_freq * t) * 
                            np.random.randn() * 0.3)
            
            # Frontal hemispheric asymmetry (crucial for emotional valence)
            if ch < n_channels // 2:  # Left hemisphere
                hemispheric_modulation = condition.frontal_asymmetry
            else:  # Right hemisphere
                hemispheric_modulation = -condition.frontal_asymmetry
            
            # Simulate occasional ocular artifacts (eye blinks)
            if np.random.random() < 0.05:  # 5% probability of artifact
                blink_start = np.random.randint(0, max(1, n_samples - 500))
                blink_duration = min(500, n_samples - blink_start)
                blink_t = t[blink_start:blink_start + blink_duration]
                blink_center = t[blink_start + blink_duration // 2]
                blink_artifact = np.exp(-((blink_t - blink_center)**2) / 0.1)
                background_noise[blink_start:blink_start + blink_duration] += blink_artifact * 5
            
            # Combine all components with hemispheric modulation
            trial_data[ch, :] = (background_noise + alpha_component + beta_component) * (1 + hemispheric_modulation)
            
        return trial_data


class EEGAnalyzer:
    """Main analyzer class for EEG data processing and analysis"""
    
    def __init__(self, config: EEGConfiguration, signal_generator: Optional[SignalGenerator] = None):
        """
        Initialize analyzer with configuration and optional signal generator
        
        Args:
            config: EEG configuration parameters
            signal_generator: Optional custom signal generator
        """
        self.config = config
        self.signal_generator = signal_generator or RealisticEEGGenerator()
        logger.info(f"EEG Analyzer initialized with {config.get_channel_count()} channels")
    
    @contextmanager
    def _error_handler(self, operation: str):
        """Context manager for consistent error handling"""
        try:
            yield
        except Exception as e:
            logger.error(f"Error during {operation}: {str(e)}")
            raise
    
    def get_experimental_conditions(self) -> Dict[str, ExperimentalCondition]:
        """Get standard experimental conditions for olfactory study"""
        return {
            'pleasant': ExperimentalCondition(
                name='pleasant',
                display_name='Pleasant Essence (Rose)',
                alpha_power=1.8,  # Higher alpha activity (relaxation, positive valence)
                beta_power=0.6,   # Lower beta activity (less threat processing)
                frontal_asymmetry=0.3  # Positive frontal asymmetry (positive valence)
            ),
            'neutral': ExperimentalCondition(
                name='neutral',
                display_name='Neutral Essence (Water)',
                alpha_power=1.0,  # Baseline activity
                beta_power=1.0,   # Baseline activity
                frontal_asymmetry=0.0  # No preferential asymmetry
            ),
            'unpleasant': ExperimentalCondition(
                name='unpleasant',
                display_name='Unpleasant Essence (Ammonia)',
                alpha_power=0.4,  # Lower alpha activity (higher alertness)
                beta_power=2.2,   # Higher beta activity (threat processing)
                frontal_asymmetry=-0.4  # Negative frontal asymmetry (negative valence)
            )
        }
    
    def simulate_experiment(self, duration_seconds: float = 300, 
                          n_trials_per_condition: int = 20) -> Dict[str, Dict[str, Any]]:
        """
        Simulate complete EEG experiment with multiple conditions
        
        Args:
            duration_seconds: Duration of each trial in seconds
            n_trials_per_condition: Number of trials per experimental condition
            
        Returns:
            Dictionary with simulated EEG data organized by condition
        """
        with self._error_handler("experiment simulation"):
            n_samples = int(duration_seconds * self.config.sampling_rate)
            conditions = self.get_experimental_conditions()
            eeg_data = {}
            
            for condition_name, condition in conditions.items():
                logger.info(f"Generating {n_trials_per_condition} trials for {condition_name} condition")
                
                trials = []
                for trial in range(n_trials_per_condition):
                    trial_data = self.signal_generator.generate_signal(n_samples, condition, self.config)
                    trials.append(trial_data)
                
                eeg_data[condition_name] = {
                    'data': np.array(trials),  # Shape: (trials, channels, samples)
                    'condition': condition
                }
            
            return eeg_data
    
    def compute_spectral_features(self, eeg_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute spectral features for all experimental conditions
        
        Args:
            eeg_data: Dictionary with EEG data organized by condition
            
        Returns:
            Dictionary with spectral features by condition and frequency band
        """
        with self._error_handler("spectral analysis"):
            spectral_results = {}
            
            for condition_name, condition_data in eeg_data.items():
                trials = condition_data['data']
                n_trials, n_channels, n_samples = trials.shape
                
                # Initialize power matrices for each frequency band
                band_powers = {band: np.zeros((n_trials, n_channels)) 
                             for band in self.config.frequency_bands.keys()}
                
                for trial_idx in range(n_trials):
                    for ch_idx in range(n_channels):
                        # Extract signal for specific channel
                        signal_data = trials[trial_idx, ch_idx, :]
                        
                        # Compute power spectral density using Welch's method
                        # Calculate segment length and overlap
                        nperseg = min(self.config.sampling_rate * 2, len(signal_data) // 4)
                        noverlap = int(nperseg * 0.5)  # 50% overlap
                        
                        frequencies, psd = signal.welch(
                            signal_data, 
                            fs=self.config.sampling_rate, 
                            window='hann',
                            nperseg=nperseg,
                            noverlap=noverlap
                        )
                        
                        # Calculate power in each frequency band
                        for band_name, (low_freq, high_freq) in self.config.frequency_bands.items():
                            # Find frequency band indices
                            freq_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
                            
                            # Integrate power in band (area under PSD curve)
                            if np.any(freq_mask):
                                band_power = np.trapz(psd[freq_mask], frequencies[freq_mask])
                                band_powers[band_name][trial_idx, ch_idx] = band_power
                
                spectral_results[condition_name] = band_powers
            
            return spectral_results
    
    def perform_statistical_analysis(self, spectral_results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Perform statistical comparisons between experimental conditions
        
        Args:
            spectral_results: Dictionary with spectral features by condition
            
        Returns:
            Dictionary with statistical test results
        """
        with self._error_handler("statistical analysis"):
            statistical_results = {}
            conditions = list(spectral_results.keys())
            
            # Compare each pair of conditions
            for i, condition1 in enumerate(conditions):
                for j, condition2 in enumerate(conditions[i+1:], i+1):
                    comparison_name = f"{condition1}_vs_{condition2}"
                    statistical_results[comparison_name] = {}
                    
                    for band_name in self.config.frequency_bands.keys():
                        # Extract power data for each condition
                        data1 = spectral_results[condition1][band_name]
                        data2 = spectral_results[condition2][band_name]
                        
                        # Average across channels for global analysis
                        mean_power1 = np.mean(data1, axis=1)  # Average per trial
                        mean_power2 = np.mean(data2, axis=1)
                        
                        # Perform Student's t-test for mean differences
                        t_stat, p_value = stats.ttest_ind(mean_power1, mean_power2)
                        
                        # Calculate effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(mean_power1)-1)*np.var(mean_power1, ddof=1) + 
                                            (len(mean_power2)-1)*np.var(mean_power2, ddof=1)) / 
                                           (len(mean_power1)+len(mean_power2)-2))
                        
                        # Avoid division by zero
                        if pooled_std > 0:
                            cohens_d = (np.mean(mean_power1) - np.mean(mean_power2)) / pooled_std
                        else:
                            cohens_d = 0.0
                        
                        statistical_results[comparison_name][band_name] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'cohens_d': cohens_d,
                            'significant': p_value < 0.05
                        }
            
            return statistical_results
    
    def train_classifier(self, spectral_results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Train machine learning classifier for automatic emotional state detection
        
        Args:
            spectral_results: Dictionary with spectral features by condition
            
        Returns:
            Dictionary with trained classifier and performance metrics
        """
        with self._error_handler("classifier training"):
            # Prepare data for classification
            X = []  # Features (spectral power)
            y = []  # Labels (experimental condition)
            
            condition_mapping = {
                'pleasant': 1,
                'neutral': 0, 
                'unpleasant': -1
            }
            
            for condition_name, band_powers in spectral_results.items():
                if condition_name not in condition_mapping:
                    continue
                    
                # Extract features for each trial
                n_trials = band_powers['alpha'].shape[0]
                for trial_idx in range(n_trials):
                    features = []
                    for band_name in self.config.frequency_bands.keys():
                        # Spatial average of power per band
                        trial_features = np.mean(band_powers[band_name][trial_idx, :])
                        features.append(trial_features)
                    
                    X.append(features)
                    y.append(condition_mapping[condition_name])
            
            X = np.array(X)
            y = np.array(y)
            
            # Train linear discriminant classifier
            classifier = LinearDiscriminantAnalysis()
            
            # Cross-validation evaluation
            cv_scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
            
            # Train final model on all data
            classifier.fit(X, y)
            
            return {
                'classifier': classifier,
                'cv_accuracy_mean': np.mean(cv_scores),
                'cv_accuracy_std': np.std(cv_scores),
                'feature_importance': classifier.coef_[0] if hasattr(classifier, 'coef_') else None,
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
    
    def calculate_frontal_asymmetry(self, spectral_results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
        """
        Calculate frontal asymmetry indices for each condition
        
        Args:
            spectral_results: Dictionary with spectral features by condition
            
        Returns:
            Dictionary with asymmetry indices by condition
        """
        asymmetries = {}
        
        # Channel indices for frontal electrodes depend on configuration
        if self.config.n_channels == ChannelConfiguration.EIGHT_CHANNELS:
            # For 8 channels, use Fp1 (0) and Fp2 (1)
            left_idx, right_idx = 0, 1
        else:
            # For 16 channels, use F3 (3) and F4 (5)
            left_idx, right_idx = 3, 5
        
        for condition, band_powers in spectral_results.items():
            # Calculate asymmetry in alpha band (ln(right) - ln(left))
            left_alpha = band_powers['alpha'][:, left_idx]
            right_alpha = band_powers['alpha'][:, right_idx]
            
            # Avoid log of zero or negative values
            left_alpha = np.maximum(left_alpha, 1e-10)
            right_alpha = np.maximum(right_alpha, 1e-10)
            
            # Calculate logarithmic asymmetry index
            asymmetry = np.mean(np.log(right_alpha) - np.log(left_alpha))
            asymmetries[condition] = asymmetry
        
        return asymmetries


class VisualizationManager:
    """Manages all visualization tasks for EEG analysis"""
    
    def __init__(self, config: EEGConfiguration):
        """Initialize visualization manager with configuration"""
        self.config = config
        self.electrode_positions = config.get_electrode_positions()
    
    def create_comprehensive_figure(self, eeg_data: Dict[str, Dict[str, Any]], 
                                  spectral_results: Dict[str, Dict[str, np.ndarray]],
                                  statistical_results: Dict[str, Dict[str, Dict[str, float]]],
                                  ml_results: Dict[str, Any],
                                  asymmetries: Dict[str, float]) -> plt.Figure:
        """
        Create comprehensive visualization figure with all analysis results
        
        Args:
            eeg_data: Raw EEG data
            spectral_results: Spectral analysis results
            statistical_results: Statistical test results
            ml_results: Machine learning results
            asymmetries: Frontal asymmetry indices
            
        Returns:
            Matplotlib figure with comprehensive visualizations
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid of subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Representative EEG signals
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_representative_signals(eeg_data, ax1)
        
        # 2. Spectral power comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_spectral_comparison(spectral_results, ax2)
        
        # 3. Topographic map
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_topographic_map(spectral_results, ax3)
        
        # 4. Statistical results
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_statistical_results(statistical_results, ax4)
        
        # 5. Frontal asymmetry
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_frontal_asymmetry(asymmetries, ax5)
        
        # 6. Machine learning results
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_ml_results(ml_results, ax6)
        
        # 7. Alpha dynamics
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_alpha_dynamics(eeg_data, ax7)
        
        # 8. Connectivity analysis
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_connectivity_analysis(eeg_data, ax8)
        
        # 9. Summary statistics
        ax9 = fig.add_subplot(gs[2, :2])
        self._plot_summary_statistics(spectral_results, ax9)
        
        # 10. Configuration info
        ax10 = fig.add_subplot(gs[2, 2:])
        self._plot_configuration_info(ax10)
        
        # Main title
        plt.suptitle(f'Comprehensive Neurophysiological Analysis: Olfactory Responses\n'
                    f'OpenBCI Cyton ({self.config.get_channel_count()} channels) + Python for Belcorp Research', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def _plot_representative_signals(self, eeg_data: Dict[str, Dict[str, Any]], ax: plt.Axes) -> None:
        """Plot representative EEG signals for each condition"""
        colors = {'pleasant': 'green', 'neutral': 'gray', 'unpleasant': 'red'}
        
        for i, (condition, color) in enumerate(colors.items()):
            if condition in eeg_data:
                # Take first trial, central channel
                central_ch = min(4, self.config.get_channel_count() - 1)
                signal_data = eeg_data[condition]['data'][0, central_ch, :1000]  # First 4 seconds
                time_axis = np.arange(len(signal_data)) / self.config.sampling_rate
                
                ax.plot(time_axis, signal_data + i*0.5, color=color, 
                       label=eeg_data[condition]['condition'].display_name, linewidth=1.5)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title('Representative EEG Signals\nCentral Channel')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_spectral_comparison(self, spectral_results: Dict[str, Dict[str, np.ndarray]], ax: plt.Axes) -> None:
        """Compare spectral power between conditions"""
        bands = list(self.config.frequency_bands.keys())
        conditions = ['pleasant', 'neutral', 'unpleasant']
        colors = ['green', 'gray', 'red']
        
        x = np.arange(len(bands))
        width = 0.25
        
        for i, condition in enumerate(conditions):
            if condition in spectral_results:
                means = []
                stds = []
                
                for band in bands:
                    band_data = spectral_results[condition][band]
                    means.append(np.mean(band_data))
                    stds.append(np.std(band_data) / np.sqrt(band_data.shape[0]))  # SEM
                
                ax.bar(x + i*width, means, width, yerr=stds, 
                      label=condition.capitalize(), color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Frequency Bands')
        ax.set_ylabel('Spectral Power (μV²/Hz)')
        ax.set_title('Spectral Power Comparison\nBy Olfactory Condition')
        ax.set_xticks(x + width)
        ax.set_xticklabels([b.capitalize() for b in bands])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_topographic_map(self, spectral_results: Dict[str, Dict[str, np.ndarray]], ax: plt.Axes) -> None:
        """Create topographic map of alpha band differences"""
        # Calculate difference pleasant vs unpleasant in alpha band
        if 'pleasant' in spectral_results and 'unpleasant' in spectral_results:
            alpha_pleasant = np.mean(spectral_results['pleasant']['alpha'], axis=0)
            alpha_unpleasant = np.mean(spectral_results['unpleasant']['alpha'], axis=0)
            alpha_diff = alpha_pleasant - alpha_unpleasant
            
            # Create simplified topographic map
            electrode_names = list(self.electrode_positions.keys())
            
            for i, (electrode, (x, y)) in enumerate(self.electrode_positions.items()):
                if i < len(alpha_diff):
                    # Normalize differences for visualization
                    norm_diff = alpha_diff[i] / (np.max(np.abs(alpha_diff)) + 1e-10)
                    color = 'red' if norm_diff > 0 else 'blue'
                    size = abs(norm_diff) * 200 + 50
                    
                    circle = Circle((x, y), 0.08, color=color, alpha=abs(norm_diff))
                    ax.add_patch(circle)
                    ax.text(x, y-0.15, electrode, ha='center', va='center', fontsize=8)
        
        # Draw head outline
        head_circle = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        ax.add_patch(head_circle)
        
        # Add nose
        nose = plt.Polygon([(0, 1), (-0.1, 1.1), (0.1, 1.1)], 
                          closed=True, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(nose)
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.set_title('Topographic Map\nAlpha Band Differences\n(Pleasant - Unpleasant)')
        ax.axis('off')
    
    def _plot_statistical_results(self, statistical_results: Dict[str, Dict[str, Dict[str, float]]], ax: plt.Axes) -> None:
        """Visualize statistical test results"""
        comparison = 'pleasant_vs_unpleasant'
        if comparison in statistical_results:
            bands = list(self.config.frequency_bands.keys())
            p_values = []
            effect_sizes = []
            
            for band in bands:
                if band in statistical_results[comparison]:
                    p_values.append(statistical_results[comparison][band]['p_value'])
                    effect_sizes.append(abs(statistical_results[comparison][band]['cohens_d']))
                else:
                    p_values.append(1.0)
                    effect_sizes.append(0.0)
            
            # Create bar plot for effect sizes
            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            bars = ax.bar(bands, effect_sizes, color=colors, alpha=0.7)
            
            # Add significance lines
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Medium Effect')
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large Effect')
            
            ax.set_ylabel("Effect Size (Cohen's d)")
            ax.set_title('Statistical Significance\nPleasant vs Unpleasant')
            ax.legend()
            
            # Annotate p-values
            for i, (bar, p_val) in enumerate(zip(bars, p_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'p={p_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_frontal_asymmetry(self, asymmetries: Dict[str, float], ax: plt.Axes) -> None:
        """Plot frontal asymmetry indices"""
        conditions = ['pleasant', 'neutral', 'unpleasant']
        colors = ['green', 'gray', 'red']
        values = [asymmetries.get(c, 0) for c in conditions]
        
        bars = ax.bar(conditions, values, color=colors, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_ylabel('Frontal Asymmetry Index\n(log(Right) - log(Left))')
        ax.set_title('Hemispheric Frontal Asymmetry\nAlpha Band')
        ax.set_xticklabels([c.capitalize() for c in conditions])
        
        # Add interpretation note
        ax.text(0.02, 0.98, 'Positive values:\nPositive emotional valence', 
                transform=ax.transAxes, va='top', ha='left', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    def _plot_ml_results(self, ml_results: Dict[str, Any], ax: plt.Axes) -> None:
        """Visualize machine learning classification results"""
        if ml_results:
            accuracy = ml_results['cv_accuracy_mean']
            std_error = ml_results['cv_accuracy_std']
            
            # Plot accuracy bar
            ax.bar(['Classification\nAccuracy'], [accuracy], 
                   yerr=[std_error], color='purple', alpha=0.8, capsize=10)
            
            ax.set_ylim(0, 1)
            ax.set_ylabel('Classification Accuracy')
            ax.set_title(f'Machine Learning Performance\nAccuracy: {accuracy:.2f} ± {std_error:.2f}')
            
            # Reference line for chance level
            ax.axhline(y=0.33, color='red', linestyle='--', alpha=0.7, 
                      label='Chance level (33%)')
            ax.legend()
            
            # Add sample size info
            ax.text(0.02, 0.02, f'N samples: {ml_results.get("n_samples", "N/A")}\n'
                              f'N features: {ml_results.get("n_features", "N/A")}',
                    transform=ax.transAxes, va='bottom', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    def _plot_alpha_dynamics(self, eeg_data: Dict[str, Dict[str, Any]], ax: plt.Axes) -> None:
        """Analyze temporal dynamics of alpha band"""
        window_size = int(2 * self.config.sampling_rate)  # 2-second window
        overlap = int(0.5 * self.config.sampling_rate)    # 0.5-second overlap
        
        conditions = ['pleasant', 'neutral', 'unpleasant']
        colors = ['green', 'gray', 'red']
        
        for condition, color in zip(conditions, colors):
            if condition in eeg_data:
                # Take first trial, central channel
                central_ch = min(self.config.get_channel_count() // 2, 
                               self.config.get_channel_count() - 1)
                signal_data = eeg_data[condition]['data'][0, central_ch, :]
                
                # Calculate alpha power in sliding windows
                alpha_power_time = []
                time_points = []
                
                for start in range(0, len(signal_data) - window_size, overlap):
                    window_data = signal_data[start:start + window_size]
                    
                    # Calculate alpha band power for this window
                    # Use smaller segment for short windows
                    nperseg = min(len(window_data) // 4, self.config.sampling_rate)
                    if nperseg > 0:
                        freqs, psd = signal.welch(
                            window_data, 
                            fs=self.config.sampling_rate,
                            nperseg=nperseg,
                            noverlap=nperseg//2
                        )
                    else:
                        continue
                    alpha_mask = (freqs >= 8) & (freqs <= 13)
                    if np.any(alpha_mask):
                        alpha_power = np.trapz(psd[alpha_mask], freqs[alpha_mask])
                        alpha_power_time.append(alpha_power)
                        time_points.append((start + window_size/2) / self.config.sampling_rate)
                
                if alpha_power_time:
                    ax.plot(time_points, alpha_power_time, color=color, 
                           label=condition.capitalize(), linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Alpha Power (μV²/Hz)')
        ax.set_title('Temporal Dynamics\nAlpha Band (8-13 Hz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_connectivity_analysis(self, eeg_data: Dict[str, Dict[str, Any]], ax: plt.Axes) -> None:
        """Analyze functional connectivity between brain regions"""
        conditions = ['pleasant', 'neutral', 'unpleasant']
        connectivity_values = []
        
        for condition in conditions:
            if condition in eeg_data:
                trial_data = eeg_data[condition]['data'][0]  # First trial
                
                # Calculate average correlation between all channel pairs
                n_channels = trial_data.shape[0]
                correlations = []
                
                for i in range(n_channels):
                    for j in range(i + 1, n_channels):
                        corr_coeff = np.corrcoef(trial_data[i], trial_data[j])[0, 1]
                        correlations.append(abs(corr_coeff))  # Absolute value for connectivity
                
                connectivity_values.append(np.mean(correlations) if correlations else 0)
        
        colors = ['green', 'gray', 'red']
        bars = ax.bar([c.capitalize() for c in conditions], connectivity_values, 
                      color=colors, alpha=0.8)
        
        ax.set_ylabel('Functional Connectivity\n(Mean Absolute Correlation)')
        ax.set_title('Global Brain Connectivity')
        ax.set_ylim(0, 1)
        
        # Annotate values
        for bar, value in zip(bars, connectivity_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_summary_statistics(self, spectral_results: Dict[str, Dict[str, np.ndarray]], ax: plt.Axes) -> None:
        """Plot summary statistics table"""
        # Create summary data
        summary_data = []
        
        for condition in ['pleasant', 'neutral', 'unpleasant']:
            if condition in spectral_results:
                row_data = {'Condition': condition.capitalize()}
                
                for band in self.config.frequency_bands.keys():
                    mean_power = np.mean(spectral_results[condition][band])
                    row_data[f'{band.capitalize()} (μV²/Hz)'] = f'{mean_power:.2f}'
                
                summary_data.append(row_data)
        
        # Create table
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Clear axis
            ax.axis('tight')
            ax.axis('off')
            
            # Create table
            table = ax.table(cellText=df.values,
                           colLabels=df.columns,
                           cellLoc='center',
                           loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for (i, j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2')
                cell.set_edgecolor('white')
        
        ax.set_title('Summary Statistics: Mean Spectral Power by Condition', pad=20)
    
    def _plot_configuration_info(self, ax: plt.Axes) -> None:
        """Display configuration information"""
        config_text = f"""
System Configuration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Channels: {self.config.get_channel_count()}
• Sampling Rate: {self.config.sampling_rate} Hz
• Channel Configuration: {self.config.n_channels.name}

Frequency Bands:
• Delta: {self.config.frequency_bands['delta']} Hz
• Theta: {self.config.frequency_bands['theta']} Hz
• Alpha: {self.config.frequency_bands['alpha']} Hz
• Beta: {self.config.frequency_bands['beta']} Hz
• Gamma: {self.config.frequency_bands['gamma']} Hz

Analysis Methods:
• Spectral Analysis: Welch's Method
• Statistics: Student's t-test with Cohen's d
• Machine Learning: Linear Discriminant Analysis
• Validation: 5-fold Cross-validation
        """
        
        ax.text(0.05, 0.95, config_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')


class ReportGenerator:
    """Generates comprehensive scientific reports"""
    
    def __init__(self, config: EEGConfiguration):
        """Initialize report generator with configuration"""
        self.config = config
    
    def generate_report(self, eeg_data: Dict[str, Dict[str, Any]], 
                       spectral_results: Dict[str, Dict[str, np.ndarray]],
                       statistical_results: Dict[str, Dict[str, Dict[str, float]]],
                       ml_results: Dict[str, Any],
                       asymmetries: Dict[str, float]) -> str:
        """
        Generate comprehensive scientific report
        
        Args:
            All analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        
        report.append("="*80)
        report.append("SCIENTIFIC REPORT: NEUROPHYSIOLOGICAL ANALYSIS OF OLFACTORY RESPONSES")
        report.append(f"System: OpenBCI Cyton ({self.config.get_channel_count()} channels) + Python | Client: Belcorp Research")
        report.append("="*80)
        
        report.append("\n1. EXECUTIVE SUMMARY")
        report.append("-" * 50)
        
        # Extract key findings
        pleasant_vs_unpleasant = statistical_results.get('pleasant_vs_unpleasant', {})
        significant_bands = [band for band, results in pleasant_vs_unpleasant.items() 
                           if results.get('significant', False)]
        
        total_trials = sum(data['data'].shape[0] for data in eeg_data.values())
        
        report.append(f"• Analyzed {total_trials} EEG trials across {len(eeg_data)} conditions")
        report.append(f"• Channel configuration: {self.config.n_channels.name} ({self.config.get_channel_count()} channels)")
        report.append(f"• Significant differences detected in {len(significant_bands)} frequency bands")
        report.append(f"• Automatic classification accuracy: {ml_results['cv_accuracy_mean']:.1%}")
        
        if ml_results['cv_accuracy_mean'] > 0.7:
            report.append("• CONCLUSION: The system can reliably distinguish between responses")
            report.append("  to pleasant vs unpleasant olfactory stimuli")
        
        report.append("\n2. MAIN NEUROPHYSIOLOGICAL FINDINGS")
        report.append("-" * 50)
        
        # Analyze alpha band
        if 'alpha' in pleasant_vs_unpleasant and pleasant_vs_unpleasant['alpha']['significant']:
            alpha_effect = pleasant_vs_unpleasant['alpha']['cohens_d']
            if alpha_effect > 0:
                report.append("• ALPHA BAND (8-13 Hz): Higher activity in response to pleasant essences")
                report.append("  → Indicative of relaxation state and positive emotional valence")
            else:
                report.append("• ALPHA BAND (8-13 Hz): Higher activity in response to unpleasant essences")
                report.append("  → Possible compensatory mechanism for emotional regulation")
        
        # Analyze beta band
        if 'beta' in pleasant_vs_unpleasant and pleasant_vs_unpleasant['beta']['significant']:
            beta_effect = pleasant_vs_unpleasant['beta']['cohens_d']
            if beta_effect < 0:  # More beta in unpleasant
                report.append("• BETA BAND (13-30 Hz): Higher activity in response to unpleasant essences")
                report.append("  → Indicative of increased cognitive processing and alert state")
        
        # Frontal asymmetry analysis
        report.append("\n3. FRONTAL ASYMMETRY ANALYSIS")
        report.append("-" * 50)
        
        for condition, asymmetry in asymmetries.items():
            valence = "positive" if asymmetry > 0 else "negative"
            report.append(f"• {condition.upper()}: Asymmetry index = {asymmetry:.3f} ({valence} valence)")
        
        report.append("\n4. IMPLICATIONS FOR BELCORP")
        report.append("-" * 50)
        report.append("• TECHNICAL FEASIBILITY: OpenBCI + Python system is suitable for research")
        report.append("• POTENTIAL APPLICATIONS:")
        report.append("  - Objective evaluation of fragrances in product development")
        report.append("  - Optimization of sensory experiences at points of sale")
        report.append("  - Research on olfactory preferences across demographic segments")
        report.append("  - Neuroscientific validation of wellness claims in products")
        
        report.append("\n5. METHODOLOGICAL RECOMMENDATIONS")
        report.append("-" * 50)
        report.append("• Implement manual trigger protocol for temporal synchronization")
        report.append("• Use at least 20 trials per condition for statistical robustness")
        report.append("• Consider confounding variables: age, gender, prior olfactory experience")
        report.append("• Establish controlled environment: temperature, humidity, lighting")
        
        report.append("\n6. TECHNICAL SPECIFICATIONS")
        report.append("-" * 50)
        report.append(f"• Sampling rate: {self.config.sampling_rate} Hz")
        report.append(f"• Number of channels: {self.config.get_channel_count()}")
        report.append(f"• Analysis methods: Welch's PSD, t-tests, LDA classification")
        report.append(f"• Cross-validation folds: 5")
        
        return "\n".join(report)


def main():
    """
    Main function demonstrating the complete analysis pipeline
    
    This function showcases the full workflow from raw EEG data
    to actionable neuroscientific insights for Belcorp.
    """
    print("="*80)
    print("NEUROPHYSIOLOGICAL ANALYSIS SYSTEM INITIALIZATION")
    print("OpenBCI Cyton + Python for Belcorp Research")
    print("="*80)
    
    # Allow user to choose channel configuration
    print("\nSelect channel configuration:")
    print("1. 8 channels (Basic Cyton)")
    print("2. 16 channels (Cyton + Daisy)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ")
            if choice == '1':
                channel_config = ChannelConfiguration.EIGHT_CHANNELS
                break
            elif choice == '2':
                channel_config = ChannelConfiguration.SIXTEEN_CHANNELS
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    # Initialize configuration
    config = EEGConfiguration(
        sampling_rate=250,
        n_channels=channel_config
    )
    
    print(f"\n✓ Configuration set: {config.get_channel_count()} channels at {config.sampling_rate} Hz")
    
    # Initialize analyzer
    analyzer = EEGAnalyzer(config)
    
    # 1. Simulate experimental data
    print("\n1. Generating experimental EEG data...")
    eeg_data = analyzer.simulate_experiment(duration_seconds=10, n_trials_per_condition=20)
    print(f"   ✓ Data generated: {sum(data['data'].shape[0] for data in eeg_data.values())} total trials")
    
    # 2. Spectral analysis
    print("\n2. Performing spectral analysis...")
    spectral_results = analyzer.compute_spectral_features(eeg_data)
    print("   ✓ Spectral analysis completed for all frequency bands")
    
    # 3. Statistical comparisons
    print("\n3. Running statistical analysis...")
    statistical_results = analyzer.perform_statistical_analysis(spectral_results)
    
    # Report significant findings
    significant_findings = 0
    for comparison, bands in statistical_results.items():
        for band, results in bands.items():
            if results['significant']:
                significant_findings += 1
    
    print(f"   ✓ Statistical analysis completed: {significant_findings} significant differences detected")
    
    # 4. Machine learning classification
    print("\n4. Training machine learning classifier...")
    ml_results = analyzer.train_classifier(spectral_results)
    print(f"   ✓ Classifier trained: {ml_results['cv_accuracy_mean']:.1%} accuracy")
    
    # 5. Calculate frontal asymmetry
    print("\n5. Calculating frontal asymmetry indices...")
    asymmetries = analyzer.calculate_frontal_asymmetry(spectral_results)
    print("   ✓ Asymmetry analysis completed")
    
    # 6. Generate visualizations
    print("\n6. Creating scientific visualizations...")
    viz_manager = VisualizationManager(config)
    figure = viz_manager.create_comprehensive_figure(
        eeg_data, spectral_results, statistical_results, ml_results, asymmetries
    )
    print("   ✓ Visualizations generated")
    
    # 7. Generate scientific report
    print("\n7. Compiling scientific report...")
    report_gen = ReportGenerator(config)
    scientific_report = report_gen.generate_report(
        eeg_data, spectral_results, statistical_results, ml_results, asymmetries
    )
    
    # Display report
    print("\n" + "="*80)
    print("GENERATED SCIENTIFIC REPORT:")
    print("="*80)
    print(scientific_report)
    
    # Show visualizations
    plt.show()
    
    # Return all results for further use
    return {
        'config': config,
        'eeg_data': eeg_data,
        'spectral_results': spectral_results,
        'statistical_results': statistical_results,
        'ml_results': ml_results,
        'asymmetries': asymmetries,
        'figure': figure,
        'report': scientific_report
    }


if __name__ == "__main__":
    # Run the demonstration
    results = main()