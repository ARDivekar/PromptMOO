"""
PromptMOO: Multi-Objective Prompt Optimization Framework

This package provides a modular framework for optimizing prompts across multiple objectives
using algorithms like OPRO with textual gradients.

Key Components:
- algorithm: Core optimization algorithms (OPRO)
- llm_workers: Concurry-based LLM worker pools with rate limiting
- data_structures: Immutable data classes (Task, Batch, PredictionResult, etc.)
- task_predictor: Generate predictions using task LLM
- loss_computer: Compute numeric and textual feedback
- gradient_computer: Generate textual gradients from feedback
- prompt_optimizer: Update prompts based on gradients
- observability: Structured logging and output management
"""

# Core algorithm classes
from .algorithm import GPO, OPRO, PromptAlgorithm, TextGrad

# Data structures
from .data_structures import (
    Batch,
    CombinedFeedback,
    DatasetSample,
    NumericFeedback,
    PredictionResult,
    Task,
    TextGradient,
    TextualFeedback,
    AlgoMetricSeries,
    ExptMetricReport,
    StepMetricResult,
)

# LLM infrastructure
from .llm_workers import LLM

# Pipeline components
from .task_predictor import StandardTaskPredictor, TaskPredictor
from .loss_computer import (
    GPOLossComputer,
    LossComputer,
    OPROLossComputer,
    TaskLevelLossComputer,
    TextGradLossComputer,
)
from .gradient_computer import (
    GradientComputer,
    GPOGradientComputer,
    OPROGradientComputer,
    StandardGradientComputer,
    TextGradGradientComputer,
)
from .prompt_optimizer import (
    GPOOptimizer,
    LLMBasedOptimizer,
    OPROOptimizer,
    PromptOptimizer,
    TextGradOptimizer,
)

# Supporting modules
from .data_input import Dataset
from .prompt_template_utils import PromptTemplate
from .observability import ObservabilityManager
from .prompt_trajectory import PromptTrajectory, TrajectoryElement

__all__ = [
    # Algorithms
    "PromptAlgorithm",
    "OPRO",
    "GPO",
    "TextGrad",
    # Data structures
    "Task",
    "DatasetSample",
    "Batch",
    "PredictionResult",
    "NumericFeedback",
    "TextualFeedback",
    "CombinedFeedback",
    "TextGradient",
    # LLM infrastructure
    "LLM",
    # Pipeline components
    "TaskPredictor",
    "StandardTaskPredictor",
    "LossComputer",
    "TaskLevelLossComputer",
    "OPROLossComputer",
    "GPOLossComputer",
    "TextGradLossComputer",
    "GradientComputer",
    "StandardGradientComputer",
    "OPROGradientComputer",
    "GPOGradientComputer",
    "TextGradGradientComputer",
    "PromptOptimizer",
    "LLMBasedOptimizer",
    "OPROOptimizer",
    "GPOOptimizer",
    "TextGradOptimizer",
    # Supporting
    "Dataset",
    "PromptTemplate",
    "ObservabilityManager",
    "PromptTrajectory",
    "TrajectoryElement",
]
