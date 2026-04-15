"""
PromptMOO: Multi-Objective Prompt Optimization Framework

This package provides a modular framework for optimizing prompts across multiple objectives
using algorithms like OPRO, GPO, and TextGrad.

Key Components:
- algorithm: Core optimization algorithms (OPRO, GPO, TextGrad)
- config: Centralized configuration system (promptmoo_config)
- data_structures: Immutable data classes (Task, Batch, PredictionResult, etc.)
- task_predictor: Generate predictions using task LLM
- loss_computer: Compute numeric and textual feedback
- gradient_computer: Generate textual gradients from feedback
- prompt_optimizer: Update prompts based on gradients
- observability: Structured logging and output management
"""

# Base algorithm class
# Concrete algorithm classes + their co-located components
from .algorithm import (
    GPO,
    OPRO,
    TextGrad,
)

# Configuration
from .config import PromptMOOConfig, PromptMOODefaults, promptmoo_config, temp_config

# Supporting modules
from .data_input import Dataset

# Data structures
from .data_structures import (
    AlgoMetricSeries,
    Batch,
    CombinedFeedback,
    DatasetSample,
    ExptMetricReport,
    NumericFeedback,
    OptimizerResult,
    PredictionResult,
    StepMetricResult,
    Task,
    TextGradient,
    TextualFeedback,
)
from .gradient_computer import (
    GradientComputer,
    StandardGradientComputer,
)
from .loss_computer import (
    LossComputer,
    TaskLevelLossComputer,
)
from .metrics import F1, LCE, Accuracy, Metric, Precision, Recall
from .observability import ObservabilityManager
from .prompt_algorithm import PromptAlgorithm, should_evaluate_at_step
from .prompt_optimizer import (
    LLMBasedOptimizer,
    PromptOptimizer,
)
from .prompt_template import PromptTemplate
from .prompt_trajectory import PromptTrajectory, TrajectoryElement
from .task_output_spec import TaskOutputSpec

# Pipeline base classes and shared implementations
from .task_predictor import StandardTaskPredictor, TaskPredictor

__all__ = [
    # Algorithms
    "PromptAlgorithm",
    "should_evaluate_at_step",
    "OPRO",
    "GPO",
    "TextGrad",
    # Configuration
    "promptmoo_config",
    "temp_config",
    "PromptMOOConfig",
    "PromptMOODefaults",
    # Data structures
    "Task",
    "DatasetSample",
    "Batch",
    "PredictionResult",
    "NumericFeedback",
    "TextualFeedback",
    "CombinedFeedback",
    "TextGradient",
    "OptimizerResult",
    # Pipeline components (base + shared)
    "TaskPredictor",
    "StandardTaskPredictor",
    "LossComputer",
    "TaskLevelLossComputer",
    "GradientComputer",
    "StandardGradientComputer",
    "PromptOptimizer",
    "LLMBasedOptimizer",
    # Supporting
    "Dataset",
    "Metric",
    "Accuracy",
    "F1",
    "Precision",
    "Recall",
    "LCE",
    "PromptTemplate",
    "ObservabilityManager",
    "PromptTrajectory",
    "TrajectoryElement",
    "TaskOutputSpec",
]

# Resolve deferred forward references now that all types are defined.
# OptimizerResult.new_prompt uses a string annotation "PromptTemplate"
# because prompt_template.py imports from data_structures.py
# (circular at module level). With all modules loaded, rebuild resolves it.
OptimizerResult.model_rebuild(_types_namespace={"PromptTemplate": PromptTemplate})
