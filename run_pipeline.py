"""
Complete Training and Evaluation Pipeline
==========================================
End-to-end script to preprocess data, train model, and evaluate.

Usage:
    python run_pipeline.py [--single-stage] [--skip-preprocessing]
    
Author: Research Team
Date: February 2026
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_preprocessing():
    """Run data preprocessing."""
    logger.info("\n" + "="*80)
    logger.info(" STEP 1: DATA PREPROCESSING")
    logger.info("="*80)
    
    from src.data_preparation.convert_annotations import main as preprocess_main
    preprocess_main()


def run_training(single_stage=False):
    """Run model training."""
    logger.info("\n" + "="*80)
    logger.info(" STEP 2: MODEL TRAINING")
    logger.info("="*80)
    
    from src.training.train_yolov8 import YOLOv8Trainer
    
    trainer = YOLOv8Trainer()
    trainer.run(use_staged_training=not single_stage)


def run_evaluation():
    """Run model evaluation."""
    logger.info("\n" + "="*80)
    logger.info(" STEP 3: MODEL EVALUATION")
    logger.info("="*80)
    
    from src.evaluation.evaluate_model import ModelEvaluator
    
    evaluator = ModelEvaluator()
    evaluator.generate_report()


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Run complete Smart Parking training pipeline'
    )
    parser.add_argument(
        '--single-stage',
        action='store_true',
        help='Use single-stage training instead of 2-stage'
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip data preprocessing (use existing processed data)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training (use existing model)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info(" SMART PARKING SYSTEM - COMPLETE PIPELINE")
    logger.info("="*80)
    logger.info(f"Single-stage training: {args.single_stage}")
    logger.info(f"Skip preprocessing: {args.skip_preprocessing}")
    logger.info(f"Skip training: {args.skip_training}")
    
    try:
        # Step 1: Preprocessing
        if not args.skip_preprocessing:
            run_preprocessing()
        else:
            logger.info("Skipping preprocessing (using existing data)")
        
        # Step 2: Training
        if not args.skip_training:
            run_training(single_stage=args.single_stage)
        else:
            logger.info("Skipping training (using existing model)")
        
        # Step 3: Evaluation
        run_evaluation()
        
        logger.info("\n" + "="*80)
        logger.info(" PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("1. Review results in 'figures/' directory")
        logger.info("2. Check predictions in 'predictions/' directory")
        logger.info("3. Start API server: python src/api/app.py")
        logger.info("4. View TensorBoard: tensorboard --logdir runs")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
