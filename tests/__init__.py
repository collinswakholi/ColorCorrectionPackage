"""
Test suite for ColorCorrectionPipeline
==================================

This directory contains comprehensive tests for the color correction pipeline:

Structure:
    - test_unit/: Unit tests for individual functions and classes
    - test_property/: Property-based tests using hypothesis
    - test_e2e/: End-to-end integration tests
    - conftest.py: Shared pytest fixtures and configuration
    - fixtures/: Test data and fixture files

Running tests:
    # All tests
    pytest
    
    # With coverage
    pytest --cov=ColorCorrectionPipeline --cov-report=term-missing
    
    # Specific test directory
    pytest tests/test_unit/
    
    # Verbose output
    pytest -v
"""
