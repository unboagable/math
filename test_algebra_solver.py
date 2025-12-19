"""
Pytest test suite for algebra_solver module.
"""

import pytest
from sympy import symbols
from algebra_solver import (
    AlgebraSolver,
    solve_equation,
    is_constant_expression,
    get_variable_or_no_solution,
    derivative,
    integral
)


class TestAlgebraSolver:
    """Test cases for AlgebraSolver class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = AlgebraSolver()
    
    def test_parse_equation(self):
        """Test equation parsing."""
        left, right = self.solver.parse_equation("2*x + 3 = 7")
        assert left == 2*symbols('x') + 3
        assert right == 7
    
    def test_parse_equation_no_equals(self):
        """Test that parsing equation without '=' raises error."""
        with pytest.raises(ValueError):
            self.solver.parse_equation("2*x + 3")
    
    def test_is_constant_expression_string(self):
        """Test constant detection with string input."""
        assert self.solver.is_constant_expression("5") == True
        assert self.solver.is_constant_expression("3 + 2") == True
        assert self.solver.is_constant_expression("10 - 5") == True
        assert self.solver.is_constant_expression("2*x") == False
        assert self.solver.is_constant_expression("x + 5") == False
    
    def test_is_constant_expression_sympy(self):
        """Test constant detection with sympy expression."""
        from sympy import sympify
        assert self.solver.is_constant_expression(sympify("5")) == True
        assert self.solver.is_constant_expression(sympify("x")) == False
    
    def test_get_variables(self):
        """Test variable extraction."""
        vars1 = self.solver.get_variables("x + 5")
        assert vars1 == ['x']
        
        vars2 = self.solver.get_variables("x*y + 2")
        assert 'x' in vars2 and 'y' in vars2
        
        vars3 = self.solver.get_variables("5")
        assert vars3 == []
    
    def test_solve_linear_equation(self):
        """Test solving simple linear equations."""
        result = self.solver.solve_equation("2*x + 3 = 7")
        assert result['variable'] == 'x'
        assert abs(result['solution'] - 2.0) < 1e-10
        assert result['is_constant'] == False
    
    def test_solve_equation_x_equals_number(self):
        """Test solving x = number."""
        result = self.solver.solve_equation("x = 5")
        assert result['variable'] == 'x'
        assert result['solution'] == 5.0
        assert result['is_constant'] == False
    
    def test_solve_equation_with_subtraction(self):
        """Test solving equations with subtraction."""
        result = self.solver.solve_equation("x - 3 = 7")
        assert result['variable'] == 'x'
        assert result['solution'] == 10.0
    
    def test_solve_equation_with_multiplication(self):
        """Test solving equations with multiplication."""
        result = self.solver.solve_equation("3*x = 15")
        assert result['variable'] == 'x'
        assert result['solution'] == 5.0
    
    def test_solve_equation_with_division(self):
        """Test solving equations with division."""
        result = self.solver.solve_equation("x / 2 = 5")
        assert result['variable'] == 'x'
        assert result['solution'] == 10.0
    
    def test_solve_equation_no_solution(self):
        """Test equation with no solution."""
        result = self.solver.solve_equation("0 = 5")
        assert result['variable'] is None
        assert result['solution'] == 'no solution'
        assert result['is_constant'] == True
    
    def test_solve_equation_infinite_solutions(self):
        """Test equation with infinite solutions."""
        result = self.solver.solve_equation("5 = 5")
        assert result['variable'] is None
        assert result['solution'] == 'infinite solutions'
        assert result['is_constant'] == True
    
    def test_solve_equation_constant_expression(self):
        """Test constant expression detection."""
        result = self.solver.solve_equation("3 + 2 = 5")
        assert result['is_constant'] == True
    
    def test_solve_quadratic_equation(self):
        """Test solving quadratic equations."""
        result = self.solver.solve_equation("x**2 - 5*x + 6 = 0")
        assert result['variable'] == 'x'
        assert isinstance(result['solution'], list)
        assert len(result['solution']) == 2
    
    def test_get_variable_or_no_solution_with_variable(self):
        """Test getting variable name when variable exists."""
        result = self.solver.get_variable_or_no_solution("x + 5 = 10")
        assert result == 'x'
    
    def test_get_variable_or_no_solution_no_variable(self):
        """Test getting 'no solution' when no variable exists."""
        result = self.solver.get_variable_or_no_solution("5 = 3")
        assert result == 'no solution'
    
    def test_get_variable_or_no_solution_constant_true(self):
        """Test getting 'no solution' for constant true equation."""
        result = self.solver.get_variable_or_no_solution("5 = 5")
        assert result == 'no solution'


class TestConvenienceFunctions:
    """Test cases for module-level convenience functions."""
    
    def test_solve_equation_function(self):
        """Test module-level solve_equation function."""
        result = solve_equation("x + 3 = 7")
        assert result['variable'] == 'x'
        assert result['solution'] == 4.0
    
    def test_is_constant_expression_function(self):
        """Test module-level is_constant_expression function."""
        assert is_constant_expression("5") == True
        assert is_constant_expression("x + 2") == False
    
    def test_get_variable_or_no_solution_function(self):
        """Test module-level get_variable_or_no_solution function."""
        assert get_variable_or_no_solution("x = 5") == 'x'
        assert get_variable_or_no_solution("5 = 5") == 'no solution'


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = AlgebraSolver()
    
    def test_equation_with_spaces(self):
        """Test equation with extra spaces."""
        result = self.solver.solve_equation("  x  +  3  =  7  ")
        assert result['variable'] == 'x'
        assert result['solution'] == 4.0
    
    def test_equation_with_different_variable_names(self):
        """Test equations with different variable names."""
        result = self.solver.solve_equation("y + 5 = 10")
        assert result['variable'] == 'y'
        assert result['solution'] == 5.0
        
        result = self.solver.solve_equation("a * 2 = 8")
        assert result['variable'] == 'a'
        assert result['solution'] == 4.0
    
    def test_equation_with_fractions(self):
        """Test equations with fractional coefficients."""
        result = self.solver.solve_equation("x/2 + 1 = 3")
        assert result['variable'] == 'x'
        assert abs(result['solution'] - 4.0) < 1e-10
    
    def test_equation_with_decimals(self):
        """Test equations with decimal numbers."""
        result = self.solver.solve_equation("2.5*x = 10")
        assert result['variable'] == 'x'
        assert abs(result['solution'] - 4.0) < 1e-10
    
    def test_complex_linear_equation(self):
        """Test more complex linear equations."""
        result = self.solver.solve_equation("2*x + 3*x - 5 = 10")
        assert result['variable'] == 'x'
        assert abs(result['solution'] - 3.0) < 1e-10


class TestDerivative:
    """Test cases for derivative functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = AlgebraSolver()
    
    def test_derivative_polynomial(self):
        """Test derivative of polynomial."""
        result = self.solver.derivative("x**2 + 3*x + 5")
        assert result['variable'] == 'x'
        assert '2*x + 3' in result['simplified'] or '3 + 2*x' in result['simplified']
        assert result['order'] == 1
    
    def test_derivative_simple(self):
        """Test derivative of simple expression."""
        result = self.solver.derivative("x**2")
        assert result['variable'] == 'x'
        assert '2*x' in result['simplified']
    
    def test_derivative_constant(self):
        """Test derivative of constant."""
        result = self.solver.derivative("5")
        assert result['derivative'] == '0'
        assert result['simplified'] == '0'
    
    def test_derivative_specified_variable(self):
        """Test derivative with specified variable."""
        result = self.solver.derivative("x**2 + y**2", variable="y")
        assert result['variable'] == 'y'
        assert '2*y' in result['simplified']
    
    def test_derivative_second_order(self):
        """Test second derivative."""
        result = self.solver.derivative("x**3 + 2*x**2", order=2)
        assert result['order'] == 2
        assert result['variable'] == 'x'
        assert '6*x' in result['simplified'] or '6*x + 4' in result['simplified']
    
    def test_derivative_trigonometric(self):
        """Test derivative of trigonometric function."""
        result = self.solver.derivative("sin(x)")
        assert result['variable'] == 'x'
        assert 'cos(x)' in result['simplified']
    
    def test_derivative_exponential(self):
        """Test derivative of exponential function."""
        result = self.solver.derivative("exp(x)")
        assert result['variable'] == 'x'
        assert 'exp(x)' in result['simplified']
    
    def test_derivative_product_rule(self):
        """Test derivative using product rule."""
        result = self.solver.derivative("x*sin(x)")
        assert result['variable'] == 'x'
        # Should contain x*cos(x) + sin(x) or similar
    
    def test_derivative_quotient(self):
        """Test derivative of quotient."""
        result = self.solver.derivative("x**2 / x")
        assert result['variable'] == 'x'
    
    def test_derivative_convenience_function(self):
        """Test module-level derivative function."""
        result = derivative("x**2 + 5*x")
        assert result['variable'] == 'x'
        assert '2*x' in result['simplified'] or '5' in result['simplified']


class TestIntegral:
    """Test cases for integral functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = AlgebraSolver()
    
    def test_indefinite_integral_polynomial(self):
        """Test indefinite integral of polynomial."""
        result = self.solver.integral("x**2")
        assert result['is_definite'] == False
        assert result['variable'] == 'x'
        assert 'x**3' in result['simplified'] or 'x**3/3' in result['simplified']
    
    def test_indefinite_integral_simple(self):
        """Test indefinite integral of simple expression."""
        result = self.solver.integral("2*x")
        assert result['is_definite'] == False
        assert result['variable'] == 'x'
        assert 'x**2' in result['simplified']
    
    def test_indefinite_integral_constant(self):
        """Test indefinite integral of constant."""
        result = self.solver.integral("5")
        assert result['is_definite'] == False
        assert '5*x' in result['simplified'] or 'x' in result['simplified']
    
    def test_definite_integral(self):
        """Test definite integral."""
        result = self.solver.integral("x**2", lower_bound=0, upper_bound=2)
        assert result['is_definite'] == True
        assert result['lower_bound'] == '0'
        assert result['upper_bound'] == '2'
        assert result['variable'] == 'x'
        # Result should be 8/3
    
    def test_definite_integral_linear(self):
        """Test definite integral of linear function."""
        result = self.solver.integral("x", lower_bound=0, upper_bound=5)
        assert result['is_definite'] == True
        assert '25/2' in result['simplified'] or '12.5' in result['simplified']
    
    def test_definite_integral_constant(self):
        """Test definite integral of constant."""
        result = self.solver.integral("3", lower_bound=1, upper_bound=4)
        assert result['is_definite'] == True
        assert '9' in result['simplified']
    
    def test_integral_specified_variable(self):
        """Test integral with specified variable."""
        result = self.solver.integral("y**2", variable="y")
        assert result['variable'] == 'y'
        assert result['is_definite'] == False
    
    def test_integral_trigonometric(self):
        """Test integral of trigonometric function."""
        result = self.solver.integral("cos(x)")
        assert result['variable'] == 'x'
        assert result['is_definite'] == False
        assert 'sin(x)' in result['simplified']
    
    def test_integral_exponential(self):
        """Test integral of exponential function."""
        result = self.solver.integral("exp(x)")
        assert result['variable'] == 'x'
        assert result['is_definite'] == False
        assert 'exp(x)' in result['simplified']
    
    def test_definite_integral_with_string_bounds(self):
        """Test definite integral with string bounds."""
        result = self.solver.integral("x", lower_bound="0", upper_bound="5")
        assert result['is_definite'] == True
        assert result['lower_bound'] == '0'
        assert result['upper_bound'] == '5'
    
    def test_integral_convenience_function(self):
        """Test module-level integral function."""
        result = integral("x**2")
        assert result['is_definite'] == False
        assert result['variable'] == 'x'
    
    def test_definite_integral_convenience_function(self):
        """Test module-level integral function with bounds."""
        result = integral("x", lower_bound=0, upper_bound=1)
        assert result['is_definite'] == True
        assert '1/2' in result['simplified'] or '0.5' in result['simplified']
    
    def test_integral_complex_expression(self):
        """Test integral of more complex expression."""
        result = self.solver.integral("x**2 + 3*x + 2")
        assert result['is_definite'] == False
        assert result['variable'] == 'x'
        assert 'x**3' in result['simplified'] or 'x**2' in result['simplified']

