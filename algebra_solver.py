"""
Algebra and Calculus Solver Module

This module provides functionality to solve algebraic equations,
determine if expressions are constant, identify variables,
and perform calculus operations (derivatives and integrals).
"""

from sympy import symbols, sympify, solve, simplify, Symbol, diff, integrate, oo
from sympy.core.expr import Expr
from typing import Union, Optional, List, Tuple


class AlgebraSolver:
    """A class to solve algebraic equations and analyze expressions."""
    
    def __init__(self):
        """Initialize the algebra solver."""
        pass
    
    def parse_equation(self, equation_str: str) -> tuple:
        """
        Parse an equation string into left and right sides.
        
        Args:
            equation_str: String representation of equation (e.g., "2*x + 3 = 7")
            
        Returns:
            Tuple of (left_side, right_side) as sympy expressions
        """
        if '=' not in equation_str:
            raise ValueError("Equation must contain '=' sign")
        
        left_str, right_str = equation_str.split('=', 1)
        left_expr = sympify(left_str.strip())
        right_expr = sympify(right_str.strip())
        
        return left_expr, right_expr
    
    def is_constant_expression(self, expression: Union[str, Expr]) -> bool:
        """
        Determine if an expression is constant (has no variables).
        
        Args:
            expression: String or sympy expression to check
            
        Returns:
            True if expression is constant, False otherwise
        """
        if isinstance(expression, str):
            expr = sympify(expression)
        else:
            expr = expression
        
        # Get all symbols in the expression
        symbols_in_expr = expr.free_symbols
        
        # If there are no symbols, it's a constant
        return len(symbols_in_expr) == 0
    
    def get_variables(self, expression: Union[str, Expr]) -> List[str]:
        """
        Extract variable names from an expression.
        
        Args:
            expression: String or sympy expression
            
        Returns:
            List of variable names found in the expression
        """
        if isinstance(expression, str):
            expr = sympify(expression)
        else:
            expr = expression
        
        return sorted([str(symbol) for symbol in expr.free_symbols])
    
    def solve_equation(self, equation_str: str) -> dict:
        """
        Solve an algebraic equation and return the variable and solution.
        
        Args:
            equation_str: String representation of equation (e.g., "2*x + 3 = 7")
            
        Returns:
            Dictionary with keys:
                - 'variable': variable name or None
                - 'solution': solution value(s) or "no solution"
                - 'is_constant': whether the equation is constant
        """
        try:
            left_expr, right_expr = self.parse_equation(equation_str)
            
            # Move everything to left side: left - right = 0
            equation = simplify(left_expr - right_expr)
            
            # Check if it's a constant expression
            if self.is_constant_expression(equation):
                # If constant and equals 0, infinite solutions; otherwise no solution
                if equation == 0:
                    return {
                        'variable': None,
                        'solution': 'infinite solutions',
                        'is_constant': True
                    }
                else:
                    return {
                        'variable': None,
                        'solution': 'no solution',
                        'is_constant': True
                    }
            
            # Get variables
            variables = self.get_variables(equation)
            
            if len(variables) == 0:
                return {
                    'variable': None,
                    'solution': 'no solution',
                    'is_constant': True
                }
            
            if len(variables) > 1:
                # Multiple variables - solve for the first one
                var_symbol = symbols(variables[0])
            else:
                var_symbol = symbols(variables[0])
            
            # Solve the equation
            solutions = solve(equation, var_symbol)
            
            if len(solutions) == 0:
                return {
                    'variable': variables[0] if variables else None,
                    'solution': 'no solution',
                    'is_constant': False
                }
            
            # Convert solutions to Python numbers if possible
            result_solutions = []
            for sol in solutions:
                try:
                    # Try to convert to float
                    result_solutions.append(float(sol.evalf()))
                except:
                    result_solutions.append(str(sol))
            
            return {
                'variable': variables[0] if variables else None,
                'solution': result_solutions[0] if len(result_solutions) == 1 else result_solutions,
                'is_constant': False
            }
            
        except Exception as e:
            return {
                'variable': None,
                'solution': 'no solution',
                'is_constant': False,
                'error': str(e)
            }
    
    def get_variable_or_no_solution(self, equation_str: str) -> str:
        """
        Return the variable name or "no solution" if none can be found.
        
        Args:
            equation_str: String representation of equation
            
        Returns:
            Variable name as string, or "no solution" if no variable found
        """
        result = self.solve_equation(equation_str)
        
        if result['variable'] is None:
            return 'no solution'
        
        return result['variable']
    
    def derivative(self, expression: Union[str, Expr], variable: Optional[str] = None, order: int = 1) -> dict:
        """
        Compute the derivative of an expression.
        
        Args:
            expression: String or sympy expression to differentiate
            variable: Variable to differentiate with respect to (if None, uses first variable found)
            order: Order of derivative (default: 1 for first derivative)
            
        Returns:
            Dictionary with keys:
                - 'derivative': derivative expression as string
                - 'variable': variable used for differentiation
                - 'order': order of derivative
                - 'simplified': simplified derivative expression as string
        """
        try:
            if isinstance(expression, str):
                expr = sympify(expression)
            else:
                expr = expression
            
            # Determine variable
            variables = self.get_variables(expr)
            if variable is None:
                if len(variables) == 0:
                    # Constant expression, derivative is 0
                    return {
                        'derivative': '0',
                        'variable': None,
                        'order': order,
                        'simplified': '0'
                    }
                var_symbol = symbols(variables[0])
                var_name = variables[0]
            else:
                var_symbol = symbols(variable)
                var_name = variable
            
            # Compute derivative
            derivative_expr = diff(expr, var_symbol, order)
            simplified_derivative = simplify(derivative_expr)
            
            return {
                'derivative': str(derivative_expr),
                'variable': var_name,
                'order': order,
                'simplified': str(simplified_derivative)
            }
            
        except Exception as e:
            return {
                'derivative': None,
                'variable': variable,
                'order': order,
                'error': str(e)
            }
    
    def integral(self, expression: Union[str, Expr], variable: Optional[str] = None, 
                 lower_bound: Optional[Union[float, str]] = None,
                 upper_bound: Optional[Union[float, str]] = None) -> dict:
        """
        Compute the integral of an expression.
        
        Args:
            expression: String or sympy expression to integrate
            variable: Variable to integrate with respect to (if None, uses first variable found)
            lower_bound: Lower bound for definite integral (None for indefinite)
            upper_bound: Upper bound for definite integral (None for indefinite)
            
        Returns:
            Dictionary with keys:
                - 'integral': integral expression as string
                - 'variable': variable used for integration
                - 'is_definite': whether it's a definite integral
                - 'lower_bound': lower bound if definite
                - 'upper_bound': upper bound if definite
                - 'simplified': simplified integral expression as string
        """
        try:
            if isinstance(expression, str):
                expr = sympify(expression)
            else:
                expr = expression
            
            # Determine variable
            variables = self.get_variables(expr)
            if variable is None:
                if len(variables) == 0:
                    # Constant expression
                    if lower_bound is not None and upper_bound is not None:
                        # Definite integral of constant
                        try:
                            lower = float(lower_bound) if isinstance(lower_bound, str) else lower_bound
                            upper = float(upper_bound) if isinstance(upper_bound, str) else upper_bound
                            result = expr * (upper - lower)
                            return {
                                'integral': str(result),
                                'variable': None,
                                'is_definite': True,
                                'lower_bound': lower_bound,
                                'upper_bound': upper_bound,
                                'simplified': str(simplify(result))
                            }
                        except:
                            pass
                    # Indefinite integral of constant
                    return {
                        'integral': f'{expr}*x',
                        'variable': 'x',
                        'is_definite': False,
                        'simplified': f'{expr}*x'
                    }
                var_symbol = symbols(variables[0])
                var_name = variables[0]
            else:
                var_symbol = symbols(variable)
                var_name = variable
            
            # Handle bounds
            is_definite = lower_bound is not None and upper_bound is not None
            
            if is_definite:
                # Definite integral
                try:
                    # Convert bounds to numbers or symbols
                    lower = sympify(str(lower_bound)) if isinstance(lower_bound, str) else lower_bound
                    upper = sympify(str(upper_bound)) if isinstance(upper_bound, str) else upper_bound
                    
                    # Handle infinity
                    if isinstance(lower_bound, str) and lower_bound.lower() in ['-oo', '-inf', '-infinity']:
                        lower = -oo
                    elif isinstance(lower_bound, str) and lower_bound.lower() in ['oo', 'inf', 'infinity']:
                        lower = oo
                    
                    if isinstance(upper_bound, str) and upper_bound.lower() in ['-oo', '-inf', '-infinity']:
                        upper = -oo
                    elif isinstance(upper_bound, str) and upper_bound.lower() in ['oo', 'inf', 'infinity']:
                        upper = oo
                    
                    integral_expr = integrate(expr, (var_symbol, lower, upper))
                    simplified_integral = simplify(integral_expr)
                    
                    return {
                        'integral': str(integral_expr),
                        'variable': var_name,
                        'is_definite': True,
                        'lower_bound': str(lower_bound),
                        'upper_bound': str(upper_bound),
                        'simplified': str(simplified_integral)
                    }
                except Exception as e:
                    return {
                        'integral': None,
                        'variable': var_name,
                        'is_definite': True,
                        'error': str(e)
                    }
            else:
                # Indefinite integral
                integral_expr = integrate(expr, var_symbol)
                simplified_integral = simplify(integral_expr)
                
                return {
                    'integral': str(integral_expr),
                    'variable': var_name,
                    'is_definite': False,
                    'simplified': str(simplified_integral)
                }
            
        except Exception as e:
            return {
                'integral': None,
                'variable': variable,
                'is_definite': lower_bound is not None and upper_bound is not None,
                'error': str(e)
            }


# Convenience functions for direct use
def solve_equation(equation_str: str) -> dict:
    """Solve an equation and return result dictionary."""
    solver = AlgebraSolver()
    return solver.solve_equation(equation_str)


def is_constant_expression(expression: Union[str, Expr]) -> bool:
    """Check if an expression is constant."""
    solver = AlgebraSolver()
    return solver.is_constant_expression(expression)


def get_variable_or_no_solution(equation_str: str) -> str:
    """Get variable name or 'no solution'."""
    solver = AlgebraSolver()
    return solver.get_variable_or_no_solution(equation_str)


def derivative(expression: Union[str, Expr], variable: Optional[str] = None, order: int = 1) -> dict:
    """Compute the derivative of an expression."""
    solver = AlgebraSolver()
    return solver.derivative(expression, variable, order)


def integral(expression: Union[str, Expr], variable: Optional[str] = None,
             lower_bound: Optional[Union[float, str]] = None,
             upper_bound: Optional[Union[float, str]] = None) -> dict:
    """Compute the integral of an expression."""
    solver = AlgebraSolver()
    return solver.integral(expression, variable, lower_bound, upper_bound)

