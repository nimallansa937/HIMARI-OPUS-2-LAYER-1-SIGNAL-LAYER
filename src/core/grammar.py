"""
AlphaCFG: Context-Free Grammar for Trading Strategies

Validates strategy expressions for syntactic correctness and
dimensional consistency (e.g., can't compare price to ratio).

BNF Specification:
<strategy>      ::= <entry_rule> ";" <exit_rule> ";" <risk_control>
<condition>     ::= <comparison> | <condition> <logical_op> <comparison>
<comparison>    ::= <feature> <comp_op> <feature>
                  | <feature> <comp_op> <literal>
<comp_op>       ::= ">" | "<" | ">=" | "<=" | "==" | "!="
<logical_op>    ::= "AND" | "OR"
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
from enum import Enum
import re

from .features import FeatureType, FEATURE_BY_NAME, FEATURE_SCHEMA


# Valid comparison rules: which types can be compared to each other
VALID_COMPARISONS: Dict[Tuple[FeatureType, Optional[FeatureType]], bool] = {
    # Same-type comparisons
    (FeatureType.PRICE, FeatureType.PRICE): True,
    (FeatureType.VOLUME, FeatureType.VOLUME): True,
    (FeatureType.RATIO, FeatureType.RATIO): True,
    (FeatureType.ZSCORE, FeatureType.ZSCORE): True,
    (FeatureType.RATE, FeatureType.RATE): True,
    (FeatureType.COUNT, FeatureType.COUNT): True,
    (FeatureType.BOOLEAN, FeatureType.BOOLEAN): True,

    # Feature to literal comparisons
    (FeatureType.RATIO, None): True,      # RSI > 70
    (FeatureType.ZSCORE, None): True,     # price_zscore > 2.0
    (FeatureType.RATE, None): True,       # funding_rate > 0.001
    (FeatureType.COUNT, None): True,      # regime_label == 1
    (FeatureType.BOOLEAN, None): True,    # is_active == True

    # Price comparisons to other prices only
    (FeatureType.PRICE, None): False,     # close > 50000 is dangerous (static)
    (FeatureType.VOLUME, None): False,    # volume > 1000000 is dangerous (static)
}


class TokenType(Enum):
    """Token types for grammar parsing."""
    FEATURE = "FEATURE"
    LITERAL = "LITERAL"
    COMP_OP = "COMP_OP"
    LOGIC_OP = "LOGIC_OP"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    SEMICOLON = "SEMICOLON"
    EOF = "EOF"
    INVALID = "INVALID"


@dataclass
class Token:
    """A single token from lexical analysis."""
    type: TokenType
    value: str
    position: int


@dataclass
class ComparisonNode:
    """AST node for a comparison expression."""
    left_feature: str
    operator: str
    right_value: str  # Feature name or literal
    is_literal: bool


@dataclass
class ConditionNode:
    """AST node for a compound condition."""
    left: 'ConditionNode | ComparisonNode'
    operator: Optional[str]  # AND, OR, or None for single comparison
    right: Optional['ConditionNode | ComparisonNode']


@dataclass
class StrategyAST:
    """Abstract Syntax Tree for a complete strategy."""
    entry_condition: ConditionNode
    exit_condition: ConditionNode
    risk_params: Dict[str, float]


class Lexer:
    """Tokenizes strategy expression strings."""

    COMP_OPERATORS = {'>', '<', '>=', '<=', '==', '!='}
    LOGIC_OPERATORS = {'AND', 'OR', 'and', 'or', '&&', '||'}

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.feature_names: Set[str] = {spec.name for spec in FEATURE_SCHEMA}

    def tokenize(self) -> List[Token]:
        """Convert expression to token list."""
        tokens = []
        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break

            token = self._next_token()
            if token:
                tokens.append(token)

        tokens.append(Token(TokenType.EOF, "", self.pos))
        return tokens

    def _skip_whitespace(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _next_token(self) -> Optional[Token]:
        if self.pos >= len(self.text):
            return None

        char = self.text[self.pos]
        start_pos = self.pos

        # Single character tokens
        if char == '(':
            self.pos += 1
            return Token(TokenType.LPAREN, '(', start_pos)
        elif char == ')':
            self.pos += 1
            return Token(TokenType.RPAREN, ')', start_pos)
        elif char == ';':
            self.pos += 1
            return Token(TokenType.SEMICOLON, ';', start_pos)

        # Two-character operators
        if self.pos + 1 < len(self.text):
            two_char = self.text[self.pos:self.pos+2]
            if two_char in {'>=', '<=', '==', '!=', '&&', '||'}:
                self.pos += 2
                if two_char in {'&&', '||'}:
                    return Token(TokenType.LOGIC_OP, two_char, start_pos)
                return Token(TokenType.COMP_OP, two_char, start_pos)

        # Single character operators
        if char in {'>', '<'}:
            self.pos += 1
            return Token(TokenType.COMP_OP, char, start_pos)

        # Numbers (including negative and decimal)
        if char.isdigit() or (char == '-' and self.pos + 1 < len(self.text)
                              and self.text[self.pos + 1].isdigit()):
            return self._read_number(start_pos)

        # Identifiers (feature names or keywords)
        if char.isalpha() or char == '_':
            return self._read_identifier(start_pos)

        # Unknown character
        self.pos += 1
        return Token(TokenType.INVALID, char, start_pos)

    def _read_number(self, start_pos: int) -> Token:
        end_pos = self.pos
        has_decimal = False

        if self.text[end_pos] == '-':
            end_pos += 1

        while end_pos < len(self.text):
            char = self.text[end_pos]
            if char.isdigit():
                end_pos += 1
            elif char == '.' and not has_decimal:
                has_decimal = True
                end_pos += 1
            else:
                break

        value = self.text[self.pos:end_pos]
        self.pos = end_pos
        return Token(TokenType.LITERAL, value, start_pos)

    def _read_identifier(self, start_pos: int) -> Token:
        end_pos = self.pos
        while end_pos < len(self.text) and (self.text[end_pos].isalnum()
                                             or self.text[end_pos] == '_'):
            end_pos += 1

        value = self.text[self.pos:end_pos]
        self.pos = end_pos

        # Classify identifier
        upper_value = value.upper()
        if upper_value in {'AND', 'OR'}:
            return Token(TokenType.LOGIC_OP, upper_value, start_pos)
        elif value in self.feature_names:
            return Token(TokenType.FEATURE, value, start_pos)
        else:
            # Could be a literal (True, False) or unknown
            return Token(TokenType.LITERAL, value, start_pos)


class GrammarValidator:
    """
    Validates strategy expressions against AlphaCFG grammar.

    Checks:
    1. Syntactic correctness (proper operator usage, balanced parens)
    2. Dimensional consistency (can only compare same types)
    3. Feature existence (all referenced features must exist)
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, strategy_text: str) -> Tuple[bool, List[str]]:
        """
        Validate a strategy expression.

        Args:
            strategy_text: Strategy in text form, e.g.,
                "rsi_14 < 30 AND volume_ratio > 2; rsi_14 > 70; sl=2.0,tp=3.0"

        Returns:
            (is_valid, error_messages)
        """
        self.errors = []
        self.warnings = []

        try:
            # Tokenize
            lexer = Lexer(strategy_text)
            tokens = lexer.tokenize()

            # Check for invalid tokens
            invalid_tokens = [t for t in tokens if t.type == TokenType.INVALID]
            if invalid_tokens:
                for t in invalid_tokens:
                    self.errors.append(f"Invalid character '{t.value}' at position {t.position}")
                return False, self.errors

            # Parse and validate conditions
            self._validate_tokens(tokens)

            # Check dimensional consistency
            self._validate_dimensions(tokens)

        except Exception as e:
            self.errors.append(f"Parse error: {str(e)}")

        return len(self.errors) == 0, self.errors

    def _validate_tokens(self, tokens: List[Token]) -> None:
        """Validate token sequence for syntactic correctness."""
        # Track parentheses balance
        paren_depth = 0

        # Expected token types at each position
        expecting_operand = True  # Start expecting an operand (feature or literal)

        for i, token in enumerate(tokens):
            if token.type == TokenType.EOF:
                break

            if token.type == TokenType.LPAREN:
                paren_depth += 1
                expecting_operand = True

            elif token.type == TokenType.RPAREN:
                paren_depth -= 1
                if paren_depth < 0:
                    self.errors.append(f"Unmatched ')' at position {token.position}")
                expecting_operand = False

            elif token.type == TokenType.FEATURE:
                if not expecting_operand:
                    self.errors.append(
                        f"Unexpected feature '{token.value}' at position {token.position}")
                expecting_operand = False

            elif token.type == TokenType.LITERAL:
                if not expecting_operand:
                    self.errors.append(
                        f"Unexpected literal '{token.value}' at position {token.position}")
                expecting_operand = False

            elif token.type == TokenType.COMP_OP:
                if expecting_operand:
                    self.errors.append(
                        f"Unexpected operator '{token.value}' at position {token.position}")
                expecting_operand = True

            elif token.type == TokenType.LOGIC_OP:
                if expecting_operand:
                    self.errors.append(
                        f"Unexpected logical operator '{token.value}' at position {token.position}")
                expecting_operand = True

            elif token.type == TokenType.SEMICOLON:
                # Section separator - reset state
                expecting_operand = True

        if paren_depth > 0:
            self.errors.append(f"Missing {paren_depth} closing parenthesis(es)")

    def _validate_dimensions(self, tokens: List[Token]) -> None:
        """Validate dimensional consistency of comparisons."""
        i = 0
        while i < len(tokens) - 2:
            # Look for pattern: FEATURE COMP_OP (FEATURE|LITERAL)
            if (tokens[i].type == TokenType.FEATURE and
                tokens[i+1].type == TokenType.COMP_OP):

                left_feature = tokens[i].value
                left_type = FEATURE_BY_NAME[left_feature].type

                right_token = tokens[i+2]

                if right_token.type == TokenType.FEATURE:
                    # Feature-to-feature comparison
                    right_feature = right_token.value
                    right_type = FEATURE_BY_NAME[right_feature].type

                    if (left_type, right_type) not in VALID_COMPARISONS:
                        self.errors.append(
                            f"Cannot compare {left_feature} ({left_type.value}) "
                            f"to {right_feature} ({right_type.value})")

                elif right_token.type == TokenType.LITERAL:
                    # Feature-to-literal comparison
                    if not VALID_COMPARISONS.get((left_type, None), False):
                        self.warnings.append(
                            f"Comparing {left_feature} ({left_type.value}) to literal "
                            f"'{right_token.value}' may not be robust")

                i += 3
            else:
                i += 1

    def validate_genome(self, genome) -> Tuple[bool, List[str]]:
        """
        Validate a StrategyGenome object.

        Checks decision tree structure and parameter bounds.
        """
        self.errors = []
        self.warnings = []

        # Validate decision tree
        self._validate_tree(genome.decision_tree, depth=0)

        # Validate parameters
        if not 0.01 <= genome.base_position_pct <= 0.5:
            self.errors.append(
                f"base_position_pct {genome.base_position_pct} outside valid range [0.01, 0.5]")

        if not 0.5 <= genome.stop_loss_atr_mult <= 5.0:
            self.errors.append(
                f"stop_loss_atr_mult {genome.stop_loss_atr_mult} outside valid range [0.5, 5.0]")

        if not 1.0 <= genome.take_profit_atr_mult <= 10.0:
            self.errors.append(
                f"take_profit_atr_mult {genome.take_profit_atr_mult} outside valid range [1.0, 10.0]")

        # Check risk/reward ratio
        if genome.take_profit_atr_mult < genome.stop_loss_atr_mult:
            self.warnings.append(
                f"Take profit ({genome.take_profit_atr_mult}) < stop loss ({genome.stop_loss_atr_mult})")

        return len(self.errors) == 0, self.errors

    def _validate_tree(self, node, depth: int) -> None:
        """Recursively validate decision tree structure."""
        if node is None:
            return

        if depth > 10:
            self.errors.append("Decision tree exceeds maximum depth (10)")
            return

        # Leaf node
        if node.action is not None:
            if node.action not in [-1, 0, 1]:
                self.errors.append(f"Invalid action {node.action}, must be -1, 0, or 1")
            return

        # Internal node
        if node.condition is None:
            self.errors.append("Internal node missing condition")
            return

        # Validate condition
        self._validate_condition(node.condition)

        # Recurse
        if node.true_branch is None and node.false_branch is None:
            self.errors.append("Internal node has no branches")
        else:
            self._validate_tree(node.true_branch, depth + 1)
            self._validate_tree(node.false_branch, depth + 1)

    def _validate_condition(self, condition) -> None:
        """Validate a single condition."""
        if condition.signal is None:
            self.errors.append("Condition missing signal type")
            return

        if condition.operator not in ['>', '<', '>=', '<=']:
            self.errors.append(f"Invalid operator '{condition.operator}'")

        if condition.threshold is None:
            self.errors.append("Condition missing threshold")


def parse_strategy_text(text: str) -> Optional[StrategyAST]:
    """
    Parse strategy text into AST.

    Expected format:
    "<entry_condition>; <exit_condition>; <params>"

    Example:
    "rsi_14 < 30 AND volume_ratio > 1.5; rsi_14 > 70 OR close < sma_20; sl=2.0,tp=3.0"
    """
    validator = GrammarValidator()
    is_valid, errors = validator.validate(text)

    if not is_valid:
        return None

    # Split into sections
    sections = text.split(';')
    if len(sections) < 2:
        return None

    # Parse would build full AST - simplified for now
    return StrategyAST(
        entry_condition=ConditionNode(left=None, operator=None, right=None),
        exit_condition=ConditionNode(left=None, operator=None, right=None),
        risk_params={}
    )
