"""Input validation utilities"""

import re
from typing import Tuple

def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email address format
    Returns: (is_valid, error_message)
    """
    if not email:
        return False, "Email is required"
    
    # Comprehensive email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Please enter a valid email address (e.g., john@example.com)"
    
    # Additional checks
    if email.count('@') != 1:
        return False, "Email must contain exactly one @ symbol"
    
    local, domain = email.split('@')
    
    if len(local) > 64:
        return False, "Email username is too long"
    
    if len(domain) > 255:
        return False, "Email domain is too long"
    
    if domain.startswith('.') or domain.endswith('.'):
        return False, "Email domain cannot start or end with a dot"
    
    if '..' in email:
        return False, "Email cannot contain consecutive dots"
    
    return True, ""

def validate_phone(phone: str) -> Tuple[bool, str]:
    """
    Validate phone number format (US/International)
    Returns: (is_valid, error_message)
    """
    if not phone:
        return False, "Phone number is required"
    
    # Remove common formatting characters
    cleaned = re.sub(r'[\s\-\(\)\.]', '', phone)
    
    # US phone number patterns
    us_pattern = r'^(\+1)?[2-9]\d{9}$'  # US format with optional +1
    intl_pattern = r'^\+\d{10,15}$'  # International format
    
    if re.match(us_pattern, cleaned):
        return True, ""
    elif re.match(intl_pattern, cleaned):
        return True, ""
    else:
        # Try to provide helpful feedback
        if len(cleaned) < 10:
            return False, "Phone number is too short. Please include area code."
        elif len(cleaned) > 15:
            return False, "Phone number is too long."
        elif not cleaned.replace('+', '').isdigit():
            return False, "Phone number can only contain digits and formatting characters."
        else:
            return False, "Please enter a valid phone number (e.g., 555-123-4567 or +1-555-123-4567)"

def validate_address(address: str) -> Tuple[bool, str]:
    """
    Validate physical address
    Returns: (is_valid, error_message)
    """
    if not address:
        return False, "Address is required"
    
    # Check minimum length
    if len(address.strip()) < 10:
        return False, "Please enter a complete address"
    
    # Check for required components (loose validation)
    address_lower = address.lower()
    
    # Should contain at least a number and street name
    has_number = any(char.isdigit() for char in address)
    
    if not has_number:
        return False, "Please include a street number in your address"
    
    # Check for common address keywords (at least one should be present)
    common_keywords = [
        'street', 'st', 'avenue', 'ave', 'road', 'rd', 'drive', 'dr',
        'lane', 'ln', 'boulevard', 'blvd', 'court', 'ct', 'place', 'pl',
        'way', 'circle', 'cir', 'plaza', 'square', 'parkway', 'pkwy',
        'apartment', 'apt', 'suite', 'ste', 'unit', '#'
    ]
    
    has_keyword = any(keyword in address_lower for keyword in common_keywords)
    
    if not has_keyword:
        # Still allow if it looks like it has enough components
        words = address.split()
        if len(words) < 3:
            return False, "Please enter a complete street address"
    
    # Check for potential SQL injection or script injection
    dangerous_patterns = ['<script', 'javascript:', 'onclick', 'DROP TABLE', 'DELETE FROM', '--']
    for pattern in dangerous_patterns:
        if pattern.lower() in address_lower:
            return False, "Invalid characters detected in address"
    
    return True, ""

def sanitize_input(text: str, max_length: int = 500) -> str:
    """
    Sanitize user input to prevent injection attacks
    """
    if not text:
        return ""
    
    # Remove any HTML/script tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove any potential SQL injection patterns
    text = re.sub(r'(DROP|DELETE|INSERT|UPDATE|SELECT|UNION|CREATE|ALTER|EXEC|EXECUTE|SCRIPT)', '', text, flags=re.IGNORECASE)
    
    # Limit length
    text = text[:max_length]
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text.strip()