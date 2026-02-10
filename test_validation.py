"""
Test script to verify the file/sheet name validation logic
"""

import re

def is_valid_generator_name(text):
    """Check if text contains valid generator-related keywords (case-insensitive, flexible underscore)"""
    if not text:
        return False
    
    # Normalize: remove underscores, convert to lowercase
    normalized = text.replace('_', ' ').replace('-', ' ').lower().strip()
    
    # Check for valid patterns
    valid_patterns = [
        r'\bgenerator[\s_]*only\b',
        r'\bgen[\s_]*only\b',
        r'\bgenerator\b'
    ]
    
    for pattern in valid_patterns:
        # Make pattern flexible with underscores and spaces
        flexible_pattern = pattern.replace('[\\s_]*', '[\\s_-]*')
        if re.search(flexible_pattern, normalized):
            return True
    
    return False


# Test cases
test_cases = [
    # Should pass
    ("Generator_Only", True),
    ("generator_only", True),
    ("GENERATOR_ONLY", True),
    ("Gen_Only", True),
    ("gen_only", True),
    ("GEN_ONLY", True),
    ("generator only", True),
    ("Generator Only", True),
    ("GEN ONLY", True),
    ("Gen Only", True),
    ("generator", True),
    ("Generator", True),
    ("GENERATOR", True),
    ("My_Generator_Only_Data.xlsx", True),
    ("gen only report.xlsx", True),
    ("Generator-Only-2024.xlsx", True),
    ("data_generator_only.xlsx", True),
    
    # Should fail
    ("Data", False),
    ("Report", False),
    ("Fuel_Consumption", False),
    ("", False),
    ("gen", False),  # Just "gen" alone doesn't match
    ("generate", False),  # Not "generator"
]

print("Testing file/sheet name validation:\n")
print("-" * 70)

all_passed = True
for test_name, expected in test_cases:
    result = is_valid_generator_name(test_name)
    status = "✓ PASS" if result == expected else "✗ FAIL"
    if result != expected:
        all_passed = False
    
    # Show normalized version for debugging
    if test_name:
        normalized = test_name.replace('_', ' ').replace('-', ' ').lower().strip()
        print(f"{status} | '{test_name}' -> {result} (expected {expected}) [normalized: '{normalized}']")
    else:
        print(f"{status} | '{test_name}' -> {result} (expected {expected})")

print("-" * 70)
if all_passed:
    print("\n🎉 All tests passed!")
else:
    print("\n❌ Some tests failed!")

# Additional specific test for the user's case
print("\n" + "=" * 70)
print("SPECIFIC TEST FOR USER'S CASE:")
print("=" * 70)
test_sheet = "Generator_only"
result = is_valid_generator_name(test_sheet)
normalized = test_sheet.replace('_', ' ').replace('-', ' ').lower().strip()
print(f"Sheet name: '{test_sheet}'")
print(f"Normalized: '{normalized}'")
print(f"Result: {result}")
print(f"Expected: True")
if result:
    print("✓ This sheet name SHOULD BE ACCEPTED")
else:
    print("✗ ERROR: This sheet name is being REJECTED but should be accepted!")
