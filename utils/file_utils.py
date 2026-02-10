"""
File handling utilities for Excel file operations
Handles file validation, reading, and sheet detection
"""

import os
import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def allowed_file(filename, allowed_extensions=None):
    """
    Check if file has allowed extension
    
    Args:
        filename: Name of the file to check
        allowed_extensions: Set of allowed extensions (default: {'xlsx', 'xls', 'ods'})
    
    Returns:
        bool: True if file extension is allowed
    """
    if allowed_extensions is None:
        allowed_extensions = {'xlsx', 'xls', 'ods'}
    
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def read_excel_robust(filepath, sheet_name=None):
    """
    Read Excel file with fallback engines for better compatibility
    
    Args:
        filepath: Path to Excel file
        sheet_name: Name of sheet to read (optional)
    
    Returns:
        pd.DataFrame: Loaded DataFrame
    
    Raises:
        Exception: If file cannot be read with any engine
    """
    engines = ['openpyxl', 'odf', 'xlrd', None]
    last_error = None
    
    for engine in engines:
        try:
            logger.debug(f"Trying to read with engine: {engine}")
            if engine:
                if sheet_name:
                    df = pd.read_excel(filepath, sheet_name=sheet_name, engine=engine)
                else:
                    df = pd.read_excel(filepath, engine=engine)
            else:
                if sheet_name:
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                else:
                    df = pd.read_excel(filepath)
            logger.debug(f"Successfully read with engine: {engine}")
            return df
        except Exception as e:
            logger.debug(f"Engine {engine} failed: {e}")
            last_error = e
            continue
    
    raise Exception(f"Could not read Excel file with any engine. Last error: {last_error}")


def is_valid_generator_name(text):
    """
    Check if text contains valid generator-related keywords
    
    Args:
        text: Text to check (sheet name or filename)
    
    Returns:
        bool: True if text matches generator criteria
    """
    if not text:
        logger.debug("is_valid_generator_name() - Empty text")
        return False
    
    # Normalize: remove underscores, convert to lowercase
    normalized = text.replace('_', ' ').replace('-', ' ').lower().strip()
    logger.debug(f"is_valid_generator_name() - Input: '{text}' -> Normalized: '{normalized}'")
    
    # Check for valid patterns
    valid_patterns = [
        r'\bgenerator[\s_]*only\b',
        r'\bgen[\s_]*only\b',
        r'\bgenerator\b'
    ]
    
    for pattern in valid_patterns:
        # Make pattern flexible with underscores and spaces
        flexible_pattern = pattern.replace('[\\s_]*', '[\\s_-]*')
        match = re.search(flexible_pattern, normalized)
        logger.debug(f"  Testing pattern '{pattern}' -> Match: {match is not None}")
        if match:
            logger.debug(f"  ✓ MATCHED with pattern '{pattern}'!")
            return True
    
    logger.debug(f"  ✗ NO MATCH for '{text}'")
    return False


def find_valid_sheet(excel_file):
    """
    Find a valid sheet in the Excel file that matches generator criteria
    
    Args:
        excel_file: Path to Excel file
    
    Returns:
        str: Name of valid sheet, or None if not found
    """
    logger.info(f"{'='*80}")
    logger.debug(f"find_valid_sheet() called with: {excel_file}")
    
    try:
        # Try multiple engines to handle different Excel file formats
        xl_file = None
        last_error = None
        for engine in ['openpyxl', 'odf', 'xlrd', None]:
            try:
                logger.debug(f"Trying engine: {engine}")
                xl_file = pd.ExcelFile(excel_file, engine=engine) if engine else pd.ExcelFile(excel_file)
                logger.debug(f"Successfully opened with engine: {engine}")
                break
            except Exception as e:
                logger.debug(f"Engine {engine} failed: {e}")
                last_error = e
                continue
        
        if xl_file is None:
            raise Exception(
                f"Could not read Excel file with any engine. "
                f"The file may be corrupted or in an unsupported format. "
                f"Please try re-saving it in Excel as .xlsx format. "
                f"Last error: {last_error}"
            )
        
        sheet_names = xl_file.sheet_names
        logger.info(f"Checking Excel file with sheets: {sheet_names}")
        logger.debug(f"Found {len(sheet_names)} sheet(s): {sheet_names}")
        logger.debug(f"Sheet names (repr): {[repr(s) for s in sheet_names]}")
        
        # First priority: check if any sheet name matches our criteria
        logger.debug("\nSTEP 1 - Checking sheet names...")
        for idx, sheet in enumerate(sheet_names):
            # Strip any leading/trailing whitespace from sheet name
            sheet_clean = sheet.strip() if isinstance(sheet, str) else sheet
            logger.debug(
                f"\nChecking sheet #{idx+1}: '{sheet}' "
                f"(stripped: '{sheet_clean}', repr: {repr(sheet)}, "
                f"bytes: {sheet.encode('utf-8') if isinstance(sheet, str) else 'N/A'})"
            )
            logger.debug(f"Checking sheet name: '{sheet_clean}'")
            
            if is_valid_generator_name(sheet_clean):
                logger.info(f"Found valid sheet name: '{sheet_clean}'")
                logger.debug(f"✓✓✓ FOUND VALID SHEET: '{sheet_clean}' - Returning this sheet")
                logger.info(f"{'='*80}\n")
                return sheet_clean
        
        # Second priority: if filename matches criteria, use first sheet
        logger.debug("\nSTEP 2 - No matching sheet found. Checking filename...")
        filename = os.path.basename(excel_file)
        logger.debug(f"Filename: '{filename}'")
        logger.debug(f"Checking filename: '{filename}'")
        
        if is_valid_generator_name(filename):
            first_sheet = sheet_names[0].strip() if isinstance(sheet_names[0], str) else sheet_names[0]
            logger.info(f"Filename '{filename}' matches criteria, using first sheet: '{first_sheet}'")
            logger.debug(f"✓✓✓ FILENAME MATCHES - Using first sheet: '{first_sheet}'")
            logger.info(f"{'='*80}\n")
            return first_sheet
        
        # No match found in either sheet names or filename
        logger.debug("\n✗✗✗ NO MATCH FOUND - Neither filename nor sheet names match")
        logger.warning(f"No valid sheet found. Filename: '{filename}', Sheet names: {sheet_names}")
        logger.info(f"{'='*80}\n")
        return None
        
    except Exception as e:
        logger.error(f"Error reading Excel file structure: {str(e)}")
        logger.debug(f"✗✗✗ EXCEPTION: {str(e)}")
        logger.info(f"{'='*80}\n")
        return None


def validate_file_format(filepath):
    """
    Validate Excel file format by checking magic bytes and internal structure
    
    Args:
        filepath: Path to file to validate
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return False, 'File is empty or does not exist'
        
        file_size = os.path.getsize(filepath)
        logger.debug(f"File size: {file_size} bytes")
        
        # Check magic bytes
        with open(filepath, 'rb') as f:
            magic_bytes = f.read(8)
            logger.debug(f"File magic bytes (hex): {magic_bytes.hex()}")
            logger.debug(f"File magic bytes (text): {magic_bytes[:4]}")
            
            # Check if it's a ZIP file (xlsx files are ZIP archives)
            is_zip = magic_bytes[:4] == b'PK\x03\x04'
            # Check if it's old Excel format
            is_old_excel = magic_bytes[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
            
            logger.debug(f"Is ZIP (xlsx): {is_zip}, Is old Excel (xls): {is_old_excel}")
            
            if not is_zip and not is_old_excel:
                return False, 'Invalid Excel file format. The file does not appear to be a valid Excel file.'
        
        # If it's a ZIP, check its internal structure
        if is_zip:
            import zipfile
            try:
                with zipfile.ZipFile(filepath, 'r') as zf:
                    file_list = zf.namelist()
                    logger.debug(f"ZIP contains {len(file_list)} files")
                    logger.debug(f"First 10 files: {file_list[:10]}")
                    
                    # Check for required xlsx files
                    has_content_types = '[Content_Types].xml' in file_list
                    has_workbook = any('workbook.xml' in f.lower() for f in file_list)
                    has_sheets = any('sheet' in f.lower() for f in file_list)
                    
                    logger.debug(f"Has [Content_Types].xml: {has_content_types}")
                    logger.debug(f"Has workbook.xml: {has_workbook}")
                    logger.debug(f"Has sheet files: {has_sheets}")
                    
                    if not has_content_types or not has_workbook:
                        return False, 'Corrupted Excel file. Missing required internal components.'
            except zipfile.BadZipFile:
                return False, 'Corrupted ZIP/Excel file.'
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error validating file format: {str(e)}")
        return False, f'Error validating file: {str(e)}'
