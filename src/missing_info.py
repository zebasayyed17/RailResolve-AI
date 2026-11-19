
import re
from datetime import datetime

from src.complaint_classifier import predict_complaint


# =====================================================
# 1. CATEGORY-SPECIFIC REQUIRED INFORMATION
# =====================================================

REQUIRED_FIELDS = {
    "Cleanliness": [
        "train_number",
        "journey_date",
        "location",
    ],

    "Ticketing": [
        "booking_reference",
    ],

    "Refund": [
        "booking_reference",
        "journey_date",
    ],

    "Delay": [
        "train_number",
        "journey_date",
        "location",
    ],

    "Facilities": [
        "train_number",
        "journey_date",
        "location",
    ],

    "Lost Property": [
        "train_number",
        "journey_date",
        "location",
        "item_description",
    ],

    "Staff Assistance": [
        "train_number",
        "journey_date",
        "location",
    ],

    "Information": [
        "train_number",
        "journey_date",
        "location",
    ],
}


# =====================================================
# 2. FOLLOW-UP QUESTIONS
# =====================================================

FIELD_QUESTIONS = {
    "booking_reference":
        "Please share your PNR or booking/transaction reference.",

    "train_number":
        "Please provide the 5-digit train number.",

    "journey_date":
        "Please provide the journey date (DD/MM/YYYY).",

    "location":
        "Please provide your coach, berth, station or platform details.",

    "item_description":
        "Please describe the missing item, including its type and colour.",
}


# =====================================================
# 3. DATE EXTRACTION
# =====================================================

def extract_date(text):
    """
    Extract and normalize a journey date.

    Numeric dates are interpreted using DD/MM/YYYY
    format, consistent with the configured Indian locale.
    """

    pattern = (
        r"(?<!\d)"
        r"(\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        r"(?!\d)"
    )

    match = re.search(pattern, text)

    if not match:
        return None

    date_text = match.group(1)

    date_formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d/%m/%y",
        "%d-%m-%y",
    ]

    for date_format in date_formats:
        try:
            parsed_date = datetime.strptime(
                date_text,
                date_format,
            )

            return parsed_date.date().isoformat()

        except ValueError:
            continue

    return None


# =====================================================
# 4. PNR / BOOKING REFERENCE EXTRACTION
# =====================================================


def extract_booking_reference(text):
    """Extract a labelled 10-digit PNR or booking/transaction reference."""
    import re

    patterns = [
        # Supports: PNR 1234567890, PNR: 1234567890,
        # PNR is 1234567890, and PNR number 1234567890.
        r"\bPNR\b\s*(?:(?:is|number|no\.?)\s*)?[:#-]?\s*(\d{10})\b",

        # Supports labelled booking references.
        r"\bbooking\s*(?:reference|ref|number|no\.?)\s*(?:is\s*)?[:#-]?\s*([A-Z0-9-]{5,20})\b",

        # Supports labelled transaction references.
        r"\btransaction\s*(?:reference|ref|number|no\.?)\s*(?:is\s*)?[:#-]?\s*([A-Z0-9-]{5,20})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None

# =====================================================
# 5. TRAIN NUMBER EXTRACTION
# =====================================================

def extract_train_number(text):
    """Extract a five-digit labelled train number."""

    pattern = (
        r"\btrain\s*(?:number|no\.?|#)"
        r"\s*[:#-]?\s*(\d{5})\b"
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


# =====================================================
# 6. LOCATION EXTRACTION
# =====================================================

def extract_location(text):
    """Extract coach, berth, platform or station details."""

    # Example: coach B2 or coach number B2
    coach_pattern = (
        r"\bcoach\s*(?:(?:number|no\.?)\s*)?"
        r"[:#-]?\s*([A-Z]{1,2}\s*-?\s*\d{1,2})\b"
    )

    coach_match = re.search(
        coach_pattern,
        text,
        re.IGNORECASE,
    )

    if coach_match:
        coach = re.sub(
            r"\s+",
            "",
            coach_match.group(1),
        )

        return f"Coach {coach.upper()}"

    # Example: berth 42
    berth_match = re.search(
        r"\bberth\s*(?:number|no\.?|#)?"
        r"\s*[:#-]?\s*(\d{1,3})\b",
        text,
        re.IGNORECASE,
    )

    if berth_match:
        return f"Berth {berth_match.group(1)}"

    # Example: platform 3
    platform_match = re.search(
        r"\bplatform\s*(?:number|no\.?|#)?"
        r"\s*[:#-]?\s*(\d+)\b",
        text,
        re.IGNORECASE,
    )

    if platform_match:
        return f"Platform {platform_match.group(1)}"

    # Example: at Pune Railway Station
    station_match = re.search(
        r"\b(?:at|from|near)\s+"
        r"([A-Za-z][A-Za-z .'-]{0,35}?)\s+"
        r"(?:railway\s+)?"
        r"(station|junction|terminal)\b",
        text,
        re.IGNORECASE,
    )

    if station_match:
        place = station_match.group(1).strip()
        station_type = station_match.group(2)

        return f"{place} {station_type.title()}"

    return None


# =====================================================
# 7. LOST ITEM DESCRIPTION EXTRACTION
# =====================================================

def extract_item_description(text):
    """Extract a basic lost-item description."""

    item_pattern = (
        r"\b("
        r"(?:(?:black|blue|brown|red|green|grey|gray|"
        r"white|yellow|pink|small|large)\s+)?"
        r"(?:document\s+folder|"
        r"school\s+bag|"
        r"handbag|"
        r"backpack|"
        r"suitcase|"
        r"laptop|"
        r"mobile\s+phone|"
        r"mobile|"
        r"phone|"
        r"wallet|"
        r"purse|"
        r"watch|"
        r"glasses|"
        r"charger|"
        r"umbrella|"
        r"jacket|"
        r"bottle|"
        r"bag)"
        r")\b"
    )

    match = re.search(
        item_pattern,
        text,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return None


# =====================================================
# 8. STRUCTURED INFORMATION EXTRACTION
# =====================================================

def extract_structured_details(text):
    """
    Extract actual values from passenger complaints.

    Returns:
        booking_reference
        train_number
        journey_date
        location
        item_description
    """

    if not isinstance(text, str):
        raise ValueError("Complaint must be text.")

    return {
        "booking_reference": extract_booking_reference(text),
        "train_number": extract_train_number(text),
        "journey_date": extract_date(text),
        "location": extract_location(text),
        "item_description": extract_item_description(text),
    }


# =====================================================
# 9. INFORMATION AVAILABILITY DETECTION
# =====================================================

def extract_complaint_details(text):
    """
    Return Boolean indicators for available information.
    """

    details = extract_structured_details(text)

    return {
        field: value is not None
        for field, value in details.items()
    }


# =====================================================
# 10. MAIN COMPLAINT ANALYSIS
# =====================================================

def analyze_complaint(complaint_text):
    """
    Classify the complaint, extract details,
    and identify missing information.
    """

    if not isinstance(complaint_text, str):
        raise ValueError("Complaint must be text.")

    if not complaint_text.strip():
        raise ValueError("Complaint cannot be empty.")

    # Step 1: Predict complaint category
    category = predict_complaint(complaint_text)

    # Step 2: Extract actual information
    structured_details = extract_structured_details(
        complaint_text
    )

    # Step 3: Detect available fields
    detected_information = {
        field: value is not None
        for field, value in structured_details.items()
    }

    # Step 4: Get required fields
    required_fields = REQUIRED_FIELDS.get(
        category,
        [],
    )

    # Step 5: Identify missing fields
    missing_information = [
        field
        for field in required_fields
        if not detected_information.get(field, False)
    ]

    # Step 6: Generate follow-up questions
    follow_up_questions = [
        FIELD_QUESTIONS[field]
        for field in missing_information
    ]

    # Step 7: Determine case status
    if missing_information:
        status = "Information Required"
    else:
        status = "Ready for Staff Review"

    return {
        "complaint": complaint_text,
        "category": category,
        "detected_information": detected_information,
        "structured_details": structured_details,
        "required_fields": required_fields,
        "missing_information": missing_information,
        "follow_up_questions": follow_up_questions,
        "status": status,
    }


# =====================================================
# 11. DISPLAY COMPLAINT ANALYSIS
# =====================================================

def display_analysis(result):

    print("\n========== COMPLAINT INTELLIGENCE ==========")

    print("\nComplaint:")
    print(result["complaint"])

    print("\nPredicted Category:")
    print(result["category"])

    print("\nDetected Information:")

    for field in result["required_fields"]:

        value = result["structured_details"].get(field)

        if value is not None:
            print(f"{field}: {value}")
        else:
            print(f"{field}: Missing")

    print("\nMissing Information:")

    if result["missing_information"]:
        for field in result["missing_information"]:
            print("-", field)
    else:
        print("No required information detected as missing.")

    print("\nSuggested Follow-up Questions:")

    if result["follow_up_questions"]:
        for question in result["follow_up_questions"]:
            print("-", question)
    else:
        print("No additional questions suggested.")

    print("\nCase Status:")
    print(result["status"])

    print("\n============================================")


# =====================================================
# 12. RUN MODULE DIRECTLY
# =====================================================

if __name__ == "__main__":

    complaint = input(
        "\nEnter a passenger complaint: "
    ).strip()

    try:
        result = analyze_complaint(complaint)

        display_analysis(result)

        print("\nStructured Details:")
        print(result["structured_details"])

    except (ValueError, FileNotFoundError) as error:
        print("\nError:", error)
