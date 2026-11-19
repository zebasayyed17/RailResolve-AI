
from src.missing_info import analyze_complaint
from src.hybrid_retriever import hybrid_search
from src.database import create_case


REFUND_SOURCE = "refund_rules.pdf"
CHARTER_SOURCE = "citizens_charter_passenger_services.pdf"


def _phrase_count(text, phrases):
    """Count distinct relevance phrases found in a passage."""
    text = (text or "").lower()
    return sum(1 for phrase in phrases if phrase in text)


def _is_general_grievance_guidance(text):
    """Return True when a passage contains actual grievance channels."""
    channel_phrases = (
        "railmadad",
        "helpline 139",
        "helpine 139",
        "web portal",
        "mobile app",
        "sms to 139",
        "complaints & suggestion book",
        "suggestion book",
        "station master",
        "cpgrams",
        "grievances may be lodged",
        "social media",
    )
    return _phrase_count(text, channel_phrases) >= 2


def _rank_refund_evidence(results, complaint):
    """Prefer refund-processing evidence and suppress lost-ticket passages."""
    complaint_lower = (complaint or "").lower()
    is_refund_status_query = any(
        phrase in complaint_lower
        for phrase in (
            "not received",
            "still waiting",
            "refund pending",
            "refund not credited",
            "amount not credited",
            "money not received",
            "refund status",
        )
    )

    processing_phrases = (
        "credited to the customer",
        "credited to same account",
        "credited back",
        "bank account",
        "tdr",
        "refund claim",
        "payment was made",
        "irctc will forward",
        "refund of fare shall be credited",
        "amount will be credited",
        "etickets@irctc.co.in",
    )
    cancellation_phrases = (
        "cancellation of e-tickets",
        "cancellation charges",
        "cancelled through internet",
        "ticket cancellation",
        "refund will be granted",
        "refund of fare",
    )

    ranked = []
    for item in results:
        passage = str(item.get("text", ""))
        passage_lower = passage.lower()

        # This passage concerns lost/misplaced paper tickets, not a delayed
        # refund after cancellation. Do not show it unless the complaint is
        # actually about a lost or misplaced ticket.
        lost_ticket_section = (
            "lost / misplaced / torn or mutilated tickets" in passage_lower
            or "duplicate ticket" in passage_lower
        )
        complaint_is_about_lost_ticket = any(
            phrase in complaint_lower
            for phrase in ("lost ticket", "misplaced ticket", "torn ticket", "duplicate ticket")
        )
        if lost_ticket_section and not complaint_is_about_lost_ticket:
            continue

        processing_score = _phrase_count(passage_lower, processing_phrases)
        cancellation_score = _phrase_count(passage_lower, cancellation_phrases)

        if is_refund_status_query:
            relevance_score = (processing_score * 3) + cancellation_score
            if processing_score == 0:
                continue
        else:
            relevance_score = (processing_score * 2) + cancellation_score
            if relevance_score == 0:
                continue

        hybrid_score = float(item.get("hybrid_score", 0.0) or 0.0)
        ranked.append((relevance_score, hybrid_score, item))

    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [row[2] for row in ranked]


def retrieve_policy_evidence(complaint, category):
    """
    Retrieve and rerank evidence using both document source and passage relevance.

    Refund complaints search the refund rules document. Other categories
    search the passenger services charter, but only passages containing
    identifiable grievance-submission channels are retained. The charter
    is not treated as proof of a category-specific resolution procedure.
    """
    category = (category or "").strip()
    complaint_text = (complaint or "").strip()
    category_key = category.lower()

    category_queries = {
        "cleanliness": "railway train station cleanliness sanitation hygiene complaint grievance",
        "ticketing": "railway ticket booking reservation ticket issue complaint grievance",
        "refund": "railway ticket cancellation refund processing credited to bank account TDR",
        "delay": "delayed train running late train delay passenger grievance reporting channels",
        "facilities": "railway station train passenger facilities drinking water toilets seating complaint",
        "lost property": "lost belongings luggage bag property on train railway station passenger grievance reporting",
        "staff assistance": "railway staff assistance staff conduct passenger complaint grievance",
        "information": "railway passenger travel information passenger services grievance guidance",
    }

    if category_key == "refund":
        source_filter = REFUND_SOURCE
        evidence_type = "Refund policy evidence"
        query = (
            category_queries["refund"]
            + " Passenger complaint: "
            + complaint_text
        )
    else:
        source_filter = CHARTER_SOURCE
        evidence_type = "General grievance submission guidance"
        query = (
            category_queries.get(
                category_key,
                "railway passenger services grievance submission guidance",
            )
            + " Passenger complaint: "
            + complaint_text
            + " RailMadad helpline 139 web portal mobile app SMS complaints "
              "suggestion book station master CPGRAMS"
        )

    results = hybrid_search(
        query=query,
        top_k=10,
        candidate_k=20,
        source_filter=source_filter,
    )

    source_matched = [
        item for item in results
        if str(item.get("source", "")).lower() == source_filter.lower()
        and str(item.get("text", "")).strip()
    ]

    if category_key == "refund":
        relevant = _rank_refund_evidence(source_matched, complaint_text)
        max_results = 3
    else:
        # In the currently indexed charter, retain only passages that contain
        # actual reporting channels (e.g. RailMadad/139/Station Master). This
        # prevents unrelated service tables such as catering deficiencies
        # from being displayed as evidence for delay or lost-property cases.
        relevant = [
            item for item in source_matched
            if _is_general_grievance_guidance(item.get("text", ""))
        ]
        max_results = 2

    # Initialize selection variables for both refund and non-refund cases.
    selected = []
    seen = set()
    seen_pages = set()

    for item in relevant:
        source = str(item.get("source", "")).lower()
        page = str(item.get("page", "")).strip()
        passage = str(item.get("text", "")).strip()

        # Remove exact duplicate passages.
        passage_key = (source, page, passage.lower())

        if passage_key in seen:
            continue

        # For refund complaints, keep at most one passage per PDF page.
        # This prevents one page from occupying all evidence slots.
        if category_key == "refund" and page and (source, page) in seen_pages:
            continue

        seen.add(passage_key)

        if category_key == "refund" and page:
            seen_pages.add((source, page))

        selected.append(item)

        if len(selected) >= max_results:
            break

    if not selected:
        evidence_type = (
            "No sufficiently relevant evidence found in the indexed policy documents"
        )
    elif category_key != "refund":
        evidence_type = "General grievance submission guidance only"

    return selected, evidence_type

def create_response_draft(analysis, evidence, evidence_type):
    """Generate a suggested response for staff review."""

    category = analysis["category"]
    missing = analysis["missing_information"]

    if missing:

        questions = "\n".join(
            f"- {question}"
            for question in analysis["follow_up_questions"]
        )

        return (
            "Dear Passenger,\n\n"
            "Thank you for contacting railway support.\n\n"
            f"Your complaint has been categorized as {category}.\n\n"
            "To help the concerned staff review your case, "
            "please provide the following information:\n\n"
            f"{questions}\n\n"
            "Once these details are received, your complaint "
            "can be reviewed further.\n\n"
            "Regards,\n"
            "Railway Grievance Support"
        )

    if not evidence:

        return (
            "Dear Passenger,\n\n"
            "Thank you for contacting railway support.\n\n"
            f"Your complaint has been categorized as {category}.\n\n"
            "The currently indexed policy documents do not "
            "contain verified supporting information for this "
            "specific issue. The concerned department should "
            "review your complaint and confirm the applicable "
            "procedure.\n\n"
            "Regards,\n"
            "Railway Grievance Support"
        )

    sources = "\n".join(
        f"- {item['source']}, Page {item['page']}"
        for item in evidence
    )

    if category == "Refund":

        return (
            "Dear Passenger,\n\n"
            "Thank you for contacting railway support.\n\n"
            "Your complaint has been categorized as Refund.\n\n"
            "The following passages were retrieved from the "
            "currently indexed refund rules document:\n\n"
            f"{sources}\n\n"
            "The applicable refund conditions must be checked "
            "against your ticket type, reservation status, "
            "cancellation timing, and current amended rules.\n\n"
            "Refund eligibility has not been determined by "
            "this prototype. The concerned staff should verify "
            "the applicable policy before responding.\n\n"
            "Regards,\n"
            "Railway Grievance Support"
        )

    return (
        "Dear Passenger,\n\n"
        "Thank you for contacting railway support.\n\n"
        f"Your complaint has been categorized as {category}.\n\n"
        "The current passenger services charter provides "
        "general grievance submission channels. Retrieved "
        "reference:\n\n"
        f"{sources}\n\n"
        "This reference does not establish the specific "
        "resolution procedure for your complaint category. "
        "The concerned railway department should review "
        "the issue and confirm the appropriate action.\n\n"
        "Regards,\n"
        "Railway Grievance Support"
    )


def save_case_to_database(complaint, analysis, evidence, response):
    """
    Save the complaint, extracted details and policy evidence.

    Train number, journey date and location are saved in their
    dedicated database fields.

    Booking reference and item description are preserved inside
    the detected_information JSON field because the current
    database schema does not have separate columns for them.
    """

    # Get structured details extracted by missing_info.py
    structured_details = analysis.get("structured_details", {})

    # Keep the existing detected-information flags.
    detected_information = dict(
        analysis.get("detected_information", {})
    )

    # Also preserve the actual extracted values.
    detected_information["structured_details"] = {
        "booking_reference": structured_details.get(
            "booking_reference"
        ),
        "train_number": structured_details.get(
            "train_number"
        ),
        "journey_date": structured_details.get(
            "journey_date"
        ),
        "location": structured_details.get(
            "location"
        ),
        "item_description": structured_details.get(
            "item_description"
        ),
    }

    # Save the complaint and all available details.
    case_id = create_case(
        complaint_text=complaint,
        category=analysis["category"],
        status=analysis["status"],
        priority="Unassessed",

        # Dedicated database fields
        train_number=structured_details.get("train_number"),
        journey_date=structured_details.get("journey_date"),
        location=structured_details.get("location"),

        # JSON field containing detection flags and extracted values
        detected_information=detected_information,

        missing_information=analysis["missing_information"],
        response_draft=response,
        evidence=evidence,
    )

    return case_id


def run_resolution_agent(complaint):
    """Coordinate classification, retrieval, drafting and storage."""

    if not isinstance(complaint, str) or not complaint.strip():
        raise ValueError("Please enter a valid complaint.")

    print("\nAnalyzing complaint...")

    # Step 1: Classify the complaint and extract information.
    analysis = analyze_complaint(complaint)

    category = analysis["category"]
    missing = analysis["missing_information"]

    # Step 2: Retrieve policy evidence only when the
    # required information is available.
    if missing:

        evidence = []
        evidence_type = "Not retrieved"

    else:

        print("Retrieving supporting policy passages...")

        evidence, evidence_type = retrieve_policy_evidence(
            complaint,
            category,
        )

    # Step 3: Generate a suggested response.
    response = create_response_draft(
        analysis,
        evidence,
        evidence_type,
    )

    # Step 4: Save the complaint and extracted information.
    print("\nSaving complaint to SQLite database...")

    case_id = save_case_to_database(
        complaint,
        analysis,
        evidence,
        response,
    )

    print("Complaint saved successfully.")

    return {
        "case_id": case_id,
        "analysis": analysis,
        "evidence": evidence,
        "evidence_type": evidence_type,
        "response_draft": response,
    }


def display_resolution(result):
    """Display the complaint analysis and generated response."""

    analysis = result["analysis"]
    evidence = result["evidence"]

    structured_details = analysis.get(
        "structured_details", {}
    )

    print("\n========== RAILRESOLVE AI ==========")

    print("\nCASE ID:")
    print(result["case_id"])

    print("\nCOMPLAINT CATEGORY:")
    print(analysis["category"])

    print("\nCASE STATUS:")
    print(analysis["status"])

    print("\nEXTRACTED INFORMATION:")

    if structured_details:

        for field, value in structured_details.items():

            if value:
                print(f"{field}: {value}")

    else:
        print("No structured details available.")

    print("\nMISSING INFORMATION:")

    if analysis["missing_information"]:

        for field in analysis["missing_information"]:
            print("-", field)

    else:
        print("No required fields detected as missing.")

    print("\nPOLICY RETRIEVAL:")

    if evidence:

        print("Evidence type:", result["evidence_type"])

        for rank, item in enumerate(evidence, start=1):

            print(f"\nReference {rank}")
            print("Source:", item["source"])
            print("Page:", item["page"])
            print("Passage:", item["text"][:700])

    else:

        print(
            "No policy retrieval performed or no supporting "
            "passages found."
        )

    print("\n========== SUGGESTED RESPONSE ==========")
    print(result["response_draft"])

    print("\n========================================")
    print("CASE SAVED IN DATABASE")
    print("STATUS: HUMAN REVIEW REQUIRED")
    print("No complaint has been submitted or resolved.")
    print("========================================")


if __name__ == "__main__":

    complaint = input(
        "\nEnter a passenger complaint: "
    ).strip()

    try:

        result = run_resolution_agent(complaint)
        display_resolution(result)

    except (ValueError, FileNotFoundError) as error:

        print("\nError:", error)
