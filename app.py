
import json
import sqlite3
import pandas as pd
import streamlit as st

from src.database import (
    init_db,
    list_cases,
    get_case,
    get_case_history,
    update_case_status,
    review_case,
    get_review_history,
)

from src.resolution_agent import (
    run_resolution_agent,
    create_response_draft,
)

from src.advanced_features import (
    CUSTOM_DEPARTMENTS,
    calculate_sla,
    build_hotspot_summary,
    build_management_insights,
    detect_duplicate_candidates,
    get_sla_targets,
    initialize_advanced_schema,
    review_duplicate_candidate,
    save_case_assignment,
    save_case_priority,
    save_sla_targets,
    sync_case_enrichment,
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="RailResolve AI",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database (does not delete existing records)
init_db()
# Create separate advanced-feature tables; existing complaint/review data is preserved.
initialize_advanced_schema()


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 32px;
        font-weight: 750;
        margin-bottom: 0;
    }

    .subtitle {
        color: #8B949E;
        font-size: 14px;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }

    .stButton button {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================

def load_cases():
    """Load saved complaint cases from SQLite."""

    records = list_cases(limit=1000)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def safe_count(dataframe, column, value):
    """Count matching records safely."""

    if dataframe.empty or column not in dataframe.columns:
        return 0

    return int((dataframe[column] == value).sum())


def show_empty_state(message):
    st.info(message)


def parse_json_field(value, default=None):
    """Safely handle JSON fields returned as strings or decoded objects."""

    if default is None:
        default = {}

    if isinstance(value, (dict, list)):
        return value

    if not value:
        return default

    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def get_case_structured_details(case):
    """Get extracted details, including values stored in the JSON field."""

    detected = parse_json_field(
        case.get("detected_information"),
        default={},
    )

    if not isinstance(detected, dict):
        detected = {}

    structured = detected.get("structured_details", {})

    if not isinstance(structured, dict):
        structured = {}

    # Dedicated database columns take precedence where populated.
    return {
        "booking_reference": (
            structured.get("booking_reference")
            or case.get("booking_reference")
        ),
        "train_number": (
            case.get("train_number")
            or structured.get("train_number")
        ),
        "journey_date": (
            case.get("journey_date")
            or structured.get("journey_date")
        ),
        "location": (
            case.get("location")
            or structured.get("location")
        ),
        "item_description": (
            structured.get("item_description")
            or case.get("item_description")
        ),
    }


def display_structured_details(details):
    """Display extracted passenger details in a consistent table."""

    labels = {
        "booking_reference": "PNR / Booking Reference",
        "train_number": "Train Number",
        "journey_date": "Journey Date",
        "location": "Coach / Berth / Platform / Station",
        "item_description": "Item Description",
    }

    rows = [
        {
            "Passenger Information": label,
            "Extracted Value": details.get(field) or "Not provided / not detected",
        }
        for field, label in labels.items()
    ]

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def is_status_only_message(value):
    """Detect status notifications mistakenly passed as a response draft."""

    text = str(value or "").strip().lower()
    if not text:
        return True

    status_prefixes = (
        "case saved successfully:",
        "complaint saved successfully:",
        "human review required.",
        "human review required:",
    )
    return text.startswith(status_prefixes) or (
        "no complaint has been submitted to indian railways" in text
        and "dear passenger" not in text
    )


def get_passenger_response_draft(result):
    """Return a passenger-facing draft, rebuilding it if a status message is stored."""

    draft = str(result.get("response_draft") or "").strip()

    # The response draft should be a message to the passenger, not a case-status
    # notification. Rebuild it from the analysis and retrieved evidence if needed.
    if is_status_only_message(draft) or not draft.lower().startswith("dear passenger"):
        try:
            draft = create_response_draft(
                result.get("analysis", {}),
                result.get("evidence", []) or [],
                result.get("evidence_type", ""),
            )
        except (KeyError, TypeError, ValueError) as error:
            return (
                "A passenger-facing response draft could not be generated. "
                "Please inspect the complaint analysis and ask staff to prepare "
                f"a response manually. Details: {error}"
            )

    return str(draft or "").strip()


# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🚆 RailResolve AI")

st.sidebar.caption(
    "Intelligent Railway Complaint Resolution"
)

st.sidebar.divider()

selected_page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Submit Complaint",
        "Case Management",
        "Grievance Analytics",
        "Advanced Intelligence",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### System Information")

st.sidebar.caption("AI-assisted grievance management")

st.sidebar.caption("Database: SQLite")

st.sidebar.caption("Retrieval: FAISS + BM25")

st.sidebar.caption("Classification: TF-IDF + Logistic Regression")

st.sidebar.divider()

st.sidebar.info(
    "Prototype application. AI-generated recommendations "
    "require human review."
)


# =====================================================
# 5. OVERVIEW DASHBOARD
# =====================================================

if selected_page == "Overview":

    st.markdown(
        '<div class="main-title">Railway Grievance Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Monitor complaints, AI insights and case progress.'
        '</div>',
        unsafe_allow_html=True,
    )

    cases_df = load_cases()

    if cases_df.empty:

        total_cases = 0
        open_cases = 0
        information_required = 0
        resolved_cases = 0

    else:

        total_cases = len(cases_df)

        open_cases = safe_count(
            cases_df,
            "status",
            "Open",
        )

        information_required = safe_count(
            cases_df,
            "status",
            "Information Required",
        )

        resolved_cases = safe_count(
            cases_df,
            "status",
            "Resolved",
        )

    # KPI cards

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Cases",
        total_cases,
    )

    col2.metric(
        "Open Cases",
        open_cases,
    )

    col3.metric(
        "Information Required",
        information_required,
    )

    col4.metric(
        "Resolved Cases",
        resolved_cases,
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Complaints by Category")

        if not cases_df.empty:

            category_counts = (
                cases_df["category"]
                .value_counts()
                .sort_values(ascending=False)
            )

            st.bar_chart(category_counts)

        else:

            show_empty_state(
                "No complaint records available."
            )

    with right:

        st.subheader("Cases by Status")

        if not cases_df.empty:

            status_counts = cases_df["status"].value_counts()

            st.bar_chart(status_counts)

        else:

            show_empty_state(
                "No case status information available."
            )

    st.divider()

    st.subheader("Recently Created Cases")

    if not cases_df.empty:

        display_columns = [
            "case_id",
            "category",
            "priority",
            "status",
            "staff_review_status",
            "created_at",
        ]

        st.dataframe(
            cases_df[display_columns].head(10),
            use_container_width=True,
            hide_index=True,
        )

    else:

        show_empty_state(
            "Submit a complaint to create your first case."
        )


# =====================================================
# 6. SUBMIT COMPLAINT
# =====================================================

elif selected_page == "Submit Complaint":

    st.title("Complaint Intelligence")

    st.caption(
        "Submit a passenger complaint for AI-assisted "
        "classification, policy retrieval and response drafting."
    )

    with st.form("complaint_submission_form"):

        complaint_text = st.text_area(
            "Enter Passenger Complaint",
            placeholder=(
                "Example: My confirmed ticket refund has not "
                "been received in my bank account."
            ),
            height=160,
        )

        analyze_button = st.form_submit_button(
            "Analyze Complaint",
            type="primary",
            use_container_width=True,
        )

    if analyze_button:

        if not complaint_text.strip():

            st.error(
                "Please enter a passenger complaint."
            )

        else:

            try:

                with st.spinner(
                    "Classifying complaint and retrieving policy evidence..."
                ):

                    result = run_resolution_agent(
                        complaint_text.strip()
                    )

                # Ensure the UI receives a passenger-facing draft. If an older
                # agent result contains only a case-status notification, rebuild
                # the draft from the analysis and retrieved policy evidence.
                result["response_draft"] = get_passenger_response_draft(result)
                st.session_state["last_result"] = result

                st.success(
                    "Complaint analyzed and saved successfully!"
                )

            except Exception as error:

                st.error(
                    f"Unable to analyze complaint: {error}"
                )

    # Display most recent analysis

    result = st.session_state.get("last_result")

    if result:

        analysis = result["analysis"]

        st.divider()

        st.subheader("AI Complaint Analysis")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Complaint Category",
            analysis["category"],
        )

        col2.metric(
            "Case Status",
            analysis["status"],
        )

        col3.metric(
            "Case ID",
            result["case_id"],
        )

        # Information detection

        st.subheader("Information Detection")

        detected = analysis["detected_information"]

        detection_rows = []

        for field in analysis["required_fields"]:

            detection_rows.append({
                "Required Information": field.replace(
                    "_", " "
                ).title(),

                "Detection": (
                    "Available"
                    if detected.get(field, False)
                    else "Missing"
                ),
            })

        if detection_rows:

            st.dataframe(
                pd.DataFrame(detection_rows),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No category-specific fields are configured."
            )

        # Extracted passenger information

        st.subheader("Extracted Passenger Details")

        structured_details = analysis.get("structured_details", {})

        if not isinstance(structured_details, dict):
            structured_details = {}

        display_structured_details(structured_details)

        # Missing details

        if analysis["follow_up_questions"]:

            st.warning(
                "Additional information is required."
            )

            for question in analysis["follow_up_questions"]:

                st.write(f"• {question}")

        else:

            st.success(
                "All prototype-required information fields were detected."
            )

        # Policy retrieval

        st.divider()

        st.subheader("Retrieved Policy Evidence")

        evidence = result["evidence"]

        if evidence:

            for rank, item in enumerate(evidence, start=1):

                title = (
                    f"Reference {rank}: "
                    f"{item['source']} — Page {item['page']}"
                )

                with st.expander(title):

                    st.write(item["text"])

                    if "hybrid_score" in item:

                        st.caption(
                            "Hybrid ranking score: "
                            f"{item['hybrid_score']:.5f}"
                        )

        else:

            st.info(
                "No policy retrieval was performed because "
                "information is missing, or no supporting passage "
                "was retrieved."
            )

        # Response draft

        st.divider()

        st.subheader("Suggested Response Draft")
        st.caption(
            "Passenger-facing draft generated from the complaint analysis "
            "and retrieved evidence. Staff review is required before use."
        )

        response_draft = get_passenger_response_draft(result)
        result["response_draft"] = response_draft

        st.text_area(
            "AI-generated passenger response",
            value=response_draft,
            height=300,
            disabled=True,
            key=f"response_{result['case_id']}",
        )

        st.success(
            f"Case saved successfully: {result['case_id']}"
        )

        st.warning(
            "Human review required. The complaint has not been "
            "submitted to Indian Railways."
        )


# =====================================================
# 7. CASE MANAGEMENT
# =====================================================

elif selected_page == "Case Management":

    st.title("Case Management")

    st.caption(
        "Search, inspect and update saved railway complaint cases."
    )

    cases_df = load_cases()

    if cases_df.empty:

        st.warning(
            "No complaint cases have been saved yet."
        )

    else:

        # -----------------------------
        # CASE FILTERS
        # -----------------------------

        col1, col2 = st.columns(2)

        status_options = [
            "All",
            *sorted(
                cases_df["status"]
                .dropna()
                .unique()
            ),
        ]

        category_options = [
            "All",
            *sorted(
                cases_df["category"]
                .dropna()
                .unique()
            ),
        ]

        with col1:

            selected_status = st.selectbox(
                "Filter by Status",
                status_options,
            )

        with col2:

            selected_category = st.selectbox(
                "Filter by Category",
                category_options,
            )

        filtered = cases_df.copy()

        if selected_status != "All":

            filtered = filtered[
                filtered["status"] == selected_status
            ]

        if selected_category != "All":

            filtered = filtered[
                filtered["category"] == selected_category
            ]

        # -----------------------------
        # COMPLAINT REGISTER
        # -----------------------------

        st.subheader("Complaint Register")

        display_columns = [
            "case_id",
            "category",
            "priority",
            "status",
            "staff_review_status",
            "created_at",
        ]

        st.dataframe(
            filtered[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # -----------------------------
        # CASE INSPECTION
        # -----------------------------

        st.subheader("Inspect a Case")

        case_ids = filtered["case_id"].tolist()

        if case_ids:

            selected_case_id = st.selectbox(
                "Select Case ID",
                case_ids,
            )

            case = get_case(selected_case_id)

            if case:

                st.markdown("### Complaint Details")

                st.write(case["complaint_text"])

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Category",
                    case["category"],
                )

                col2.metric(
                    "Priority",
                    case["priority"],
                )

                col3.metric(
                    "Status",
                    case["status"],
                )

                # Extracted passenger information

                st.subheader("Extracted Passenger Details")

                structured_details = get_case_structured_details(case)
                display_structured_details(structured_details)

                # Missing information

                st.subheader("Missing Information")

                missing = case["missing_information"]

                if missing:

                    st.warning(
                        ", ".join(missing)
                    )

                else:

                    st.success(
                        "No prototype-required fields detected as missing."
                    )

                # Response draft

                with st.expander(
                    "View Suggested Response",
                    expanded=False,
                ):

                    saved_draft = str(case.get("response_draft") or "").strip()
                    if is_status_only_message(saved_draft) or not saved_draft.lower().startswith("dear passenger"):
                        st.warning(
                            "This case does not contain a valid passenger-facing "
                            "response draft. The case record is preserved; staff "
                            "should regenerate or prepare the response before review."
                        )
                    else:
                        st.text_area(
                            "Saved passenger response draft",
                            value=saved_draft,
                            height=240,
                            disabled=True,
                            key=f"saved_response_{selected_case_id}",
                        )

                # Retrieved evidence

                with st.expander(
                    "View Policy Evidence",
                    expanded=False,
                ):

                    evidence = case["evidence"]

                    if evidence:

                        for rank, item in enumerate(
                            evidence,
                            start=1,
                        ):

                            st.markdown(
                                f"**Reference {rank}: "
                                f"{item['source']} — Page {item['page']}**"
                            )

                            st.write(item["text"])

                            st.divider()

                    else:

                        st.info(
                            "No policy evidence was stored for this case."
                        )

                # -----------------------------
                # UPDATE CASE STATUS
                # -----------------------------

                st.divider()

                st.subheader("Update Case Status")

                available_statuses = [
                    "Open",
                    "Information Required",
                    "Under Review",
                    "In Progress",
                    "Resolved",
                    "Closed",
                ]

                current_status = case["status"]

                if current_status not in available_statuses:

                    available_statuses.append(current_status)

                with st.form(
                    key=f"status_form_{selected_case_id}"
                ):

                    new_status = st.selectbox(
                        "New Status",
                        available_statuses,
                        index=available_statuses.index(
                            current_status
                        ),
                    )

                    status_reviewer = st.text_input(
                        "Staff Name",
                        value="Demo Staff",
                    )

                    status_notes = st.text_area(
                        "Staff Notes",
                        placeholder=(
                            "Enter reason for changing the case status."
                        ),
                    )

                    status_submitted = st.form_submit_button(
                        "Save Status Update",
                        type="primary",
                    )

                if status_submitted:

                    try:

                        update_case_status(
                            case_id=selected_case_id,
                            new_status=new_status,
                            changed_by=status_reviewer,
                            notes=status_notes,
                        )

                        st.success(
                            "Case status updated successfully!"
                        )

                        st.rerun()

                    except (ValueError, sqlite3.Error) as error:

                        st.error(str(error))

                # -----------------------------
                # CASE STATUS HISTORY
                # -----------------------------

                st.divider()

                st.subheader("Case History")

                history = get_case_history(
                    selected_case_id
                )

                if history:

                    st.dataframe(
                        pd.DataFrame(history),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.info(
                        "No status history records available."
                    )

                # =================================================
                # STAFF REVIEW WORKBENCH
                # =================================================

                st.divider()

                st.subheader("Staff Review Workbench")

                st.caption(
                    "Review the AI-generated response draft "
                    "before it is used."
                )

                current_review = case.get(
                    "staff_review_status",
                    "Pending",
                )

                if current_review == "Approved":

                    st.success(
                        "Latest response draft decision: Approved"
                    )

                elif current_review == "Rejected":

                    st.error(
                        "Latest response draft decision: Rejected"
                    )

                else:

                    st.info(
                        "This response draft is awaiting staff review."
                    )

                if not case.get("response_draft"):

                    st.warning(
                        "This case does not contain a response draft. "
                        "There is nothing to approve or reject."
                    )

                else:

                    with st.form(
                        key=f"review_form_{selected_case_id}"
                    ):

                        decision = st.radio(
                            "Staff Decision",
                            ["Approved", "Rejected"],
                            horizontal=True,
                        )

                        reviewer = st.text_input(
                            "Reviewer Name",
                            value="Demo Staff",
                        )

                        review_comments = st.text_area(
                            "Review Comments",
                            placeholder=(
                                "Enter approval comments or explain "
                                "why the response draft is being rejected."
                            ),
                        )

                        review_submitted = st.form_submit_button(
                            "Save Review Decision",
                            type="primary",
                            use_container_width=True,
                        )

                    if review_submitted:

                        try:

                            review_case(
                                case_id=selected_case_id,
                                decision=decision,
                                reviewed_by=reviewer,
                                comments=review_comments,
                            )

                            st.success(
                                f"Response draft {decision.lower()} successfully!"
                            )

                            st.rerun()

                        except (ValueError, sqlite3.Error) as error:

                            st.error(str(error))

                # -----------------------------
                # REVIEW DECISION HISTORY
                # -----------------------------

                st.subheader("Review Decision History")

                review_records = get_review_history(
                    selected_case_id
                )

                if review_records:

                    review_df = pd.DataFrame(
                        review_records
                    )

                    # Do not display the entire saved draft
                    # inside the history table.
                    review_df = review_df.drop(
                        columns=[
                            "response_draft_snapshot"
                        ],
                        errors="ignore",
                    )

                    st.dataframe(
                        review_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.info(
                        "This response draft has not yet "
                        "been reviewed by staff."
                    )

        else:

            st.info(
                "No cases match the selected filters."
            )


# =====================================================
# 8. GRIEVANCE ANALYTICS
# =====================================================

elif selected_page == "Grievance Analytics":

    st.title("Grievance Analytics")

    st.caption(
        "SQL-backed analysis of saved railway complaint cases."
    )

    cases_df = load_cases()

    if cases_df.empty:

        st.warning(
            "No cases available for analysis."
        )

    else:

        total_cases = len(cases_df)

        resolved_cases = safe_count(
            cases_df,
            "status",
            "Resolved",
        )

        open_cases = safe_count(
            cases_df,
            "status",
            "Open",
        )

        information_required = safe_count(
            cases_df,
            "status",
            "Information Required",
        )

        resolution_rate = (
            resolved_cases / total_cases * 100
            if total_cases > 0
            else 0
        )

        # -----------------------------
        # ANALYTICS KPIs
        # -----------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Complaints",
            total_cases,
        )

        col2.metric(
            "Open Cases",
            open_cases,
        )

        col3.metric(
            "Information Required",
            information_required,
        )

        col4.metric(
            "Resolved Percentage",
            f"{resolution_rate:.1f}%",
        )

        st.divider()

        # -----------------------------
        # CATEGORY AND STATUS CHARTS
        # -----------------------------

        left, right = st.columns(2)

        with left:

            st.subheader("Category Distribution")

            category_data = (
                cases_df["category"]
                .value_counts()
            )

            st.bar_chart(category_data)

        with right:

            st.subheader("Status Distribution")

            status_data = (
                cases_df["status"]
                .value_counts()
            )

            st.bar_chart(status_data)

        st.divider()

        # -----------------------------
        # PRIORITY DISTRIBUTION
        # -----------------------------

        st.subheader("Priority Distribution")

        priority_data = (
            cases_df["priority"]
            .value_counts()
        )

        st.bar_chart(priority_data)

        st.divider()

        # -----------------------------
        # STAFF REVIEW ANALYTICS
        # -----------------------------

        st.subheader("Staff Review Summary")

        review_data = (
            cases_df["staff_review_status"]
            .fillna("Pending")
            .value_counts()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Pending Reviews",
            int(review_data.get("Pending", 0)),
        )

        col2.metric(
            "Approved Drafts",
            int(review_data.get("Approved", 0)),
        )

        col3.metric(
            "Rejected Drafts",
            int(review_data.get("Rejected", 0)),
        )

        st.bar_chart(review_data)

        st.divider()

        # -----------------------------
        # MONTHLY CASE TREND
        # -----------------------------

        st.subheader("Complaint Creation Trend")

        trend_df = cases_df.copy()

        trend_df["created_at"] = pd.to_datetime(
            trend_df["created_at"],
            errors="coerce",
            utc=True,
        )

        trend_df = trend_df.dropna(
            subset=["created_at"]
        )

        if not trend_df.empty:

            trend_df["month"] = (
                trend_df["created_at"]
                .dt.strftime("%Y-%m")
            )

            monthly_counts = (
                trend_df.groupby("month")
                .size()
                .sort_index()
            )

            st.line_chart(monthly_counts)

        else:

            st.info(
                "No valid creation dates are available."
            )

        st.divider()

        # -----------------------------
        # DETAILED CASE DATA
        # -----------------------------

        st.subheader("Detailed Case Data")

        display_columns = [
            "case_id",
            "category",
            "priority",
            "status",
            "staff_review_status",
            "created_at",
            "updated_at",
        ]

        st.dataframe(
            cases_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Analytics reflect the records currently stored in "
            "the prototype database. They are not live Indian "
            "Railways operational statistics."
        )


# =====================================================
# 9. ADVANCED INTELLIGENCE
# =====================================================

elif selected_page == "Advanced Intelligence":

    st.title("Advanced Grievance Intelligence")
    st.caption(
        "Department-facing recommendations for prioritisation, duplicate review, "
        "SLA monitoring, routing, location patterns and management insights."
    )
    st.info(
        "Priority, department routing and SLA targets are configurable prototype "
        "recommendations. They are not official Indian Railways rules or deadlines. "
        "Potential duplicates are never merged automatically."
    )

    cases_df = load_cases()

    if cases_df.empty:
        st.warning("No cases are available yet. Submit complaints to populate this workspace.")
    else:
        case_records = cases_df.fillna("").to_dict(orient="records")

        with st.spinner("Updating case recommendations and checking for potential duplicates..."):
            enrichment_map = sync_case_enrichment(case_records)
            duplicate_rows = detect_duplicate_candidates(case_records)

        # Build reusable tables from the existing case records and the separate
        # advanced-feature extension tables.
        intelligence_rows = []
        sla_rows = []
        for case in case_records:
            case_id = str(case.get("case_id", ""))
            enrichment = enrichment_map.get(case_id, {})
            effective_priority = (
                enrichment.get("confirmed_priority")
                or enrichment.get("suggested_priority")
                or "Unassessed"
            )
            reasons = parse_json_field(enrichment.get("priority_reasons"), default=[])
            intelligence_rows.append({
                "Case ID": case_id,
                "Category": case.get("category", ""),
                "Status": case.get("status", ""),
                "Suggested Priority": enrichment.get("suggested_priority", "Unassessed"),
                "Staff-Confirmed Priority": enrichment.get("confirmed_priority") or "Not set",
                "Effective Priority": effective_priority,
                "Priority Score": enrichment.get("priority_score", 0),
                "Priority Reasons": "; ".join(reasons) if isinstance(reasons, list) else str(reasons),
                "Suggested Department": enrichment.get("suggested_department", ""),
                "Assigned Department": enrichment.get("assigned_department") or "Not assigned",
                "Assigned Staff": enrichment.get("assigned_to") or "Not assigned",
            })
            sla = calculate_sla(case, enrichment)
            sla_rows.append({
                "Case ID": case_id,
                "Category": case.get("category", ""),
                "Case Status": case.get("status", ""),
                "Priority": effective_priority,
                "SLA Status": sla["sla_status"],
                "Age (hours)": sla["age_hours"],
                "Hours Remaining": sla["hours_remaining"],
                "Demo Due At (UTC)": sla["due_at"],
            })

        intelligence_df = pd.DataFrame(intelligence_rows)
        sla_df = pd.DataFrame(sla_rows)

        tab_priority, tab_duplicates, tab_sla, tab_assignment, tab_hotspots, tab_insights = st.tabs([
            "1. Prioritisation",
            "2. Duplicate Detection",
            "3. SLA & Escalation",
            "4. Department Assignment",
            "5. Location Hotspots",
            "6. Management Insights",
        ])

        # -------------------------------------------------
        # FEATURE 1: INTELLIGENT PRIORITISATION
        # -------------------------------------------------
        with tab_priority:
            st.subheader("Explainable Complaint Prioritisation")
            st.caption("Rule-based recommendation using configured urgency and passenger-impact phrases.")

            priority_counts = intelligence_df["Effective Priority"].value_counts()
            p1, p2, p3 = st.columns(3)
            p1.metric("High Priority", int(priority_counts.get("High", 0)))
            p2.metric("Medium Priority", int(priority_counts.get("Medium", 0)))
            p3.metric("Low Priority", int(priority_counts.get("Low", 0)))

            st.dataframe(
                intelligence_df[[
                    "Case ID", "Category", "Suggested Priority",
                    "Staff-Confirmed Priority", "Priority Score", "Priority Reasons"
                ]],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Confirm or override a priority")
            priority_case_id = st.selectbox(
                "Select case for priority review",
                options=[row["Case ID"] for row in intelligence_rows],
                key="advanced_priority_case",
            )
            selected_priority_row = next(row for row in intelligence_rows if row["Case ID"] == priority_case_id)
            current_priority = selected_priority_row["Staff-Confirmed Priority"]
            suggested_priority = selected_priority_row["Suggested Priority"]
            default_priority = current_priority if current_priority in ["High", "Medium", "Low"] else suggested_priority
            with st.form("advanced_priority_form"):
                st.write(f"AI/rules suggestion: **{suggested_priority}**")
                confirmed_priority = st.selectbox(
                    "Staff-confirmed priority",
                    ["High", "Medium", "Low"],
                    index=["High", "Medium", "Low"].index(default_priority),
                )
                priority_submitted = st.form_submit_button("Save Priority Decision", type="primary")
            if priority_submitted:
                try:
                    save_case_priority(priority_case_id, confirmed_priority)
                    st.success(f"Priority for {priority_case_id} saved as {confirmed_priority}.")
                    st.rerun()
                except (ValueError, sqlite3.Error) as error:
                    st.error(f"Unable to save priority: {error}")

        # -------------------------------------------------
        # FEATURE 2: DUPLICATE COMPLAINT DETECTION
        # -------------------------------------------------
        with tab_duplicates:
            st.subheader("Potential Duplicate Complaint Detection")
            st.caption(
                "Text similarity and matching train/date/category context identify cases for human comparison. "
                "Similarity is a screening signal, not proof that two complaints are duplicates."
            )
            if duplicate_rows:
                dup_df = pd.DataFrame(duplicate_rows)
                visible_dup_columns = [
                    "id", "case_id", "possible_duplicate_case_id", "similarity",
                    "reason", "review_status", "reviewed_by", "review_notes"
                ]
                st.dataframe(
                    dup_df[[column for column in visible_dup_columns if column in dup_df.columns]],
                    use_container_width=True,
                    hide_index=True,
                )
                pending_dup_count = sum(1 for row in duplicate_rows if row.get("review_status") == "Pending")
                st.metric("Pairs awaiting human review", pending_dup_count)

                candidate_options = {
                    f"#{row['id']} | {row['case_id']} ↔ {row['possible_duplicate_case_id']} | {float(row['similarity']):.0%}": int(row["id"])
                    for row in duplicate_rows
                }
                selected_candidate_label = st.selectbox(
                    "Select a candidate pair to review",
                    list(candidate_options.keys()),
                    key="duplicate_candidate_select",
                )
                selected_candidate_id = candidate_options[selected_candidate_label]
                selected_candidate = next(row for row in duplicate_rows if int(row["id"]) == selected_candidate_id)
                with st.expander("Compare the two complaints", expanded=True):
                    first_case = next((case for case in case_records if case.get("case_id") == selected_candidate["case_id"]), {})
                    second_case = next((case for case in case_records if case.get("case_id") == selected_candidate["possible_duplicate_case_id"]), {})
                    compare_left, compare_right = st.columns(2)
                    with compare_left:
                        st.markdown(f"**{selected_candidate['case_id']}**")
                        st.write(first_case.get("complaint_text", "Complaint text unavailable."))
                    with compare_right:
                        st.markdown(f"**{selected_candidate['possible_duplicate_case_id']}**")
                        st.write(second_case.get("complaint_text", "Complaint text unavailable."))
                with st.form("duplicate_review_form"):
                    duplicate_decision = st.selectbox(
                        "Review decision",
                        ["Pending", "Confirmed Related", "Not Duplicate", "Needs Investigation"],
                        index=["Pending", "Confirmed Related", "Not Duplicate", "Needs Investigation"].index(selected_candidate.get("review_status", "Pending")),
                    )
                    duplicate_reviewer = st.text_input("Reviewer", value="Demo Staff")
                    duplicate_notes = st.text_area("Review notes", placeholder="Explain why the pair is related, not duplicate, or needs investigation.")
                    duplicate_submitted = st.form_submit_button("Save Duplicate Review", type="primary")
                if duplicate_submitted:
                    try:
                        review_duplicate_candidate(
                            selected_candidate_id,
                            duplicate_decision,
                            duplicate_reviewer,
                            duplicate_notes,
                        )
                        st.success("Duplicate review decision saved.")
                        st.rerun()
                    except (ValueError, sqlite3.Error) as error:
                        st.error(str(error))
            else:
                st.success("No potential duplicate pairs were detected among the currently stored cases.")

        # -------------------------------------------------
        # FEATURE 3: SLA AND ESCALATION TRACKING
        # -------------------------------------------------
        with tab_sla:
            st.subheader("Configurable SLA & Escalation Monitor")
            st.warning(
                "These are demonstration targets only. Set targets approved by your project scenario or organisation; "
                "they are not published Indian Railways response deadlines."
            )
            sla_targets = get_sla_targets()
            with st.form("sla_targets_form"):
                s1, s2, s3 = st.columns(3)
                high_hours = s1.number_input("High priority target (hours)", min_value=1, max_value=2160, value=int(sla_targets["High"]))
                medium_hours = s2.number_input("Medium priority target (hours)", min_value=1, max_value=2160, value=int(sla_targets["Medium"]))
                low_hours = s3.number_input("Low priority target (hours)", min_value=1, max_value=2160, value=int(sla_targets["Low"]))
                sla_settings_submitted = st.form_submit_button("Save Demo SLA Targets", type="primary")
            if sla_settings_submitted:
                try:
                    save_sla_targets({"High": high_hours, "Medium": medium_hours, "Low": low_hours})
                    st.success("SLA demonstration targets saved. Case due times will be recalculated on refresh.")
                    st.rerun()
                except (ValueError, sqlite3.Error) as error:
                    st.error(str(error))

            overdue_count = int(sla_df["SLA Status"].eq("Overdue").sum())
            on_track_count = int(sla_df["SLA Status"].eq("On Track").sum())
            stopped_count = int(sla_df["SLA Status"].str.startswith("Stopped").sum())
            s1, s2, s3 = st.columns(3)
            s1.metric("Overdue (demo target)", overdue_count)
            s2.metric("On Track", on_track_count)
            s3.metric("Closed / Resolved", stopped_count)
            st.dataframe(sla_df, use_container_width=True, hide_index=True)
            st.caption("Overdue means the configured demonstration target has elapsed since case creation. It does not imply an official breach.")

        # -------------------------------------------------
        # FEATURE 4: DEPARTMENT-WISE ASSIGNMENT
        # -------------------------------------------------
        with tab_assignment:
            st.subheader("Suggested Department Routing")
            st.caption("Routing suggestions are category-based starting points. Staff can change the department and record an assignee.")
            st.dataframe(
                intelligence_df[["Case ID", "Category", "Suggested Department", "Assigned Department", "Assigned Staff"]],
                use_container_width=True,
                hide_index=True,
            )
            assignment_case_id = st.selectbox(
                "Select case to assign",
                options=[row["Case ID"] for row in intelligence_rows],
                key="assignment_case_select",
            )
            assignment_row = next(row for row in intelligence_rows if row["Case ID"] == assignment_case_id)
            current_department = assignment_row["Assigned Department"]
            suggested_department_value = assignment_row["Suggested Department"].replace(" (suggested)", "")
            department_options = list(CUSTOM_DEPARTMENTS)
            if current_department not in department_options and current_department != "Not assigned":
                department_options.append(current_department)
            default_department = current_department if current_department in department_options else (
                suggested_department_value if suggested_department_value in department_options else "Other / Manual Routing"
            )
            with st.form("department_assignment_form"):
                department_choice = st.selectbox(
                    "Department / handling team",
                    department_options,
                    index=department_options.index(default_department),
                )
                assignee = st.text_input("Assigned staff member (optional)", value="")
                assignment_submitted = st.form_submit_button("Save Department Assignment", type="primary")
            if assignment_submitted:
                try:
                    save_case_assignment(assignment_case_id, department_choice, assignee)
                    st.success(f"Case {assignment_case_id} assigned to {department_choice}.")
                    st.rerun()
                except (ValueError, sqlite3.Error) as error:
                    st.error(str(error))

        # -------------------------------------------------
        # FEATURE 5: LOCATION HOTSPOT INTELLIGENCE
        # -------------------------------------------------
        with tab_hotspots:
            st.subheader("Reported-Location Hotspot Intelligence")
            st.caption(
                "Groups cases by station-like place names extracted from complaint text, or by stored location details. "
                "These are reported-location counts, not a geographic heatmap or verified station statistics."
            )
            hotspot_df = build_hotspot_summary(case_records)
            if hotspot_df.empty:
                st.info("No usable location details have been extracted from the stored complaints yet.")
            else:
                h1, h2 = st.columns([1.2, 1])
                with h1:
                    st.bar_chart(hotspot_df.set_index("Reported Location")["Complaints"])
                with h2:
                    st.dataframe(hotspot_df, use_container_width=True, hide_index=True)

        # -------------------------------------------------
        # FEATURE 6: MANAGEMENT INSIGHTS
        # -------------------------------------------------
        with tab_insights:
            st.subheader("Management Insights")
            insights = build_management_insights(case_records, enrichment_map, duplicate_rows)
            for insight in insights:
                st.markdown(f"- {insight}")
            st.divider()
            st.caption(
                "Insights are calculated from the current SQLite prototype records. "
                "They should not be interpreted as live railway-wide operational data."
            )


# =====================================================
# 10. APPLICATION FOOTER
# =====================================================

st.divider()

st.caption(
    "RailResolve AI | Intelligent Railway Complaint "
    "Resolution Prototype | AI-generated recommendations "
    "require human review."
)
