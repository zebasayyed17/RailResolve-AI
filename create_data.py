
from pathlib import Path
import pandas as pd

# Create data directory
Path("data").mkdir(parents=True, exist_ok=True)

# 15 unique synthetic complaints per category
complaints_by_category = {

    "Cleanliness": [
        "The floor of coach B2 was covered with food waste.",
        "The toilets were dirty throughout the journey.",
        "Garbage bins in the passenger compartment were overflowing.",
        "The seats in my reserved coach had visible stains.",
        "There was an unpleasant smell inside the sleeper coach.",
        "The cleaning staff did not clean our compartment.",
        "The washbasin area was extremely dirty.",
        "Food wrappers were scattered under the passenger seats.",
        "The coach windows had not been cleaned properly.",
        "The dustbins were not emptied during the journey.",
        "The railway coach was poorly maintained and unhygienic.",
        "The washroom floor was wet and covered in dirt.",
        "There were insects around the garbage collection area.",
        "The passenger compartment had not been cleaned before boarding.",
        "The toilets were overflowing and needed immediate cleaning."
    ],

    "Ticketing": [
        "My payment was deducted but the ticket booking failed.",
        "I received a payment confirmation but no PNR number.",
        "My e-ticket is not appearing in my booking history.",
        "The website generated two tickets for one payment.",
        "I entered the wrong passenger details during booking.",
        "I cannot download the ticket after successful payment.",
        "My booking transaction is showing as pending.",
        "The ticket confirmation message has not arrived.",
        "I am unable to book a ticket through the railway website.",
        "The passenger name printed on my ticket is incorrect.",
        "My reservation status is not updating on the portal.",
        "The payment gateway showed success but the ticket was not issued.",
        "I cannot access my ticket using my booking reference.",
        "The booking website deducted money twice for one transaction.",
        "My ticket details are missing from my registered account."
    ],

    "Refund": [
        "I cancelled my confirmed ticket but have not received the refund.",
        "The cancellation amount deducted from my ticket seems incorrect.",
        "My refund request is still pending after ticket cancellation.",
        "I submitted a TDR and need an update about its status.",
        "The railway has cancelled my train and I need refund information.",
        "My bank account has not received the eligible ticket refund.",
        "I cancelled a waitlisted ticket and need refund details.",
        "The refund amount credited is lower than I expected.",
        "I need clarification about refund deductions on my reservation.",
        "My online refund request has not been processed.",
        "I want to know the refund procedure for a cancelled train.",
        "My refund claim was submitted but no status is available.",
        "I need information about cancellation charges for my ticket.",
        "My TDR refund has not been credited to my account.",
        "I received a refund notification but the amount is not showing."
    ],

    "Delay": [
        "My train has been delayed by more than two hours.",
        "The train has not arrived at the departure station.",
        "I missed my connecting train because of a delay.",
        "The arrival time displayed on the board keeps changing.",
        "There has been no update about the delayed express train.",
        "My train is running behind its scheduled timetable.",
        "The train stopped at a station for an unusually long time.",
        "I need the latest arrival estimate for my delayed train.",
        "The departure was postponed without a clear announcement.",
        "The train is running several hours late today.",
        "I was unable to reach my destination on the scheduled time.",
        "The railway announcement mentioned an unexpected delay.",
        "My connecting journey was affected by the delayed service.",
        "The live running status does not show the current train position.",
        "Passengers have been waiting because the train has not arrived."
    ],

    "Facilities": [
        "The air conditioning in coach A1 stopped working.",
        "There is no drinking water available in the coach.",
        "The charging point next to my seat is not functioning.",
        "The ceiling fans in the compartment are not working.",
        "The lights inside our coach are not switching on.",
        "The washroom water supply has stopped during the journey.",
        "The coach doors are difficult to open and close.",
        "There is no working mobile charging socket near my berth.",
        "The compartment is extremely hot because the AC is not working.",
        "The water tap in the toilet is broken.",
        "The wheelchair assistance facility was not available.",
        "The passenger information display inside the coach is not working.",
        "The berth reading light has stopped functioning.",
        "The air ventilation system in my coach is not operating.",
        "The drinking water dispenser at the station is out of service."
    ],

    "Lost Property": [
        "I left my blue backpack on the train after getting off.",
        "My mobile phone is missing from my reserved seat.",
        "I forgot my wallet in the sleeper coach.",
        "I cannot find the suitcase I carried during the journey.",
        "I left a laptop bag in the overhead luggage rack.",
        "A black handbag was left behind in my compartment.",
        "I misplaced my glasses while travelling by train.",
        "My document folder was left near the window seat.",
        "I forgot a shopping bag underneath my berth.",
        "I may have left my wristwatch in the railway coach.",
        "My travel bag is missing after I left the station.",
        "I accidentally left my phone charger near my seat.",
        "A small suitcase was left inside the train after arrival.",
        "I lost my purse while travelling and need assistance.",
        "I need help reporting a missing item from my journey."
    ],

    "Staff Assistance": [
        "The coach attendant did not respond to my assistance request.",
        "A railway employee spoke rudely when I asked for help.",
        "The ticket examiner did not address my reservation concern.",
        "I could not get assistance from the station staff.",
        "The onboard staff ignored my request for medical assistance.",
        "A staff member provided incorrect information about my coach.",
        "The train attendant refused to help with my luggage.",
        "I experienced unprofessional behaviour from railway personnel.",
        "The station employee did not guide me to the correct platform.",
        "I requested help several times but received no response.",
        "The staff member was unavailable when I needed assistance.",
        "I want to report misconduct by a railway employee.",
        "The coach supervisor did not address the passenger complaint.",
        "The railway staff did not assist an elderly passenger.",
        "My request for help from the train staff remains unresolved."
    ],

    "Information": [
        "The station display shows a different platform from the announcement.",
        "I need information about the platform for my arriving train.",
        "The coach position displayed at the station appears incorrect.",
        "I cannot find the scheduled departure time for my train.",
        "The train route information is not available on the display.",
        "I need clarification about the train's boarding station.",
        "The station announcement did not clearly mention the train number.",
        "I am unable to locate the correct coach position.",
        "The train timetable on the website appears outdated.",
        "I need information about the stoppages on this route.",
        "The platform number was changed but no announcement was made.",
        "I want to know where the train will terminate.",
        "The station information board is showing incorrect details.",
        "I cannot find the arrival platform information online.",
        "The passenger information system is not displaying coach details."
    ]
}

# Separate evaluation complaints.
# These descriptions must not be included in the training CSV.

evaluation_by_category = {
    "Cleanliness": [
        "There is a strong odour coming from the unclean passenger toilet.",
        "The railway staff have not removed the accumulated waste from our coach."
    ],
    "Ticketing": [
        "My card was charged successfully but the reservation was never created.",
        "I am unable to retrieve my electronic ticket after completing the booking."
    ],
    "Refund": [
        "The refund for my cancelled reservation has not reached my original payment account.",
        "I need to understand the deduction applied to my cancelled railway ticket."
    ],
    "Delay": [
        "My scheduled train arrival is overdue and I need an updated estimate.",
        "A late-running railway service caused me to miss another connection."
    ],
    "Facilities": [
        "The electrical outlet provided beside my berth has stopped supplying power.",
        "There is no functioning cooling system inside the passenger compartment."
    ],
    "Lost Property": [
        "I accidentally left my brown suitcase in the train compartment.",
        "My handbag appears to be missing following my railway journey."
    ],
    "Staff Assistance": [
        "The onboard employee did not help when I reported a passenger issue.",
        "I want to register a complaint about the behaviour of a railway worker."
    ],
    "Information": [
        "The station's electronic board is displaying an incorrect boarding platform.",
        "I need the latest information about the location of my train's coach."
    ]
}

# --------------------------------
# Create the complaint dataset
# --------------------------------

records = []

base_date = pd.Timestamp("2026-10-01")

complaint_id = 1

for category, messages in complaints_by_category.items():

    for message in messages:

        records.append({
            "complaint_id": f"RR{complaint_id:04d}",
            "complaint_text": message,
            "category": category,
            "priority": "Unassessed",
            "status": "Open",
            "train_number": "",
            "journey_date": "",
            "created_at": (
                base_date
                + pd.Timedelta(days=(complaint_id - 1) % 30)
            ).date().isoformat(),
            "resolved_at": "",
            "data_source": "synthetic"
        })

        complaint_id += 1

df = pd.DataFrame(records)

# Shuffle records to avoid category-based row ordering.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/complaints.csv", index=False)

# --------------------------------
# Create separate evaluation data
# --------------------------------

evaluation_records = []

evaluation_id = 1

for category, messages in evaluation_by_category.items():

    for message in messages:

        evaluation_records.append({
            "case_id": f"CL{evaluation_id:03d}",
            "complaint_text": message,
            "expected_category": category,
            "data_source": "synthetic"
        })

        evaluation_id += 1

evaluation_df = pd.DataFrame(evaluation_records)

evaluation_df.to_csv(
    "data/classification_evaluation.csv",
    index=False
)

# --------------------------------
# Final verification
# --------------------------------

print("\n========== DATASET CREATED ==========")

print("Training records:", len(df))
print("Unique complaint texts:", df["complaint_text"].nunique())
print("Training categories:", df["category"].nunique())

print("\nTraining records by category:")
print(df["category"].value_counts().sort_index())

print("\nEvaluation records:", len(evaluation_df))
print(
    "Unique evaluation texts:",
    evaluation_df["complaint_text"].nunique()
)

overlap = set(df["complaint_text"]) & set(
    evaluation_df["complaint_text"]
)

print("Training/evaluation text overlap:", len(overlap))

print("\nFiles created:")
print("1. data/complaints.csv")
print("2. data/classification_evaluation.csv")

print("=====================================")
