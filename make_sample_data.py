"""Generate sample HOUSING SCENARIO XLSX files for demo/testing.

These are synthetic comments in the spirit of the CoMap housing feedback —
they let you exercise the app without the real export. Replace with the
real HOUSING SCENARIO 1/2/3.xlsx files for actual analysis.
"""
import pandas as pd
import random

random.seed(42)

s1 = [
    ("Housing does not belong inside a public park. Keep this land as open space.", "Disapprove"),
    ("We voted for parkland under Measure LC, not more development.", "Disapprove"),
    ("No housing here. Santa Monica has plenty of commercial corridors for housing.", "Disapprove"),
    ("Please keep the entire site as park. Housing should go elsewhere in the city.", "Disapprove"),
    ("Affordable housing is desperately needed — some housing on the edge could work.", "Approve"),
    ("I support a small amount of affordable housing if it doesn't reduce park space.", "Approve"),
    ("Housing along the boundary streets makes sense if the interior stays green.", "Approve"),
    ("Absolutely no residential development on this land.", "Disapprove"),
    ("The city needs housing but not at the expense of our one chance at a great park.", "Disapprove"),
    ("Traffic is already terrible. Housing here would make Bundy and Centinela worse.", "Disapprove"),
    ("Where will the infrastructure come from to support housing on this site?", "None"),
    ("Would supportive housing be considered, or only market rate?", "None"),
    ("Keep it green. Housing anywhere on the site betrays the community vision.", "Disapprove"),
    ("Mixed feelings — housing crisis is real but this park opportunity is unique.", "None"),
    ("Affordable units near the future park would be wonderful for working families.", "Approve"),
    ("This scenario puts too much housing in the middle of the site.", "Disapprove"),
    ("Housing should be located near transit on Lincoln, not inside the park.", "Disapprove"),
    ("If Measure LC requires a vote for housing, let the voters decide.", "None"),
    ("I want gardens and trees, not apartment blocks.", "Disapprove"),
    ("Some senior housing overlooking the park would be lovely.", "Approve"),
    ("Do not repeat the mistakes of overdevelopment. Open space first.", "Disapprove"),
    ("Park space and habitat should take priority over any housing.", "Disapprove"),
    ("A modest affordable housing component could fund park maintenance.", "Approve"),
    ("Housing plus a park can coexist if designed carefully at the edges.", "Approve"),
    ("The noise and traffic from construction would hurt nearby neighborhoods.", "Disapprove"),
    ("What does Measure LC actually allow here? Please clarify before proposing housing.", "None"),
]

s2 = [
    ("Strongly opposed to any housing on the airport site — park, full stop.", "Disapprove"),
    ("If housing is added I would completely lose support for this plan.", "Disapprove"),
    ("This scenario's housing footprint is far too large.", "Disapprove"),
    ("Put the housing along the perimeter and keep the center wild.", "Approve"),
    ("Affordable housing for teachers and nurses would be a public good here.", "Approve"),
    ("Housing near the new park edge with ground-floor cafes could be great.", "Approve"),
    ("We need open space more than we need more luxury apartments.", "Disapprove"),
    ("Measure LC exists for a reason. Respect the voters.", "Disapprove"),
    ("The housing shown here would block the view corridors to the mountains.", "Disapprove"),
    ("Traffic on Ocean Park Blvd cannot absorb thousands of new residents.", "Disapprove"),
    ("Is the housing in this scenario affordable or market rate?", "None"),
    ("I could accept 100% affordable housing but nothing else.", "Approve"),
    ("No housing. Museums, gardens, trails — yes. Apartments — no.", "Disapprove"),
    ("The scale of housing in scenario 2 overwhelms the park.", "Disapprove"),
    ("Please study the infrastructure impacts before committing to housing.", "None"),
    ("Housing elsewhere in the city; this land is for everyone.", "Disapprove"),
    ("Some live-work artist housing could give the park character.", "Approve"),
    ("Concerned about shadows from taller housing on the sports fields.", "Disapprove"),
    ("A land trust model for permanently affordable homes would be acceptable.", "Approve"),
    ("The housing crisis is real — this site can be part of the solution.", "Approve"),
    ("Not against housing generally, but against it here.", "Disapprove"),
    ("What happens to the wildlife habitat if housing goes in this corner?", "None"),
    ("Keep every acre for park and recreation. Zero housing.", "Disapprove"),
    ("Family-sized affordable units near the park would keep families in the city.", "Approve"),
    ("This is our Central Park moment. Do not squander it on development.", "Disapprove"),
    ("The proposed housing area sits on land that needs environmental cleanup first.", "None"),
    ("Density belongs downtown, not in the middle of our future park.", "Disapprove"),
    ("Housing revenue could sustain park operations — worth considering.", "Approve"),
]

s3 = [
    ("Scenario 3 has too much development overall, and the housing is the worst part.", "Disapprove"),
    ("Absolutely not. This much housing turns a park into a subdivision.", "Disapprove"),
    ("The community said park first. This scenario ignores that.", "Disapprove"),
    ("I actually like that this scenario faces the housing question honestly.", "Approve"),
    ("If we must have housing, this layout at least keeps it compact.", "Approve"),
    ("Affordable housing integrated with the park could be a national model.", "Approve"),
    ("Measure LC requires voter approval for this — has that been considered?", "None"),
    ("The traffic study for this many units needs to happen before anything else.", "None"),
    ("Too dense. Too tall. Too much. No.", "Disapprove"),
    ("Housing plus revenue uses here feel like a land grab.", "Disapprove"),
    ("Keep the runway area open; put nothing but landscape there.", "Disapprove"),
    ("Support housing only if it is 100% affordable and under community control.", "Approve"),
    ("What about the schools? More housing means more crowded classrooms.", "None"),
    ("This is the least park-like scenario and it shows in the housing footprint.", "Disapprove"),
    ("Would rather see zero housing and modest revenue uses to fund the park.", "Disapprove"),
    ("The housing block placement cuts off the neighborhood connection to the park.", "Disapprove"),
    ("Some housing makes the park safer with eyes on the green at night.", "Approve"),
    ("No more concrete. The city promised a park.", "Disapprove"),
    ("Mixed-income housing on the edge, wild park in the middle — that I'd support.", "Approve"),
    ("This scenario should be dropped entirely.", "Disapprove"),
    ("Prove the infrastructure works first: water, sewer, power, traffic.", "None"),
    ("Our neighborhood already absorbed growth. The park is owed to us.", "Disapprove"),
    ("Please consider senior and workforce housing rather than market-rate.", "Approve"),
    ("Housing here violates the spirit of everything the community asked for.", "Disapprove"),
]


def build(rows, prefix):
    data = []
    for i, (comment, reaction) in enumerate(rows):
        # some respondents leave multiple comments — response IDs can repeat
        rid = f"{prefix}-{1000 + (i if random.random() > 0.18 else max(0, i - 1))}"
        data.append({"comment": comment, "reaction": reaction, "responseId": rid})
    return pd.DataFrame(data)


build(s1, "R1").to_excel("sample_data/HOUSING SCENARIO 1.xlsx", index=False)
build(s2, "R2").to_excel("sample_data/HOUSING SCENARIO 2.xlsx", index=False)
build(s3, "R3").to_excel("sample_data/HOUSING SCENARIO 3.xlsx", index=False)
print("Sample files written to sample_data/")
