from typing import Any, Dict, List

# RLVS Evaluation Queries
RLVS_QUERIES = [
    {
        "query": "violent physical fight between people on the street",
        "target_class": "violence",
        "positive_classes": ["violence"],
    },
    {
        "query": "aggressive street brawl with punching and physical altercation",
        "target_class": "violence",
        "positive_classes": ["violence"],
    },
    {
        "query": "physical violence assault and mob struggle",
        "target_class": "violence",
        "positive_classes": ["violence"],
    },
    {
        "query": "people walking peacefully and having a normal conversation",
        "target_class": "non_violence",
        "positive_classes": ["non_violence"],
    },
    {
        "query": "calm public street activity with ordinary pedestrians",
        "target_class": "non_violence",
        "positive_classes": ["non_violence"],
    },
    {
        "query": "non-violent benign outdoor movement without fighting",
        "target_class": "non_violence",
        "positive_classes": ["non_violence"],
    },
]

# UCF-ARG Evaluation Queries (10 Actions)
UCF_ARG_QUERIES = [
    {"query": "person boxing and throwing punches", "target_class": "boxing", "positive_classes": ["boxing"]},
    {"query": "individual carrying an object or box while moving", "target_class": "carrying", "positive_classes": ["carrying"]},
    {"query": "person clapping both hands together rhythmically", "target_class": "clapping", "positive_classes": ["clapping"]},
    {"query": "person digging into the ground with a tool", "target_class": "digging", "positive_classes": ["digging"]},
    {"query": "individual jogging steadily across open space", "target_class": "jogging", "positive_classes": ["jogging"]},
    {"query": "person opening and closing a vehicle trunk", "target_class": "openclosetrunk", "positive_classes": ["openclosetrunk"]},
    {"query": "person running quickly through the surveillance area", "target_class": "running", "positive_classes": ["running"]},
    {"query": "person throwing an object with an overhead motion", "target_class": "throwing", "positive_classes": ["throwing"]},
    {"query": "person walking at a regular pace", "target_class": "walking", "positive_classes": ["walking"]},
    {"query": "individual raising hands and waving in the air", "target_class": "waving", "positive_classes": ["waving"]},
]

# TinyVIRAT-v2 Evaluation Queries
TINYVIRAT_QUERIES = [
    {"query": "person walking slowly across the camera view", "target_class": "activity_walking", "positive_classes": ["activity_walking", "Walking"]},
    {"query": "person running rapidly in surveillance area", "target_class": "activity_running", "positive_classes": ["activity_running", "Running"]},
    {"query": "person carrying a heavy bag or object", "target_class": "activity_carrying", "positive_classes": ["activity_carrying", "Carrying"]},
    {"query": "vehicle driving and moving through the scene", "target_class": "vehicle_moving", "positive_classes": ["vehicle_moving"]},
    {"query": "vehicle coming to a complete stop", "target_class": "vehicle_stopping", "positive_classes": ["vehicle_stopping"]},
    {"query": "vehicle making a left turn", "target_class": "vehicle_turning_left", "positive_classes": ["vehicle_turning_left"]},
    {"query": "vehicle making a right turn", "target_class": "vehicle_turning_right", "positive_classes": ["vehicle_turning_right"]},
    {"query": "person opening a door or vehicle entrance", "target_class": "Opening", "positive_classes": ["Opening"]},
    {"query": "person closing a door or container", "target_class": "Closing", "positive_classes": ["Closing"]},
    {"query": "person entering a building or vehicle", "target_class": "Entering", "positive_classes": ["Entering"]},
    {"query": "person exiting a facility or vehicle", "target_class": "Exiting", "positive_classes": ["Exiting"]},
    {"query": "person standing still and observing", "target_class": "activity_standing", "positive_classes": ["activity_standing"]},
    {"query": "two or more people talking and interacting", "target_class": "Talking", "positive_classes": ["Talking"]},
]


# UCF-Crime Evaluation Queries (14 Anomaly Classes + Normal)
UCF_CRIME_QUERIES = [
    {"query": "physical abuse and domestic violence assault", "target_class": "abuse", "positive_classes": ["abuse"]},
    {"query": "police officers arresting a suspect with handcuffs", "target_class": "arrest", "positive_classes": ["arrest"]},
    {"query": "arson fire deliberately set in building or structure", "target_class": "arson", "positive_classes": ["arson"]},
    {"query": "violent physical assault on an individual", "target_class": "assault", "positive_classes": ["assault"]},
    {"query": "burglary breaking and entering commercial or residential property", "target_class": "burglary", "positive_classes": ["burglary"]},
    {"query": "violent explosion blast and fireball destruction", "target_class": "explosion", "positive_classes": ["explosion"]},
    {"query": "street brawl and physical fighting between people", "target_class": "fighting", "positive_classes": ["fighting"]},
    {"query": "normal surveillance activity with peaceful pedestrian movement", "target_class": "normal", "positive_classes": ["normal"]},
    {"query": "severe road accident vehicle collision and crash", "target_class": "roadaccidents", "positive_classes": ["roadaccidents"]},
    {"query": "armed robbery stealing from store or cashier at gunpoint", "target_class": "robbery", "positive_classes": ["robbery"]},
    {"query": "active shooter gunfire shooting in public facility", "target_class": "shooting", "positive_classes": ["shooting"]},
    {"query": "shoplifting stealing merchandise from retail store", "target_class": "shoplifting", "positive_classes": ["shoplifting"]},
    {"query": "stealing unattended property bicycle or belongings", "target_class": "stealing", "positive_classes": ["stealing"]},
    {"query": "vandalism damaging public property or breaking windows", "target_class": "vandalism", "positive_classes": ["vandalism"]},
]


def get_queries_for_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    """Returns curated natural-language evaluation queries for a given dataset."""
    dname = dataset_name.lower().replace("-", "_")
    if "crime" in dname:
        return UCF_CRIME_QUERIES
    if "rlvs" in dname:
        return RLVS_QUERIES
    if "arg" in dname:
        return UCF_ARG_QUERIES
    if "tiny" in dname or "virat" in dname:
        return TINYVIRAT_QUERIES
    return RLVS_QUERIES
