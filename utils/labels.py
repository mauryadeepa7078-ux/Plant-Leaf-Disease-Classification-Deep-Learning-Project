"""Single source of truth for the 16 LeafGuard AI classes.

The order of ``CLASSES`` defines the model's output-index order.  Never reorder
this list without retraining the model.

The ``dataset_label`` values are the exact folder names used by PlantVillage.
The shorter ``name`` values are what the website displays.  Keeping both makes
it possible to prove that the project did not invent any dataset categories.
"""

from __future__ import annotations

PLANTS = ["Tomato", "Apple", "Grape", "Corn"]

CLASSES = [
    {
        "index": 0,
        "name": "Tomato Healthy",
        "plant": "Tomato",
        "disease": "Healthy",
        "dataset_label": "Tomato___healthy",
        "healthy": True,
        "summary": "No disease symptoms are visible on this tomato leaf.",
        "description": (
            "A healthy tomato leaf is normally green, flat and free of the "
            "dark spots, powdery growth, yellowing or curling associated with "
            "the diseases in this dataset."
        ),
        "symptoms": "Uniform green colour; no obvious lesions, mould or yellow halos.",
        "cause": "Healthy plant tissue.",
        "prevention": [
            "Water at soil level and keep foliage dry where possible.",
            "Space plants for good airflow and rotate tomato crops.",
            "Monitor leaves weekly and remove seriously damaged foliage.",
        ],
    },
    {
        "index": 1,
        "name": "Tomato Early Blight",
        "plant": "Tomato",
        "disease": "Early Blight",
        "dataset_label": "Tomato___Early_blight",
        "healthy": False,
        "summary": "A common fungal disease caused mainly by Alternaria solani.",
        "description": (
            "Early blight usually begins on older, lower tomato leaves. It "
            "forms brown spots with concentric rings and a yellow halo, and "
            "can spread upward as the season progresses."
        ),
        "symptoms": "Brown concentric-ring lesions, yellow halos, lower-leaf yellowing.",
        "cause": "Fungal pathogen Alternaria solani; favoured by warm, humid conditions.",
        "prevention": [
            "Rotate away from tomato, potato and pepper for at least two seasons.",
            "Remove infected lower leaves and keep debris out of the soil.",
            "Mulch to reduce soil splash and avoid overhead watering.",
            "Use resistant varieties and follow local extension fungicide guidance.",
        ],
    },
    {
        "index": 2,
        "name": "Tomato Late Blight",
        "plant": "Tomato",
        "disease": "Late Blight",
        "dataset_label": "Tomato___Late_blight",
        "healthy": False,
        "summary": "A fast-moving disease caused by the oomycete Phytophthora infestans.",
        "description": (
            "Late blight produces irregular, water-soaked grey-green or brown "
            "lesions that can enlarge rapidly. In humid weather a white "
            "mould-like growth may appear on the underside of leaves."
        ),
        "symptoms": "Water-soaked lesions, dark patches, white underside growth, rapid collapse.",
        "cause": "Phytophthora infestans; cool to moderate, wet weather strongly favours spread.",
        "prevention": [
            "Plant certified disease-free seed and resistant varieties where available.",
            "Improve airflow, avoid leaf wetness and remove volunteer plants.",
            "Destroy infected plants promptly; do not compost symptomatic material.",
            "Apply locally recommended protectant treatments before wet periods.",
        ],
    },
    {
        "index": 3,
        "name": "Tomato Bacterial Spot",
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "dataset_label": "Tomato___Bacterial_spot",
        "healthy": False,
        "summary": "A bacterial disease that causes small dark spots and leaf yellowing.",
        "description": (
            "Bacterial spot appears as small, dark, water-soaked spots that "
            "become brown or black, often with yellow halos. Heavy infection "
            "can cause leaves to yellow and drop."
        ),
        "symptoms": "Small dark spots, yellow halos, scabby fruit lesions in severe cases.",
        "cause": "Xanthomonas species; spread by splashing water, tools and contaminated seed.",
        "prevention": [
            "Use pathogen-free seed and clean tools between plants.",
            "Avoid overhead irrigation and working with wet plants.",
            "Rotate crops and remove infected debris after harvest.",
            "Choose resistant varieties and consult local extension advice for bactericides.",
        ],
    },
    {
        "index": 4,
        "name": "Apple Healthy",
        "plant": "Apple",
        "disease": "Healthy",
        "dataset_label": "Apple___healthy",
        "healthy": True,
        "summary": "No disease symptoms are visible on this apple leaf.",
        "description": (
            "A healthy apple leaf is generally green and unblemished. This "
            "class is the reference for leaves without scab, rot or rust "
            "lesions in the PlantVillage collection."
        ),
        "symptoms": "Green, intact leaf surface without clear lesions or pustules.",
        "cause": "Healthy plant tissue.",
        "prevention": [
            "Prune for light penetration and airflow through the canopy.",
            "Keep the orchard floor clear of fallen leaves and fruit.",
            "Monitor new growth during warm, wet spring weather.",
        ],
    },
    {
        "index": 5,
        "name": "Apple Scab",
        "plant": "Apple",
        "disease": "Apple Scab",
        "dataset_label": "Apple___Apple_scab",
        "healthy": False,
        "summary": "A fungal disease that produces olive to black velvety lesions.",
        "description": (
            "Apple scab lesions begin as olive-green spots and can turn dark "
            "and velvety. Badly infected leaves may pucker, yellow and fall "
            "early. The disease also affects fruit."
        ),
        "symptoms": "Olive-black velvety spots, leaf puckering, premature leaf drop.",
        "cause": "Venturia inaequalis; overwinters in fallen leaves and spreads in wet spring weather.",
        "prevention": [
            "Rake and remove fallen leaves to reduce overwintering spores.",
            "Prune for airflow and avoid prolonged leaf wetness.",
            "Plant scab-resistant apple cultivars where suitable.",
            "Follow a locally validated spring fungicide programme if pressure is high.",
        ],
    },
    {
        "index": 6,
        "name": "Apple Black Rot",
        "plant": "Apple",
        "disease": "Black Rot",
        "dataset_label": "Apple___Black_rot",
        "healthy": False,
        "summary": "A fungal disease causing frog-eye leaf spots and fruit rot.",
        "description": (
            "On leaves, black rot forms purplish-brown circular spots that "
            "develop lighter centres and concentric rings, resembling a frog's "
            "eye. Infected fruit can rot and become mummified."
        ),
        "symptoms": "Purple-brown frog-eye spots, light centres, fruit rot and mummies.",
        "cause": "Botryosphaeria obtusa; survives in cankers, mummified fruit and dead wood.",
        "prevention": [
            "Remove mummified fruit, dead wood and cankers during pruning.",
            "Improve tree vigour with appropriate water and nutrition.",
            "Keep fruit and foliage dry where practical.",
            "Use local extension guidance for fungicide timing in high-risk orchards.",
        ],
    },
    {
        "index": 7,
        "name": "Apple Cedar Apple Rust",
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "dataset_label": "Apple___Cedar_apple_rust",
        "healthy": False,
        "summary": "A rust disease that needs both apple and juniper/cedar hosts.",
        "description": (
            "Cedar apple rust causes bright yellow-orange spots on apple "
            "leaves. The underside may develop small raised structures that "
            "release orange spores. The pathogen completes part of its life "
            "cycle on junipers and other cedar hosts."
        ),
        "symptoms": "Yellow-orange leaf spots, orange spore structures on the underside.",
        "cause": "Gymnosporangium juniperi-virginianae; alternate host is eastern red cedar/juniper.",
        "prevention": [
            "Reduce nearby juniper/cedar hosts where practical and legal.",
            "Plant rust-resistant apple cultivars.",
            "Prune to improve airflow and monitor during wet spring weather.",
            "Apply locally recommended protectant sprays at the correct growth stage.",
        ],
    },
    {
        "index": 8,
        "name": "Grape Healthy",
        "plant": "Grape",
        "disease": "Healthy",
        "dataset_label": "Grape___healthy",
        "healthy": True,
        "summary": "No disease symptoms are visible on this grape leaf.",
        "description": (
            "A healthy grape leaf is green and fully expanded, without the "
            "necrotic patches, yellow halos or dark fruiting bodies associated "
            "with the grape diseases in this dataset."
        ),
        "symptoms": "Even green colour, intact leaf margins, no obvious lesions.",
        "cause": "Healthy plant tissue.",
        "prevention": [
            "Open the canopy with pruning so leaves dry quickly.",
            "Remove fallen leaves and manage weeds around the vine.",
            "Inspect new growth regularly during humid periods.",
        ],
    },
    {
        "index": 9,
        "name": "Grape Black Rot",
        "plant": "Grape",
        "disease": "Black Rot",
        "dataset_label": "Grape___Black_rot",
        "healthy": False,
        "summary": "A fungal disease that damages leaves and shrivels grape berries.",
        "description": (
            "Grape black rot leaf spots are tan or brown with dark margins. "
            "Tiny black fruiting bodies may be visible inside the lesion. The "
            "disease is especially destructive to fruit, which shrivels into "
            "black mummies."
        ),
        "symptoms": "Tan lesions with dark borders, black dots, shrivelled black berries.",
        "cause": "Guignardia bidwellii; overwinters in mummified berries and infected canes.",
        "prevention": [
            "Remove mummified berries and infected canes during dormant pruning.",
            "Keep the canopy open and avoid prolonged leaf wetness.",
            "Manage weeds and fallen debris around the vineyard.",
            "Use locally recommended fungicides from early season where pressure is high.",
        ],
    },
    {
        "index": 10,
        "name": "Grape Esca",
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "dataset_label": "Grape___Esca_(Black_Measles)",
        "healthy": False,
        "summary": "A wood disease complex associated with several fungal pathogens.",
        "description": (
            "Esca is a grapevine trunk disease. Leaves may show interveinal "
            "yellowing or browning that forms a tiger-stripe pattern, and "
            "affected vines can decline or die. Fruit may show dark spotting "
            "known as black measles."
        ),
        "symptoms": "Tiger-stripe leaf necrosis, interveinal chlorosis, vine decline.",
        "cause": "A complex of wood-infecting fungi; pruning wounds are an important entry point.",
        "prevention": [
            "Use clean planting material and protect large pruning wounds.",
            "Prune during dry weather and disinfect tools between vines.",
            "Remove severely declining vines and avoid carrying wood debris through the vineyard.",
            "Consult a viticulture adviser; trunk diseases are difficult to cure once established.",
        ],
    },
    {
        "index": 11,
        "name": "Grape Leaf Blight",
        "plant": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "dataset_label": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "healthy": False,
        "summary": "A fungal leaf spot disease also called Isariopsis leaf spot.",
        "description": (
            "Grape leaf blight causes brown, irregular or angular lesions, "
            "often with yellow margins. Severe infections can lead to early "
            "defoliation and reduce vine vigour."
        ),
        "symptoms": "Brown angular lesions, yellow halos, premature leaf drop.",
        "cause": "Pseudocercospora/Isariopsis fungal complex; warm, humid conditions favour it.",
        "prevention": [
            "Remove infected leaves and vine debris after harvest.",
            "Improve canopy airflow and avoid overhead irrigation.",
            "Keep vines adequately nourished but avoid excessive nitrogen.",
            "Use locally approved fungicides when scouting indicates a need.",
        ],
    },
    {
        "index": 12,
        "name": "Corn Healthy",
        "plant": "Corn",
        "disease": "Healthy",
        "dataset_label": "Corn_(maize)___healthy",
        "healthy": True,
        "summary": "No disease symptoms are visible on this corn leaf.",
        "description": (
            "A healthy corn leaf is green and upright with an intact surface. "
            "It lacks the elongated lesions and raised pustules that "
            "characterise the corn diseases included in this project."
        ),
        "symptoms": "Green, intact leaf with no clear necrotic lesions or pustules.",
        "cause": "Healthy plant tissue.",
        "prevention": [
            "Rotate corn with non-host crops and manage crop residue.",
            "Choose regionally adapted hybrids with good disease resistance.",
            "Scout fields regularly, especially after warm, humid weather.",
        ],
    },
    {
        "index": 13,
        "name": "Corn Common Rust",
        "plant": "Corn",
        "disease": "Common Rust",
        "dataset_label": "Corn_(maize)___Common_rust_",
        "healthy": False,
        "summary": "A rust disease that forms cinnamon-brown pustules on corn leaves.",
        "description": (
            "Common rust produces small, round-to-elongate cinnamon-brown "
            "pustules on both sides of corn leaves. The pustules release rusty "
            "spores when rubbed. Severe outbreaks can reduce photosynthesis."
        ),
        "symptoms": "Cinnamon-brown raised pustules scattered across the leaf surface.",
        "cause": "Puccinia sorghi; wind-borne spores and cool, humid conditions favour infection.",
        "prevention": [
            "Plant resistant or moderately resistant corn hybrids.",
            "Avoid excessive nitrogen and maintain balanced fertility.",
            "Scout from early vegetative stages and record disease pressure.",
            "Follow regional extension thresholds before considering fungicide.",
        ],
    },
    {
        "index": 14,
        "name": "Corn Northern Leaf Blight",
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "dataset_label": "Corn_(maize)___Northern_Leaf_Blight",
        "healthy": False,
        "summary": "A fungal disease that forms long, cigar-shaped grey-green lesions.",
        "description": (
            "Northern corn leaf blight begins as small oval spots that grow "
            "into long, cigar-shaped grey-green or tan lesions. Heavy infection "
            "can destroy large areas of leaf tissue and reduce yield."
        ),
        "symptoms": "Long cigar-shaped grey-green/tan lesions, often starting on lower leaves.",
        "cause": "Exserohilum turcicum; survives in residue and spreads during moderate, wet weather.",
        "prevention": [
            "Rotate crops and manage infected crop residue.",
            "Plant hybrids with good northern-leaf-blight resistance.",
            "Scout lower leaves before tasselling and follow local thresholds.",
            "Use fungicides only where extension advice indicates a likely economic benefit.",
        ],
    },
    {
        "index": 15,
        "name": "Corn Cercospora Leaf Spot",
        "plant": "Corn",
        "disease": "Cercospora Leaf Spot (Gray Leaf Spot)",
        "dataset_label": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "healthy": False,
        "summary": "A residue-borne leaf spot disease grouped with gray leaf spot in PlantVillage.",
        "description": (
            "This class uses PlantVillage's combined Cercospora leaf spot / "
            "gray leaf spot folder. Lesions are typically narrow, rectangular "
            "and tan to grey, running between leaf veins, and may merge during "
            "severe infections. The public dataset does not separate the two "
            "visually similar labels, so LeafGuard AI reports the combined "
            "class honestly rather than inventing a split."
        ),
        "symptoms": "Narrow rectangular tan-grey lesions between veins; yellowing and blighting.",
        "cause": "Cercospora and related residue-borne fungi; warm, humid, continuous-corn conditions favour them.",
        "prevention": [
            "Rotate away from corn and manage residue to reduce inoculum.",
            "Plant hybrids with resistance to gray leaf spot/Cercospora complex.",
            "Improve field airflow and avoid excessive plant density.",
            "Scout upper leaves around tasselling and follow local fungicide thresholds.",
        ],
    },
]

# Convenience maps used by training, evaluation and the web app.
CLASS_NAMES = [item["name"] for item in CLASSES]
CLASS_TO_INDEX = {item["name"]: item["index"] for item in CLASSES}
INDEX_TO_CLASS = {item["index"]: item for item in CLASSES}
DATASET_LABEL_TO_INDEX = {item["dataset_label"]: item["index"] for item in CLASSES}
DATASET_LABEL_TO_CLASS = {item["dataset_label"]: item for item in CLASSES}
PLANT_CLASSES = {plant: [item for item in CLASSES if item["plant"] == plant] for plant in PLANTS}

# A small, reusable sample of what the website shows on category cards.
PLANT_META = {
    "Tomato": {
        "latin": "Solanum lycopersicum",
        "tagline": "Four common foliar conditions",
        "colour": "#c2410c",
        "soft": "#fff4ed",
    },
    "Apple": {
        "latin": "Malus domestica",
        "tagline": "Orchard leaf health check",
        "colour": "#b91c1c",
        "soft": "#fef2f2",
    },
    "Grape": {
        "latin": "Vitis vinifera",
        "tagline": "Vineyard disease screening",
        "colour": "#6d28d9",
        "soft": "#f5f3ff",
    },
    "Corn": {
        "latin": "Zea mays",
        "tagline": "Field-crop leaf diagnosis",
        "colour": "#a16207",
        "soft": "#fefce8",
    },
}

IMAGE_SIZE = (224, 224)
LOW_CONFIDENCE_THRESHOLD = 0.60
LOW_MARGIN_THRESHOLD = 0.15


def class_name_for_dataset_label(label: str) -> str:
    """Map a PlantVillage folder name to the display label used in the model."""
    try:
        return DATASET_LABEL_TO_CLASS[label]["name"]
    except KeyError as exc:
        raise ValueError(f"Unsupported PlantVillage class: {label}") from exc


def validate_taxonomy() -> None:
    """Fail loudly if the promised 4 plants x 4 classes are not present."""
    if len(CLASSES) != 16:
        raise ValueError(f"Expected 16 classes, found {len(CLASSES)}")
    for plant in PLANTS:
        count = len(PLANT_CLASSES[plant])
        if count < 4:
            raise ValueError(f"{plant} has only {count} classes; at least 4 required")
    if len({item["dataset_label"] for item in CLASSES}) != 16:
        raise ValueError("Duplicate PlantVillage dataset labels in CLASSES")


validate_taxonomy()
