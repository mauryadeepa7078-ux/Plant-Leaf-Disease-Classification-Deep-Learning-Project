"""The 16-class taxonomy is a hard requirement; test it explicitly."""

from utils.labels import CLASSES, CLASS_NAMES, DATASET_LABEL_TO_INDEX, PLANTS, PLANT_CLASSES


def test_exactly_sixteen_classes():
    assert len(CLASSES) == 16
    assert len(CLASS_NAMES) == 16
    assert len(set(CLASS_NAMES)) == 16


def test_four_plants_with_four_classes_each():
    assert PLANTS == ["Tomato", "Apple", "Grape", "Corn"]
    for plant in PLANTS:
        assert len(PLANT_CLASSES[plant]) == 4, plant


def test_all_dataset_folders_are_unique():
    labels = [item["dataset_label"] for item in CLASSES]
    assert len(labels) == len(set(labels)) == 16
    assert len(DATASET_LABEL_TO_INDEX) == 16


def test_required_dataset_folders_exist_in_taxonomy():
    required = {
        "Tomato___healthy",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Bacterial_spot",
        "Apple___healthy",
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Grape___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Corn_(maize)___healthy",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    }
    assert set(DATASET_LABEL_TO_INDEX) == required


def test_indices_are_contiguous():
    assert [item["index"] for item in CLASSES] == list(range(16))
