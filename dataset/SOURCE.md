# Dataset source, verification and licence

## PlantVillage

| Item | Detail |
|---|---|
| Dataset | PlantVillage, colour (unsegmented) images |
| Original repository | <https://github.com/spMohanty/PlantVillage-Dataset> |
| Hugging Face mirror | <https://huggingface.co/datasets/mohanty/PlantVillage> |
| Paper | Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for Image-Based Plant Disease Detection.* Frontiers in Plant Science, 7, 1419. <https://doi.org/10.3389/fpls.2016.01419> |
| Licence declared by the mirror | **CC BY-SA 3.0 Unported** (<https://creativecommons.org/licenses/by-sa/3.0/>) |
| Images used | 4,000 (250 per class, 16 classes) |
| Image format | JPEG, 256 × 256 source images |
| Download method | Deterministic sample from the official `raw/color` tree (`scripts/download_dataset.py`) |

The 4,000 files were downloaded from the dataset authors' GitHub repository.
The download script seeds Python's random generator with `42`, sorts each
folder's filenames, and samples 250 images per class. It fails loudly if a
folder is missing or has fewer than 250 exposed files. No class was invented,
renamed in the source tree, or silently substituted.

## Verified class folders

Every folder below was confirmed to exist in the public repository before use:

```text
Apple___Apple_scab
Apple___Black_rot
Apple___Cedar_apple_rust
Apple___healthy
Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
Corn_(maize)___Common_rust_
Corn_(maize)___Northern_Leaf_Blight
Corn_(maize)___healthy
Grape___Black_rot
Grape___Esca_(Black_Measles)
Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
Grape___healthy
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
Tomato___healthy
```

That is exactly four classes for each of tomato, apple, grape and corn.

## Corn class naming note

PlantVillage groups **Cercospora leaf spot** and **gray leaf spot** in a single
folder, `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`. LeafGuard AI
displays the brief's short name **Corn Cercospora Leaf Spot**, but its About
page and model metadata explicitly state that the public dataset combines the
two visually similar diseases. The project does not claim to distinguish them.

## Attribution and share-alike

The images remain the property of their original contributors and are used
under CC BY-SA 3.0. If you redistribute the images or a derived dataset:

1. Keep the attribution and source links above.
2. Share adaptations of the dataset under the same CC BY-SA 3.0 licence.
3. Do not imply that the dataset authors endorse this project.

The LeafGuard AI source code is separately licensed under the MIT Licence; the
dataset licence governs the images, not the code.

## Dataset limitations

PlantVillage images were collected under controlled conditions, with leaves
placed against mostly plain backgrounds. Performance on these images does not
represent performance on field photographs with complex backgrounds, variable
lighting, occlusion, multiple leaves or different cultivars. See the project
README and About page for the full limitations section.
