import ktrain
from cached_path import cached_path

_MODEL_PATH = (
    "https://github.com/PathwayCommons/pathway-abstract-classifier/releases/download/"
    "pretrained-models/title_abstract_model.zip"
)


def test_model_load_and_predict():
    # Load model
    model_path = cached_path(_MODEL_PATH, extract_archive=True)
    model = ktrain.load_predictor(model_path)

    # Should return 0
    assert (model.predict("Testing, testing")) == 0

    # First article should return 1, second 0
    titles = [
        (
            "YTHDC1-mediated augmentation of miR-30d in repressing pancreatic tumorigenesis via"
            " attenuation of RUNX1-induced transcriptional activation of Warburg effect"
        ),
        "Loss of 15-lipoxygenase disrupts T reg differentiation altering their pro-resolving functions",
    ]

    abstracts = [
        (
            "Pancreatic ductal adenocarcinoma (PDAC) is one of the most lethal human cancers. It"
            " thrives in a malnourished environment; however, little is known about the mechanisms by"
            " which PDAC cells actively promote aerobic glycolysis to maintain their metabolic needs."
            " Gene Expression Omnibus (GEO) was used to identify differentially expressed miRNAs. The"
            " expression pattern of miR-30d in normal and PDAC tissues was studied by in situ"
            " hybridization. The role of miR-30d/RUNX1 in vitro and in vivo was evaluated by CCK8 assay"
            " and clonogenic formation as well as transwell experiment, subcutaneous xenograft model"
            " and liver metastasis model, respectively. Glucose uptake, ATP and lactate production were"
            " tested to study the regulatory effect of miR-30d/RUNX1 on aerobic glycolysis in PDAC"
            " cells. Quantitative real-time PCR, western blot, Chip assay, promoter luciferase activity,"
            " RIP, MeRIP, and RNA stability assay were used to explore the molecular mechanism of"
            " YTHDC1/miR-30d/RUNX1 in PDAC. Here, we discover that miR-30d expression was remarkably"
            " decreased in PDAC tissues and associated with good prognosis, contributed to the"
            " suppression of tumor growth and metastasis, and attenuation of Warburg effect."
            " Mechanistically, the m6A reader YTHDC1 facilitated the biogenesis of mature miR-30d via"
            " m6A-mediated regulation of mRNA stability. Then, miR-30d inhibited aerobic glycolysis"
            " through regulating SLC2A1 and HK1 expression by directly targeting the transcription"
            " factor RUNX1, which bound to the promoters of the SLC2A1 and HK1 genes. Moreover, miR-30d"
            " was clinically inversely correlated with RUNX1, SLC2A1 and HK1, which function as adverse"
            " prognosis factors for overall survival in PDAC tissues. Overall, we demonstrated that"
            " miR-30d is a functional and clinical tumor-suppressive gene in PDAC. Our findings further"
            " uncover that miR-30d is a novel target for YTHDC1 through m6A modification, and miR-30d"
            " represses pancreatic tumorigenesis via suppressing aerobic glycolysis.",
        ),
        (
            "Regulatory T-cells (Tregs) are central in the maintenance of homeostasis and resolution"
            " of inflammation. However, the mechanisms that govern their differentiation and function"
            " are not completely understood. Herein, we demonstrate a central role for the lipid"
            " mediator biosynthetic enzyme 15-lipoxygenase (ALOX15) in regulating key aspects of Treg"
            " biology. Pharmacological inhibition or genetic deletion of ALOX15 in Tregs decreased"
            " FOXP3 expression, altered Treg transcriptional profile and shifted their metabolism."
            " This was linked with an impaired ability of Alox15-deficient cells to exert their"
            " pro-resolving actions, including a decrease in their ability to upregulate macrophage"
            " efferocytosis and a downregulation of interferon gamma expression in Th1 cells."
            " Incubation of Tregs with the ALOX15-derived specilized pro-resolving mediators"
            " (SPM)s Resolvin (Rv)D3 and RvD5n-3 DPA rescued FOXP3 expression in cells where ALOX15"
            " activity was inhibited. In vivo, deletion of Alox15 led to increased vascular lipid load"
            " and expansion of Th1 cells in mice fed western diet, a phenomenon that was reversed when"
            " Alox15-deficient mice were reconstituted with wild type Tregs. Taken together these"
            " findings demonstrate a central role of pro-resolving lipid mediators in governing the"
            " differentiation of naive T-cells to Tregs."
        ),
    ]

    sep_token = model.preproc.get_tokenizer().sep_token
    texts = [" ".join([title, sep_token, abstract]) for title, abstract in zip(titles, abstracts)]
    predictions = model.predict(texts)
    assert predictions == [1, 0]

    # Prints if execution was not stopped by an assert statement
    print("Articles correctly classified")


# Tests to see that dependencies and model were installed and loaded correctly.
# Also verifies that Quickstart code works.
if __name__ == "__main__":
    test_model_load_and_predict()
