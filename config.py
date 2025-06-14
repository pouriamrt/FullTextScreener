import os
from dotenv import load_dotenv

load_dotenv()

# INCLUSION_CRITERIA = [
#     """
#     Population - Populations at risk of developing an NCD (Non-Communicable Diseases)
#     Included are studies of a real or generalizable human population or subpopulation defined by geographic, demographic or social characteristics (e.g., national population, region, age group, socioeconomic group). Excluded are studies where the population consists exclusively of individuals already diagnosed with an NCD (e.g., diabetes, cancer, cardiovascular disease).
#     """,
#     """
#     Intervention, exposure, or scenario (includes comparator) 
#     Included are studies evaluating potential health impacts of exposures, interventions or policies on NCD (Non-Communicable Diseases) outcomes by simulating "what if" scenarios. These include burden-of-disease studies, comparative risk analysis/assessments, and studies of primary or secondary prevention. Excluded are studies focused exclusively on tertiary prevention or methodological development without assessing specific health impact scenarios. 
#     """,
#     """
#     Outcome - Selected non-communicable diseases (NCDs - Non-Communicable Diseases) or NCD risk factors 
#     Included are studies reporting outcomes related to selected major NCDs (cardiovascular diseases, cancer, diabetes, chronic respiratory diseases, mental health conditions, or neurological conditions, injury and musculoskeletal diseases) or their risk factors. Excluded are studies primarily designed as health economic evaluations (e.g., cost-effectiveness analyses) or those focusing exclusively on non-NCD conditions without NCD outcomes.
#     """,
#     """
#     Study approach - Computational simulation modelling 
#     Included are studies using computational simulation modelling methods as the primary analytical tool, such as system dynamics models, agent-based models, microsimulation, Markov models, or attributable risk models. Excluded are observational studies, prediction models for individual risk, statistical regression models, or trend forecasting models that don't simulate interventions. 
#     """,
#     """
#     Integration of multiple data sources 
#     Studies typically integrate multiple distinct data sources, including empirical data, expert opinion or literature-based estimates of model parameters such as causal risk-outcome relationships or intervention effectiveness. 
#     """,
#     """
#     Model adaptability 
#     The model should be adaptable to incorporate new data, test alternative hypotheses or explore additional scenarios.
#     """
# ]

INCLUSION_CRITERIA = [
    """
    Population - Populations at risk of developing an NCD (Non-Communicable Diseases)
    Studies of real or generalizable human populations at risk of developing NCDs. These may be defined by geographic, demographic, or social characteristics (e.g., national population, region, age group).
    """,
    """
    Intervention, exposure, or scenario (includes comparator) 
    Studies evaluating health impacts of exposures, interventions, or policies on NCD outcomes through simulation of hypothetical scenarios. Includes burden-of-disease and comparative risk assessment studies.
    """,
    """
    Outcome - Selected non-communicable diseases (NCDs - Non-Communicable Diseases) or NCD risk factors 
    Studies reporting on outcomes related to major NCDs (e.g., cardiovascular disease, cancer, diabetes, chronic respiratory diseases, mental health, neurological disorders, injury, or musculoskeletal conditions) or their risk factors.
    """,
    """
    Study approach - Computational simulation modelling 
    Studies that use computational simulation modeling (e.g., system dynamics, agent-based models, microsimulation, Markov models, or attributable risk models) as the primary method of analysis.
    """,
    """
    Integration of multiple data sources
    Studies that integrate multiple data sources (e.g., empirical data, expert opinion, literature-based estimates) for model parameters or intervention effects.
    """,
    """
    Model adaptability
    Models that can be adapted to incorporate new data, test different hypotheses, or simulate alternative scenarios.
    """
]

EXCLUSION_CRITERIA = [
    """
    Studies focusing exclusively on individuals already diagnosed with NCDs or using highly specific clinical cohorts without generalizability.
    """,
    """
    Studies that focus exclusively on tertiary prevention or do not simulate hypothetical scenarios (e.g., purely observational studies).
    """,
    """
    Studies that are primarily health economic evaluations (e.g., cost-effectiveness analyses) or those focusing only on non-NCD outcomes.
    """,
    """
    Studies using statistical regression, observational methods, trend forecasts, or risk prediction models that do not simulate interventions.
    """,
    """
    Studies using only one data source or models that lack empirical grounding (e.g., purely hypothetical assumptions).
    """,
    """
    Models that are not designed to be updated or modified for different scenarios or new data inputs.
    """
]


CRITERIA_COLORS = {
    0: (1, 0.9, 0),      # Yellow
    1: (0.6, 1, 0.6),    # Light green
    2: (0.8, 0.9, 1),    # Light blue
    3: (1, 0.6, 0.6),    # Light red
    4: (0.9, 0.9, 0.6),  # Light yellow
    5: (0.6, 0.6, 1),    # Light purple
}

CRITERIA_LABELS = {
    0: "Population",
    1: "Intervention",
    2: "Outcome",
    3: "Study approach",
    4: "Integration of multiple data sources",
    5: "Model adaptability"
}

OPENAI_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1-mini"
# CHUNK_SIZE = 100
# OVERLAP = 10
SENTENCES_PER_CHUNK = 4
SENTENCES_OVERLAP = 1
# SIMILARITY_THRESHOLD = 0.5
# SIMILARITY_THRESHOLDS = [0.4, 0.45, 0.42, 0.48, 0.37, 0.37]
SIMILARITY_THRESHOLDS = [0.05, 0.05, 0.05, 0.05, 0.13, 0.1]
MU_CUTOFF = 0.4
PDF_FOLDER = "data/papers"
OUTPUT_FOLDER = "data/output"
PLOT_FOLDER = "data/plots"
OVERWRITE = True
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
